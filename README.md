# Time-Series Forecasting — Stock Next-Day Close Price Prediction

A regression pipeline that predicts the **next trading day's close price** for Indian equities using engineered features from historical OHLCV data. Ten models are trained, tuned with time-series cross-validation, and ranked by directional accuracy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Pipeline and Integrity](#data-pipeline-and-integrity)
- [Bug Fixes Applied](#bug-fixes-applied)
- [Setup and Usage](#setup-and-usage)
- [Outputs](#outputs)
- [Limitations and Honest Assessment](#limitations-and-honest-assessment)

---

## Project Overview

**Goal:** Given a stock's historical price/volume data up to today's market close, predict tomorrow's closing price.

**Target variable:** `Target = Close Price.shift(-1)` — the next trading day's close price.

**Prediction boundary:** At row *t*, the model knows everything up to and including day *t* (today's open, high, low, close, volume). It predicts day *t+1*.

**Datasets:**

| File | Stock Code | Rows | Date Range |
|------|-----------|------|------------|
| `500209_combined.csv` | 500209 (Infosys) | ~1187 | 2021–2026 (5 years, combined) |
| `500510.csv` | 500510 | ~247 | 2025–2026 (1 year) |
| `532174.csv` | 532174 | ~247 | 2025–2026 (1 year) |

> **Note:** `500209.csv` (1-year subset) was removed from the pipeline — it is a strict subset of `500209_combined.csv` and was causing duplicate analysis with unreliably small data.

---

## Architecture

The codebase follows a **single-responsibility** design. Each module does one job:

```
main.py              — Orchestrator: runs the full pipeline for each dataset
├── config.py        — All constants, hyperparameter grids, file paths
├── data_loader.py   — Load CSV, parse dates, clean, detect date-gap boundaries
├── features.py      — All feature engineering (lags, rolling, returns, technicals, calendar)
├── splitter.py      — Chronological 80/20 train/test split (no shuffle)
├── models.py        — 10 model definitions + hyperparameter tuning with TimeSeriesSplit
├── benchmark.py     — Train all models, collect metrics, build comparison table
├── evaluate.py      — MAE, RMSE, MAPE, R², directional accuracy
└── visualize.py     — Price history, correlation heatmap, prediction overlays, feature importance
```

### Data Flow

```
CSV → data_loader.load_and_clean()
    → features.build_features()         # 100+ engineered features
    → features.select_features()         # mutual-information feature selection
    → splitter.chronological_split()     # 80% train / 20% test
    → benchmark.run_benchmark()          # train, predict, evaluate all models
    → outputs/                           # plots, CSVs, benchmark tables
```

---

## Feature Engineering

All features use data available at prediction time (up to and including today's close). The target is `Close Price.shift(-1)`, which creates a natural 1-day buffer — **no look-ahead leakage is possible by using today's data**.

### Feature Groups

| Group | Count | Description |
|-------|-------|-------------|
| **Raw OHLCV** | ~9 | Today's Open, High, Low, Close, WAP, Volume, Trades, Spreads |
| **Lag features** | ~45 | Price/volume values from 1, 2, 3, 5, 10 days ago |
| **Rolling statistics** | ~28 | Rolling mean, std, min, max, range% over 3/5/10/20-day windows for close and volume |
| **Return features** | ~7 | Daily return, log return, 3/5/10-day momentum, 5/10-day realized volatility |
| **Technical indicators** | ~25 | SMA (5/10/20/50), EMA (5/10/20), RSI-14, MACD/signal/histogram, Bollinger Bands, ATR, Stochastic %K/%D, price-vs-SMA ratios, crossover signals |
| **Calendar features** | 3 | Day of week, month, quarter |

### Feature Selection

An adaptive mutual-information selector caps the feature count at `max(15, train_rows / 5)` to prevent overfitting on small datasets. Selection is computed on the training portion only to avoid leakage.

---

## Models

Ten models are trained and tuned, spanning linear, kernel, tree, boosting, and ensemble methods:

| Model | Type | Tuning Method | Key Parameters |
|-------|------|--------------|----------------|
| **Ridge** | Linear (L2) | GridSearchCV | `alpha` |
| **Lasso** | Linear (L1) | GridSearchCV | `alpha` |
| **ElasticNet** | Linear (L1+L2) | GridSearchCV | `alpha`, `l1_ratio` |
| **SVR** | Kernel | GridSearchCV | `C`, `epsilon`, `kernel` |
| **Random Forest** | Bagging ensemble | RandomizedSearchCV | `n_estimators`, `max_depth`, `min_samples_leaf` |
| **Gradient Boosting** | Sequential boosting | RandomizedSearchCV | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| **XGBoost** | Regularized boosting | RandomizedSearchCV | `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree`, `reg_alpha/lambda` |
| **LightGBM** | Leaf-wise boosting | RandomizedSearchCV | `n_estimators`, `num_leaves`, `learning_rate`, `colsample_bytree` |
| **AdaBoost** | Adaptive boosting | RandomizedSearchCV | `n_estimators`, `learning_rate`, `estimator__max_depth` |
| **Stacking** | Meta-ensemble | Fixed config | Ridge + RF + XGB + LGBM → Ridge meta-learner |

All tuning uses **`TimeSeriesSplit(n_splits=5)`** — never random shuffling — to respect temporal ordering. The stacking ensemble also uses `TimeSeriesSplit` for generating out-of-fold predictions.

A **Naive Baseline** (predict tomorrow = today's close) is always included. Any model that fails to beat the naive baseline on directional accuracy likely has no genuine predictive signal.

---

## Evaluation Metrics

| Metric | What it measures | Better |
|--------|-----------------|--------|
| **MAE** | Mean Absolute Error (in ₹) | Lower |
| **RMSE** | Root Mean Squared Error (penalizes large errors) | Lower |
| **MAPE%** | Mean Absolute Percentage Error | Lower |
| **R²** | Variance explained (1.0 = perfect) | Higher |
| **Dir Acc%** | Directional accuracy vs previous day's actual close | Higher |

### Directional Accuracy Details

- Measures whether the model correctly predicts the **direction** of price movement (up or down) relative to the previous day's actual close.
- **Tie policy:** If the model predicts exactly the previous close (no directional information), it scores **0.5** instead of 0 — reflecting "no information" rather than "actively wrong."
- **Flat days excluded:** Days where the actual price did not move are excluded from the denominator, since there is no correct directional answer.
- The naive baseline gets an honest **50% floor** under this policy.

---

## Data Pipeline and Integrity

### Date-Gap Detection

The `500209_combined.csv` file is stitched from multiple per-year CSVs. This creates **multi-week gaps** at year boundaries where the source files don't overlap:

- **Jun 2 → Jul 4, 2022** (32-day gap)
- **Jun 7 → Jul 3, 2023** (26-day gap)

The data loader now **automatically detects** gaps > 5 calendar days and drops:
1. The boundary row itself (its `Target` spans the gap — a multi-week price move labelled as "next-day close")
2. The 50 rows before each boundary (their rolling features are computed across the gap, mixing pre-gap and post-gap prices)

This removes ~8% of rows from the combined dataset but eliminates corrupt training signal.

### No-Leakage Guarantee

The pipeline enforces temporal integrity at every stage:

1. **Feature engineering:** All features use data up to and including today (`shift(-1)` target provides the buffer).
2. **Train/test split:** Chronological — oldest 80% trains, newest 20% tests. No shuffling.
3. **Hyperparameter tuning:** `TimeSeriesSplit` — each fold trains on past data, evaluates on future data.
4. **Stacking CV:** `TimeSeriesSplit` — out-of-fold predictions flow forward in time, never backward.
5. **Feature selection:** Mutual information computed on training portion only.

---

## Bug Fixes Applied

Seven bugs were identified through a comprehensive audit and fixed:

### Critical Fixes

| # | Bug | File(s) | Fix |
|---|-----|---------|-----|
| 1 | **Features one day stale** — `.shift(1)` on all derived features discarded today's close, the single most predictive input | `features.py` | Removed all `.shift(1)` calls from `add_technical_indicators`, `add_rolling_features`, and `add_return_features`. The target `shift(-1)` already provides the leakage buffer. |
| 2 | **False targets at file-stitch boundaries** — 32-day and 26-day gaps in combined CSV created phantom single-day price moves | `data_loader.py` | Added gap detection (>5 days) that drops boundary rows and the 50 preceding rows with corrupt rolling features. |
| 3 | **Stacking used KFold (temporal leakage)** — `KFold(shuffle=False)` trained base models on future data to predict past rows | `models.py` | Replaced `KFold` with `TimeSeriesSplit(n_splits=5)`. |

### Major Fixes

| # | Bug | File(s) | Fix |
|---|-----|---------|-----|
| 4 | **500209 analyzed twice** — both `500209.csv` (1-year subset) and `500209_combined.csv` (5-year full) in DATA_PATHS | `config.py` | Removed `500209.csv` from `DATA_PATHS`. |
| 5 | **Raw OHLCV columns excluded from features** — today's price levels were invisible to the model | `features.py` | Trimmed the exclude list to keep Open, High, Low, Close, WAP, Volume, Trades, and Spreads as direct features. |

### Moderate/Minor Fixes

| # | Bug | File(s) | Fix |
|---|-----|---------|-----|
| 6 | **Nested n_jobs parallelism** — XGBoost/LightGBM OpenMP threads × CV parallel jobs = thread oversubscription | `models.py` | Set `n_jobs=1` on XGBRegressor and LGBMRegressor; outer CV handles parallelism. |
| 7 | **Python 3.10+ type hints** — `str \| None` and `list[str]` fail on Python 3.9 | `visualize.py`, `splitter.py` | Replaced with `Optional[str]` and `List[str]` from `typing`. |

---

## Setup and Usage

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Athmeeya2006/time-series-forecasting.git
cd time-series-forecasting

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run full pipeline for all datasets
python main.py

# Run for a specific dataset
python main.py 500209_combined.csv

# Run data loader standalone (inspect data)
python data_loader.py

# Run feature engineering standalone (inspect features)
python features.py
```

### Dependencies

```
numpy
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
```

---

## Outputs

All outputs are saved to the `outputs/` directory with a dataset prefix:

| File | Description |
|------|-------------|
| `{prefix}_01_price_history.png` | Close price history plot |
| `{prefix}_02_correlation_heatmap.png` | Feature correlation heatmap (top 15) |
| `{prefix}_03_all_predictions_overlay.png` | All models overlaid on actual prices |
| `{prefix}_pred_{model}.png` | Per-model actual vs predicted |
| `{prefix}_fi_{model}.png` | Feature importance (tree-based models only) |
| `{prefix}_04_benchmark_comparison.png` | MAPE and directional accuracy bar charts |
| `{prefix}_benchmark_results.csv` | Full metrics table for all models |

---

## Limitations and Honest Assessment

- **Small datasets:** The 1-year datasets (~247 rows) have limited training data after the 80/20 split. Feature selection is capped to prevent overfitting, but results should be interpreted cautiously.
- **No transaction costs:** Directional accuracy does not account for bid-ask spreads, brokerage fees, or slippage.
- **Single-stock models:** Each stock is modeled independently. Cross-stock signals are not captured.
- **No exogenous features:** The model uses only price/volume data. Macro indicators, news sentiment, and sector indices are excluded.
- **If a model scores ~50% directional accuracy** after all fixes, that is an honest result — it means the features do not contain predictive signal for that stock's next-day direction. Do not trade on it.

> **Important:** After applying all 7 fixes, you should re-run the full pipeline (`python main.py`) to get updated benchmark results. The previous results in the outputs folder were generated with the buggy code and are no longer valid.
