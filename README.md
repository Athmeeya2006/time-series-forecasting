# Stock Price Forecasting Pipeline

A machine learning and quantitative analysis pipeline designed to predict the next-day closing price of equities. The system processes historical OHLCV (Open, High, Low, Close, Volume) data, engineers a comprehensive set of predictive features, performs time-series cross-validation to prevent data leakage, and benchmarks multiple predictive models ranging from linear regressions to tree ensembles and custom stacking models.

---

## Core Objectives

* **Predictive Modeling**: Develop and evaluate machine learning models to forecast the next-day close price based on current and historical trading data.
* **Leakage-Free Validation**: Enforce strict temporal separation during training, hyperparameter tuning, and model evaluation to prevent look-ahead bias.
* **Feature Engineering**: Generate technical, rolling, lag, momentum, and calendar-based signals to capture market dynamics.
* **Benchmark Evaluation**: Compare model performance against a naive baseline (predicting tomorrow's price as today's close) using robust statistical metrics.

---

## System Architecture

The project is structured as a modular pipeline following a single-responsibility layout.

```
├── config.py                  # Central configuration, hyperparameter grids, and file paths
├── data_loader.py             # Data loading, date parsing, cleaning, and date-gap removal
├── features.py                # Feature engineering and adaptive mutual-information selection
├── splitter.py                # Chronological train/test splitting
├── models.py                  # Regressor definitions and custom time-series stacking implementation
├── benchmark.py               # Benchmark orchestration, metrics collection, and evaluation
├── evaluate.py                # Custom evaluation metrics (MAE, RMSE, MAPE, R2, Directional Accuracy)
└── visualize.py               # Chart generation (price history, correlation, overlays, importances)
```

### Process Flow

1. **Load and Clean**: `data_loader.py` reads historical CSV files, formats headers, parses dates chronologically, and removes stitched file anomalies.
2. **Feature Engineering**: `features.py` generates over 100 base features and filters them using mutual-information regression to prevent overfitting.
3. **Dataset Splitting**: `splitter.py` splits features and targets chronologically into 80% training and 20% testing sets to preserve temporal order.
4. **Model Tuning & Training**: `models.py` runs hyperparameter searches (Grid or Randomized) with `TimeSeriesSplit` cross-validation.
5. **Evaluation**: `benchmark.py` and `evaluate.py` compute predictive accuracy and compare performance against the naive baseline.
6. **Visualization**: `visualize.py` saves performance plots, prediction overlays, and feature importances.

---

## Data Pipeline and Integrity

Predicting financial time-series requires strict data cleaning and leakage prevention.

### Handling Year-Stitch Boundaries

Equities datasets stitched from multiple annual files often contain multi-week gaps (for example, missing data over year transitions). These gaps introduce two critical issues:
1. **Invalid Target Labeling**: The next-day closing price across a 30-day gap represents a multi-week shift, not a single-day transition.
2. **Corrupted rolling features**: Rolling averages computed across a gap mix unrelated price levels from different periods.

The data loader automatically flags any gap exceeding 5 trading days. It drops the boundary row (preventing corrupt targets) and the 50 preceding trading rows (preventing corrupted rolling features).

### Strict Leakage Prevention

To ensure evaluation metrics reflect real-world performance, the pipeline implements:
* **Natural Buffer**: The target variable is `Close Price.shift(-1)`. All feature calculations use data up to and including today's close, preventing look-ahead leakage.
* **Chronological Splits**: No random shuffling. The training set strictly precedes the test set.
* **TimeSeriesSplit CV**: Hyperparameter tuning uses rolling folds where validation sets always succeed training sets.
* **Out-of-Fold Stacking**: The custom stacking regressor uses chronological cross-validation to generate out-of-fold meta-features, avoiding future information leaks.
* **Isolated Feature Selection**: Mutual-information scores are calculated strictly on training data.

---

## Feature Engineering and Selection

The system engineers a wide array of features categorized into five distinct groups:

| Feature Group | Description |
|---|---|
| **Lags** | Historical values of prices, volume, spreads, and transactions from 1, 2, 3, 5, and 10 days ago. |
| **Rolling Stats** | Rolling mean, standard deviation, minimum, maximum, and normalized range percentages across 3, 5, 10, and 20-day windows. |
| **Returns** | Daily percentage returns, log returns, momentum (3, 5, and 10-day returns), and realized volatility. |
| **Technical Indicators** | Moving averages (SMA, EMA), Relative Strength Index (RSI-14), MACD (line, signal, histogram), Bollinger Bands (upper, lower, width, and position percentage), Stochastic Oscillator (%K, %D), and normalized Average True Range (ATR). |
| **Calendar** | Day of the week, month, and quarter to capture temporal and seasonal patterns. |

### Adaptive Feature Selection

To prevent overfitting (especially when working with smaller datasets), the pipeline computes mutual-information regression scores between features and targets on the training split. It ranks and limits the final feature count to `max(15, training_rows / 5)`.

---

## Model Zoo

The system evaluates ten different modeling approaches:

* **Linear Regressors**: Ridge (L2 regularization), Lasso (L1 regularization), and ElasticNet (L1 and L2 combination).
* **Kernel Methods**: Support Vector Regression (SVR) with linear and radial basis function (RBF) kernels.
* **Decision Trees & Bagging**: Random Forest Regressor using parallelized ensemble trees.
* **Boosting Algorithms**: Gradient Boosting, AdaBoost, XGBoost, and LightGBM models.
* **Stacking Ensemble**: A custom stacking regressor that trains base models (Ridge, RF, XGBoost, LightGBM) chronologically and feeds their out-of-fold predictions into a Ridge meta-regressor.

### Custom Stacking Implementation

Standard scikit-learn stacking uses K-Fold cross-validation, which partitions data randomly and violates time-series sequencing (training on future data to predict past data). Our custom `TimeSeriesStackingRegressor` utilizes a rolling `TimeSeriesSplit` cv-iterator to construct meta-features chronologically.

---

## Evaluation Metrics

Model performance is quantified using the following metrics:

* **MAE** (Mean Absolute Error): The average absolute difference between predicted and actual prices in currency units.
* **RMSE** (Root Mean Squared Error): The square root of the mean squared errors, placing a higher penalty on larger errors.
* **MAPE** (Mean Absolute Percentage Error): The average percentage deviation of predictions.
* **R2** (Coefficient of Determination): The proportion of target variance explained by the model features.
* **Directional Accuracy (Dir Acc %)**: The percentage of days where the model correctly predicts whether the price will close higher or lower than the previous day.

### Directional Accuracy Details

To prevent artificial bias and establish a true baseline:
* **Tie Handling**: If a model predicts no price change, it is credited 0.5 (representing no directional information rather than an incorrect prediction).
* **Flat-Day Exclusion**: Days where the actual price does not change are excluded from the calculation, as there is no true directional movement.
* **Naive Floor**: Under this policy, the Naive Baseline receives a consistent 50% score, serving as the benchmark to beat.

---

## Setup and Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Athmeeya2006/time-series-forecasting.git
   cd time-series-forecasting
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

* **Run Full Pipeline**: Processes and benchmarks all datasets configured in `config.py`:
  ```bash
  python main.py
  ```

* **Run Specific Dataset**: Specify a single CSV file:
  ```bash
  python main.py 500209_combined.csv
  ```

* **Standalone Price-Volume Analysis**: Run the lead-lag correlation utility:
  ```bash
  python volume_price_correlation.py
  ```

---

## Generated Artifacts

Results and visualizations are stored in the `outputs/` directory:

* `*_01_price_history.png`: Visual historical trend of the stock's closing price.
* `*_02_correlation_heatmap.png`: Correlation matrix of the top 15 selected features.
* `*_03_all_predictions_overlay.png`: Combined visualization overlaying all model predictions against actual test prices.
* `*_pred_{model_name}.png`: Detailed view comparing individual model predictions against actual close prices.
* `*_fi_{model_name}.png`: Feature importance rankings for tree-based models.
* `*_04_benchmark_comparison.png`: Side-by-side bar chart comparison of MAPE and Directional Accuracy across all models.
* `*_benchmark_results.csv`: Table containing compiled evaluation metrics for all tested models.

---

## Limitations

* **Transaction Friction**: Directional accuracy metrics do not account for bid-ask spreads, trading commissions, or execution slippage.
* **Exogenous Variables**: The pipeline relies solely on price and volume dynamics. Macroeconomic data, news sentiment, and industry indices are not included.
* **Asset Isolation**: Models are trained on individual assets independently and do not capture cross-asset dependencies or sector correlations.
