# Time Series Forecasting - Stock Next Day Close

## What this project does
This project trains multiple regression models to predict the next day close price for Indian stock data. It builds features from historical prices and volumes, evaluates each model, and ranks them by directional accuracy.

## How this project works
1. Load and clean CSV data.
2. Build lag, rolling, return, technical indicator, and calendar features (109 features total).
3. Split the data in chronological order (80/20).
4. Train and tune multiple models using time series cross validation (5-fold).
5. Evaluate models with MAE, RMSE, MAPE, R2, and directional accuracy.
6. Save plots and benchmark tables into the outputs folder.

## Models
Ridge, Lasso, ElasticNet, SVR, Random Forest, Gradient Boosting, XGBoost, LightGBM, AdaBoost, and a Stacking ensemble (Ridge + RF + XGB + LGBM with a Ridge meta-learner).

## Directional accuracy metric
Directional accuracy measures whether the model correctly predicts the direction of price movement relative to the previous day's close. The tie policy scores predictions that exactly match the previous close as 0.5 (reflecting no information rather than a wrong answer). Days where the actual price did not move are excluded from the denominator. This gives the naive baseline an honest 50% floor.

## Results (directional accuracy winners)
Winners are chosen by highest directional accuracy, with MAPE as a tie breaker.
These figures reflect the latest run outputs in outputs/ (updated 2026-06-04).

| Dataset | Winner | Directional Accuracy | MAPE |
| --- | --- | --- | --- |
| 500209 | Naive Baseline | 50.0% | 1.5623% |
| 500510 | Lasso | 75.0% | 1.0983% |
| 532174 | LightGBM | 62.5% | 2.0705% |

Full benchmark tables are saved in outputs/ as *_benchmark_results.csv.

## Run locally (use the existing virtual environment)
1. Create the virtual environment if it does not exist:
   - python -m venv .venv
2. Activate the virtual environment:
   - Linux or macOS: source .venv/bin/activate
3. Install dependencies:
   - python -m pip install -r requirements.txt
4. Run the pipeline:
   - python main.py

## Outputs
- Plots are saved in outputs/ with a dataset prefix.
- Benchmark CSV files are saved in outputs/.
