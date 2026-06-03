"""
config.py - All constants and hyperparameter grids in one place.
Change things here only. Nothing else needs editing for tuning.
"""

from pathlib import Path


BASE_DIR     = Path(__file__).resolve().parent
DATA_PATH    = BASE_DIR / "500209.csv"
TARGET_COL   = "Close Price"
DATE_COL     = "Date"
DATE_FORMAT  = "%d-%B-%Y"

TRAIN_RATIO     = 0.80
LAG_STEPS       = [1, 2, 3, 5]
ROLLING_WINDOWS = [3, 5, 10]
TSCV_SPLITS     = 5
OUTPUT_DIR      = "outputs"

# Ridge
RIDGE_ALPHAS = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]

# Lasso
LASSO_ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]

# ElasticNet
ENET_GRID = {
    "alpha":    [0.01, 0.1, 0.5, 1],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# SVR  (great for small datasets)
SVR_GRID = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5, 1],
    "kernel":  ["rbf", "linear"],
}

# Random Forest
RF_GRID = {
    "n_estimators":   [100, 200],
    "max_depth":      [3, 5, None],
    "min_samples_leaf": [2, 4],
}

# XGBoost
XGB_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

# LightGBM
LGBM_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 4],
    "learning_rate": [0.05, 0.1],
    "num_leaves":    [15, 31],
}
