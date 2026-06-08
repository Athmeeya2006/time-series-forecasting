"""
config.py - Centralized constants and hyperparameters.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATHS = [
    "500209_combined.csv",  # Full 5-year history (500209.csv is a 1-year subset)
    "500510.csv",
    "532174.csv",
]

DATE_COL = "Date"
DATE_FORMAT = "%d-%B-%Y"
TARGET_COL = "Close Price"

OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_RATIO = 0.8
TSCV_SPLITS = 5

LAG_STEPS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [3, 5, 10, 20]

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

ENET_GRID = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

SVR_GRID = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.001, 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

RF_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 2, 5],
    "min_samples_split": [2, 5],
}

GBM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "min_samples_leaf": [1, 3, 5],
}

XGB_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0],
}

LGBM_GRID = {
    "n_estimators": [200, 500],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 5.0],
    "min_child_samples": [5, 10, 20],
}
