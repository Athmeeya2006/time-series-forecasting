"""
models.py - All model definitions and hyperparameter tuning. One job only.

Each train_* function:
  - Takes X_train, y_train
  - Tunes hyperparameters using TimeSeriesSplit (NO future data leakage)
  - Returns (best_fitted_model, best_params_dict)

MODEL_REGISTRY maps name -> function so benchmark.py can loop over all.
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, StackingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import (
    TSCV_SPLITS, RIDGE_ALPHAS, LASSO_ALPHAS,
    ENET_GRID, SVR_GRID, RF_GRID, GBM_GRID, XGB_GRID, LGBM_GRID
)

TSCV = TimeSeriesSplit(n_splits=TSCV_SPLITS)

# Threshold: if grid has more combinations than this, use RandomizedSearchCV
_RANDOMIZED_THRESHOLD = 60
_RANDOMIZED_ITERS = 40


def _grid_size(param_grid):
    """Estimate the number of combinations in a parameter grid."""
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size


def _tune(estimator, param_grid, X_train, y_train, scale=True):
    """
    Wraps estimator in a Pipeline and runs GridSearchCV or RandomizedSearchCV
    with TimeSeriesSplit.
    scale=True adds StandardScaler (required for Ridge, Lasso, SVR).
    scale=False skips scaler (tree models don't need it).
    """
    steps = [("model", estimator)]
    if scale:
        steps = [("scaler", StandardScaler())] + steps

    prefixed = {f"model__{k}": v for k, v in param_grid.items()}
    pipe = Pipeline(steps)

    n_combos = _grid_size(prefixed)
    if n_combos > _RANDOMIZED_THRESHOLD:
        gs = RandomizedSearchCV(
            pipe, prefixed,
            n_iter=min(_RANDOMIZED_ITERS, n_combos),
            cv=TSCV,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
            random_state=42,
        )
    else:
        gs = GridSearchCV(
            pipe, prefixed,
            cv=TSCV,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_ridge(X_train, y_train):
    """
    Ridge Regression: Linear model with L2 penalty.
    Parameter: alpha = how strongly to penalize large coefficients.
    Higher alpha = more regularization = simpler model.
    """
    return _tune(Ridge(), {"alpha": RIDGE_ALPHAS}, X_train, y_train)


def train_lasso(X_train, y_train):
    """
    Lasso Regression: Linear model with L1 penalty.
    Unlike Ridge, Lasso can zero out irrelevant features entirely.
    Parameter: alpha = regularization strength.
    """
    return _tune(Lasso(max_iter=50000), {"alpha": LASSO_ALPHAS}, X_train, y_train)


def train_elasticnet(X_train, y_train):
    """
    ElasticNet: Combines L1 (Lasso) + L2 (Ridge) penalties.
    Parameters:
      alpha    = total penalty strength
      l1_ratio = mix (0=pure Ridge, 1=pure Lasso, 0.5=half-half)
    Often outperforms both individually.
    """
    return _tune(ElasticNet(max_iter=100000), ENET_GRID, X_train, y_train)


def train_svr(X_train, y_train):
    """
    Support Vector Regression: finds a tube around data, ignores points inside.
    Excellent for small datasets because it focuses on hard examples.
    Parameters:
      C       = penalty for points outside the tube (higher = less regularization)
      epsilon = width of the tube (tolerance margin)
      kernel  = rbf for non-linear, linear for linear
    """
    return _tune(SVR(), SVR_GRID, X_train, y_train)


def train_random_forest(X_train, y_train):
    """
    Random Forest: ensemble of decision trees trained on random subsets.
    Parameters:
      n_estimators     = number of trees (more = more stable, slower)
      max_depth        = how deep each tree can grow (None = unlimited, risky on small data)
      min_samples_leaf = minimum samples at a leaf node (higher = simpler trees)
    """
    return _tune(
        RandomForestRegressor(random_state=42),
        RF_GRID, X_train, y_train, scale=False
    )


def train_gradient_boosting(X_train, y_train):
    """
    Gradient Boosting: builds trees sequentially, each one corrects prior errors.
    Parameters:
      n_estimators  = number of boosting rounds
      max_depth     = tree depth per round (keep low: 3-5)
      learning_rate = how much each tree contributes (lower = slower but better)
    """
    return _tune(
        GradientBoostingRegressor(random_state=42),
        GBM_GRID, X_train, y_train, scale=False
    )


def train_xgboost(X_train, y_train):
    """
    XGBoost: optimized, regularized gradient boosting.
    Adds subsample (fraction of rows per tree) and column subsampling.
    Often the strongest classical ML model.
    """
    return _tune(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
        XGB_GRID, X_train, y_train, scale=False
    )


def train_lightgbm(X_train, y_train):
    """
    LightGBM: leaf-wise tree growth (vs depth-wise in XGBoost).
    Faster, often comparable accuracy.
    num_leaves controls complexity directly.
    """
    return _tune(
        LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
        LGBM_GRID, X_train, y_train, scale=False
    )


def train_adaboost(X_train, y_train):
    """
    AdaBoost: adaptive boosting with decision stumps.
    Focuses on hard-to-predict samples by re-weighting.
    """
    grid = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.5],
        "estimator__max_depth": [1, 2, 3],
    }
    steps = [("model", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=2),
        random_state=42,
    ))]
    prefixed = {f"model__{k}": v for k, v in grid.items()}
    pipe = Pipeline(steps)

    n_combos = 1
    for v in prefixed.values():
        n_combos *= len(v)
    if n_combos > _RANDOMIZED_THRESHOLD:
        gs = RandomizedSearchCV(
            pipe, prefixed,
            n_iter=min(_RANDOMIZED_ITERS, n_combos),
            cv=TSCV,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
            random_state=42,
        )
    else:
        gs = GridSearchCV(
            pipe, prefixed,
            cv=TSCV,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_stacking(X_train, y_train):
    """
    Stacking Ensemble: combines multiple diverse base models with a Ridge
    meta-learner. Each base model's out-of-fold predictions become features
    for the meta-learner, capturing complementary signals.
    """
    base_estimators = [
        ("ridge", Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )),
        ("xgb", XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0
        )),
        ("lgbm", LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            random_state=42, verbose=-1
        )),
    ]

    # TimeSeriesSplit maintains temporal ordering and provides a
    # full-partition CV (every sample in exactly one test fold).
    # KFold was wrong here — it trains on future data to predict the past.
    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        cv=TimeSeriesSplit(n_splits=TSCV_SPLITS),
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return stack, {"ensemble": "Ridge+RF+XGB+LGBM -> Ridge meta"}


MODEL_REGISTRY = {
    "Ridge":             train_ridge,
    "Lasso":             train_lasso,
    "ElasticNet":        train_elasticnet,
    "SVR":               train_svr,
    "Random Forest":     train_random_forest,
    "Gradient Boosting": train_gradient_boosting,
    "XGBoost":           train_xgboost,
    "LightGBM":          train_lightgbm,
    "AdaBoost":          train_adaboost,
    "Stacking":          train_stacking,
}
