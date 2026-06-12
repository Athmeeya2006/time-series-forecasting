"""
Model definitions, custom stacking ensemble, and hyperparameter tuning functions.
All models use time-series cross-validation to prevent temporal data leakage.
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
    """Train and tune a Ridge Regression model."""
    return _tune(Ridge(), {"alpha": RIDGE_ALPHAS}, X_train, y_train)


def train_lasso(X_train, y_train):
    """Train and tune a Lasso Regression model."""
    return _tune(Lasso(max_iter=50000), {"alpha": LASSO_ALPHAS}, X_train, y_train)


def train_elasticnet(X_train, y_train):
    """Train and tune an ElasticNet Regression model."""
    return _tune(ElasticNet(max_iter=100000), ENET_GRID, X_train, y_train)


def train_svr(X_train, y_train):
    """Train and tune a Support Vector Regression model."""
    return _tune(SVR(), SVR_GRID, X_train, y_train)


def train_random_forest(X_train, y_train):
    """Train and tune a Random Forest Regressor."""
    return _tune(
        RandomForestRegressor(random_state=42),
        RF_GRID, X_train, y_train, scale=False
    )


def train_gradient_boosting(X_train, y_train):
    """Train and tune a Gradient Boosting Regressor."""
    return _tune(
        GradientBoostingRegressor(random_state=42),
        GBM_GRID, X_train, y_train, scale=False
    )


def train_xgboost(X_train, y_train):
    """Train and tune an XGBoost Regressor."""
    return _tune(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
        XGB_GRID, X_train, y_train, scale=False
    )


def train_lightgbm(X_train, y_train):
    """Train and tune a LightGBM Regressor."""
    return _tune(
        LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
        LGBM_GRID, X_train, y_train, scale=False
    )


def train_adaboost(X_train, y_train):
    """Train and tune an AdaBoost Regressor."""
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


from sklearn.base import BaseEstimator, RegressorMixin, clone

class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Custom stacking regressor for time-series cross-validation.
    Trains base models chronologically and fits the meta-estimator
    on out-of-fold predictions.
    """
    def __init__(self, estimators, final_estimator, cv):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):

        self.fitted_estimators_ = []
        for name, est in self.estimators:
            fitted_est = clone(est).fit(X, y)
            self.fitted_estimators_.append(fitted_est)


        oof_preds = {name: np.zeros(len(X)) for name, _ in self.estimators}
        test_indices_all = []

        for train_idx, test_idx in self.cv.split(X, y):
            X_train_fold = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_test_fold = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]

            for name, est in self.estimators:
                fold_est = clone(est).fit(X_train_fold, y_train_fold)
                preds = fold_est.predict(X_test_fold)
                oof_preds[name][test_idx] = preds
            test_indices_all.extend(test_idx)


        test_indices_all = np.array(sorted(test_indices_all))


        meta_features = np.column_stack([oof_preds[name][test_indices_all] for name, _ in self.estimators])
        meta_targets = y.iloc[test_indices_all] if hasattr(y, "iloc") else y[test_indices_all]


        self.fitted_final_estimator_ = clone(self.final_estimator).fit(meta_features, meta_targets)
        return self

    def predict(self, X):

        meta_features = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        return self.fitted_final_estimator_.predict(meta_features)


def train_stacking(X_train, y_train):
    """Train a custom time-series stacking ensemble using base regressors and a meta-learner."""
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


    stack = TimeSeriesStackingRegressor(
        estimators=base_estimators,
        final_estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        cv=TimeSeriesSplit(n_splits=TSCV_SPLITS),
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
