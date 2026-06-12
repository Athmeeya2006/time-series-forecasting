"""Evaluation metrics for time-series forecasting models."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _directional_accuracy(y_pred, y_true, prev_actual) -> float:
    """
    Directional accuracy relative to prior actual values.

    Tie policy (pred_dir == 0, i.e. model predicts exactly prev_actual):
      - Scores 0.5 instead of 0, reflecting "no information" rather than
        "actively wrong." This gives the naive baseline an honest 50% floor.

    Days where the actual price did not move (true_dir == 0) are excluded
    from the denominator entirely, since there is no correct directional
    answer on those days and they dilute every model equally.
    """
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    prev_actual_arr = np.asarray(prev_actual, dtype=float)

    if len(y_true_arr) == 0:
        return np.nan
    if not (
        len(y_pred_arr) == len(y_true_arr) == len(prev_actual_arr)
    ):
        raise ValueError("y_pred, y_true, and prev_actual must be the same length")

    pred_dir = np.sign(y_pred_arr - prev_actual_arr)
    true_dir = np.sign(y_true_arr - prev_actual_arr)


    mask = true_dir != 0
    if mask.sum() == 0:
        return np.nan

    pred_dir = pred_dir[mask]
    true_dir = true_dir[mask]


    hits = np.where(pred_dir == true_dir, 1.0, np.where(pred_dir == 0, 0.5, 0.0))
    return np.mean(hits) * 100


def compute_metrics(y_true, y_pred, prev_actual, model_name: str = "Model") -> dict:
    """Returns all 5 key metrics as a dict."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    prev_actual_arr = np.asarray(prev_actual, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-9))) * 100
    r2 = r2_score(y_true_arr, y_pred_arr)
    dir_acc = _directional_accuracy(y_pred_arr, y_true_arr, prev_actual_arr)

    result = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE%": round(mape, 4),
        "R2": round(r2, 4),
        "Dir Acc%": round(dir_acc, 2),
    }
    return result


def naive_baseline_metrics(y_test, prev_actual) -> dict:
    """
    Naive: predict tomorrow = today.
    Uses prev_actual so lengths align across all models.
    """
    y_test_arr = np.asarray(y_test)
    y_pred = np.asarray(prev_actual)
    return compute_metrics(y_test_arr, y_pred, prev_actual, model_name="Naive Baseline")


def print_metrics(metrics: dict):
    dir_acc_str = f"{metrics['Dir Acc%']:.1f}%" if not np.isnan(metrics['Dir Acc%']) else "N/A"
    print(f"\n  {'Model':<25} MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}"
          f"  MAPE={metrics['MAPE%']:.2f}%  R2={metrics['R2']:.4f}"
          f"  DirAcc={dir_acc_str}")
