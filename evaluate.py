"""
evaluate.py - All metrics in one place. One job only.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Returns all 5 key metrics as a dict."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    r2   = r2_score(y_true, y_pred)

    # Directional accuracy: did consecutive predictions move in the same direction as actuals?
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    actual_dir  = np.sign(np.diff(y_true_arr))
    pred_dir    = np.sign(np.diff(y_pred_arr))
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    result = {
        "Model": model_name,
        "MAE":   round(mae,  4),
        "RMSE":  round(rmse, 4),
        "MAPE%": round(mape, 4),
        "R2":    round(r2,   4),
        "Dir Acc%": round(dir_acc, 2),
    }
    return result


def naive_baseline_metrics(y_true) -> dict:
    """
    Naive: predict tomorrow = today. This is the floor.
    Every model must beat this Dir Acc and MAPE.
    """
    y_arr = np.array(y_true)
    y_pred = y_arr[:-1]      # today as prediction
    y_eval = y_arr[1:]       # tomorrow as actual
    return compute_metrics(y_eval, y_pred, model_name="Naive Baseline")


def print_metrics(metrics: dict):
    print(f"\n  {'Model':<25} MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}"
          f"  MAPE={metrics['MAPE%']:.2f}%  R2={metrics['R2']:.4f}"
          f"  DirAcc={metrics['Dir Acc%']:.1f}%")
