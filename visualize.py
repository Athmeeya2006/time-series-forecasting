"""Visualization utilities for forecasting performance and data analysis."""

import os
from typing import Optional

from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")


def _output_path(filename: str, output_prefix: Optional[str] = None) -> str:
    if output_prefix:
        filename = f"{output_prefix}_{filename}"
    return str(OUTPUT_DIR / filename)


def plot_price_history(df, date_col="Date", price_col="Close Price", output_prefix=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[date_col], df[price_col], color="steelblue", linewidth=1.5)
    ax.set_title("Close Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Rs.)")
    plt.tight_layout()
    path = _output_path("01_price_history.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_correlation_heatmap(df, feature_cols, output_prefix=None):
    corr = df[feature_cols + ["Target"]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = _output_path("02_correlation_heatmap.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_predictions(y_test, predictions_dict: dict, model_name: str, output_prefix=None):
    """Overlay actual vs predicted for one model."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_test.values, label="Actual", color="black", linewidth=2)
    ax.plot(predictions_dict[model_name], label=f"Predicted ({model_name})",
            color="crimson", linewidth=1.5, linestyle="--")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Next-Day Close Price (Rs.)")
    ax.legend()
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = _output_path(f"pred_{safe_name}.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_all_predictions_overlay(y_test, predictions_dict: dict, output_prefix=None):
    """All models on one plot for comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.values, label="Actual", color="black", linewidth=2.5, zorder=5)
    colors = plt.cm.tab10.colors
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax.plot(preds, label=name, color=colors[i % len(colors)],
                linewidth=1.2, alpha=0.8)
    ax.set_title("All Models - Actual vs Predicted (Test Set)")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Next-Day Close Price (Rs.)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = _output_path("03_all_predictions_overlay.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_benchmark_table(results_df: pd.DataFrame, output_prefix=None):
    """Bar chart comparison of all models by MAPE% and Dir Acc%."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results_df.sort_values("MAPE%").plot(
        kind="bar", x="Model", y="MAPE%", ax=axes[0],
        color="steelblue", legend=False
    )
    axes[0].set_title("MAPE% by Model (lower = better)")
    axes[0].set_ylabel("MAPE%")
    axes[0].tick_params(axis="x", rotation=30)

    # Filter out NaN Dir Acc% rows for the bar chart
    dir_acc_df = results_df.dropna(subset=["Dir Acc%"])
    if not dir_acc_df.empty:
        dir_acc_df.sort_values("Dir Acc%", ascending=False).plot(
            kind="bar", x="Model", y="Dir Acc%", ax=axes[1],
            color="seagreen", legend=False
        )
    axes[1].set_title("Directional Accuracy% (higher = better)")
    axes[1].set_ylabel("Dir Acc%")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = _output_path("04_benchmark_comparison.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def _extract_model(model):
    """Unwrap Pipeline / StackingRegressor to get the inner estimator."""
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def plot_feature_importance(model, feature_cols: list, model_name: str, output_prefix=None):
    """For tree-based models that have feature_importances_."""
    m = _extract_model(model)
    if not hasattr(m, "feature_importances_"):
        return
    imp = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.head(20).plot(kind="bar", ax=ax, color="coral")
    ax.set_title(f"Top 20 Feature Importances - {model_name}")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    path = _output_path(f"fi_{safe}.png", output_prefix)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")

