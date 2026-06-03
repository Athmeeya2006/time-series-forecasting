"""
visualize.py - All plots. One job only.
Call any function independently or from benchmark/main.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")


def plot_price_history(df, date_col="Date", price_col="Close Price"):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[date_col], df[price_col], color="steelblue", linewidth=1.5)
    ax.set_title("Infosys Close Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Rs.)")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_price_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_correlation_heatmap(df, feature_cols):
    corr = df[feature_cols + ["Target"]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_predictions(y_test, predictions_dict: dict, model_name: str):
    """Overlay actual vs predicted for one model."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_test.values, label="Actual", color="black", linewidth=2)
    ax.plot(predictions_dict[model_name], label=f"Predicted ({model_name})",
            color="crimson", linewidth=1.5, linestyle="--")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Next-Day Close Price (Rs.)")
    ax.legend()
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = f"{OUTPUT_DIR}/pred_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_all_predictions_overlay(y_test, predictions_dict: dict):
    """All models on one plot for comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.values, label="Actual", color="black", linewidth=2.5, zorder=5)
    colors = plt.cm.tab10.colors
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax.plot(preds, label=name, color=colors[i % len(colors)],
                linewidth=1.2, alpha=0.8)
    ax.set_title("All Models — Actual vs Predicted (Test Set)")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Next-Day Close Price (Rs.)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_all_predictions_overlay.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_benchmark_table(results_df: pd.DataFrame):
    """Bar chart comparison of all models by MAPE%."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results_df.sort_values("MAPE%").plot(
        kind="bar", x="Model", y="MAPE%", ax=axes[0],
        color="steelblue", legend=False
    )
    axes[0].set_title("MAPE% by Model (lower = better)")
    axes[0].set_ylabel("MAPE%")
    axes[0].tick_params(axis="x", rotation=30)

    results_df.sort_values("Dir Acc%", ascending=False).plot(
        kind="bar", x="Model", y="Dir Acc%", ax=axes[1],
        color="seagreen", legend=False
    )
    axes[1].set_title("Directional Accuracy% (higher = better)")
    axes[1].set_ylabel("Dir Acc%")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_benchmark_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")


def plot_feature_importance(model, feature_cols: list, model_name: str):
    """For tree-based models that have feature_importances_."""
    # unwrap Pipeline if needed
    m = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if not hasattr(m, "feature_importances_"):
        return
    imp = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.head(20).plot(kind="bar", ax=ax, color="coral")
    ax.set_title(f"Top 20 Feature Importances — {model_name}")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    path = f"{OUTPUT_DIR}/fi_{safe}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[visualize] Saved {path}")
