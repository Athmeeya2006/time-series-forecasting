"""
benchmark.py - Trains ALL models, collects metrics, builds comparison table.
One job only. The main comparison engine.
"""

import pandas as pd
from evaluate import compute_metrics, naive_baseline_metrics, print_metrics
from models import MODEL_REGISTRY
from visualize import (
    plot_predictions, plot_all_predictions_overlay,
    plot_benchmark_table, plot_feature_importance
)
from config import OUTPUT_DIR
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_benchmark(X_train, X_test, y_train, y_test, feature_cols):
    results = []
    predictions_dict = {}

    # Naive baseline first
    naive = naive_baseline_metrics(y_test)
    results.append(naive)
    print_metrics(naive)

    # All ML models
    for name, train_fn in MODEL_REGISTRY.items():
        print(f"\n[benchmark] Training {name}...")
        model, best_params = train_fn(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds, model_name=name)
        results.append(metrics)
        predictions_dict[name] = preds
        print_metrics(metrics)
        print(f"           Best params: {best_params}")

        # Per-model prediction plot
        plot_predictions(y_test, predictions_dict, name)

        # Feature importance if applicable
        plot_feature_importance(model, feature_cols, name)

    # Overlay all models
    plot_all_predictions_overlay(y_test, predictions_dict)

    # Summary table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAPE%").reset_index(drop=True)
    print("\n\n========== BENCHMARK RESULTS ==========")
    print(results_df.to_string(index=False))

    # Save CSV
    csv_path = f"{OUTPUT_DIR}/benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[benchmark] Saved results to {csv_path}")

    # Bar chart
    plot_benchmark_table(results_df)

    return results_df, predictions_dict
