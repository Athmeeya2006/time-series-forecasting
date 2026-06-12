"""Benchmark suite to train, evaluate, and compare multiple models."""

import numpy as np
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


def run_benchmark(X_train, X_test, y_train, y_test, feature_cols, output_prefix=None):
    results = []
    predictions_dict = {}

    prev_actual = np.concatenate([[y_train.iloc[-1]], y_test.iloc[:-1].to_numpy()])

    # Naive baseline first
    naive = naive_baseline_metrics(y_test, prev_actual)
    results.append(naive)
    print_metrics(naive)

    # All ML models
    for name, train_fn in MODEL_REGISTRY.items():
        print(f"\n[benchmark] Training {name}...")
        model, best_params = train_fn(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds, prev_actual, model_name=name)
        results.append(metrics)
        predictions_dict[name] = preds
        print_metrics(metrics)
        print(f"           Best params: {best_params}")

        # Per-model prediction plot
        plot_predictions(y_test, predictions_dict, name, output_prefix=output_prefix)

        # Feature importance if applicable
        plot_feature_importance(model, feature_cols, name, output_prefix=output_prefix)

    # Overlay all models
    plot_all_predictions_overlay(y_test, predictions_dict, output_prefix=output_prefix)

    # Summary table (rank by directional accuracy, then MAPE as a tiebreaker)
    results_df = pd.DataFrame(results)
    # Handle NaN in Dir Acc% for sorting: NaN goes to the bottom
    results_df = results_df.sort_values(
        ["Dir Acc%", "MAPE%"], ascending=[False, True],
        na_position="last"
    ).reset_index(drop=True)
    print("\n\n========== BENCHMARK RESULTS ==========")
    print(results_df.to_string(index=False))

    # Save CSV
    csv_name = "benchmark_results.csv"
    if output_prefix:
        csv_name = f"{output_prefix}_{csv_name}"
    csv_path = f"{OUTPUT_DIR}/{csv_name}"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[benchmark] Saved results to {csv_path}")

    # Bar chart
    plot_benchmark_table(results_df, output_prefix=output_prefix)

    return results_df, predictions_dict
