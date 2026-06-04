"""
main.py - Orchestrator. Runs the full pipeline in order. One job only.
"""

import sys
from pathlib import Path

from config import BASE_DIR, DATA_PATHS
from data_loader import load_and_clean
from features import build_features
from splitter import chronological_split
from visualize import plot_price_history, plot_correlation_heatmap
from benchmark import run_benchmark


def _resolve_data_paths(data_paths=None):
    paths = data_paths or DATA_PATHS
    resolved_paths = []
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
        resolved_paths.append(path)
    return resolved_paths


def _output_prefix(data_path: Path) -> str:
    return data_path.stem


def run_pipeline(data_path: Path):
    output_prefix = _output_prefix(data_path)

    print("=" * 50)
    print(f"  Stock Price Prediction Pipeline: {data_path.name}")
    print("=" * 50)

    # Step 1: Load + clean
    df = load_and_clean(data_path)

    # Step 2: EDA plots
    plot_price_history(df, output_prefix=output_prefix)

    # Step 3: Feature engineering
    df, feature_cols = build_features(df)
    plot_correlation_heatmap(df, feature_cols[:15], output_prefix=output_prefix)  # top 15 for readability

    # Step 4: Chronological split
    X_train, X_test, y_train, y_test, _ = chronological_split(df, feature_cols)

    # Step 5: Benchmark all models
    results_df, predictions = run_benchmark(
        X_train, X_test, y_train, y_test, feature_cols, output_prefix=output_prefix
    )

    # Step 6: Print winner (directional accuracy ranked)
    best = results_df.iloc[0]
    import numpy as _np
    dir_acc_str = f"{best['Dir Acc%']:.1f}%" if not _np.isnan(best['Dir Acc%']) else "N/A"
    print(
        f"\n WINNER: {best['Model']} | DirAcc={dir_acc_str}"
        f" | MAPE={best['MAPE%']:.2f}%"
    )

    return results_df


def main(data_paths=None):
    results = {}
    for data_path in _resolve_data_paths(data_paths):
        results[data_path.stem] = run_pipeline(data_path)

    if len(results) == 1:
        return next(iter(results.values()))
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
