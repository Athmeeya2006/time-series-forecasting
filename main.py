"""
main.py - Orchestrator. Runs the full pipeline in order. One job only.
"""

from data_loader import load_and_clean
from features import build_features
from splitter import chronological_split
from visualize import plot_price_history, plot_correlation_heatmap
from benchmark import run_benchmark


def main():
    print("=" * 50)
    print("  Infosys Stock Price Prediction Pipeline")
    print("=" * 50)

    # Step 1: Load + clean
    df = load_and_clean()

    # Step 2: EDA plots
    plot_price_history(df)

    # Step 3: Feature engineering
    df, feature_cols = build_features(df)
    plot_correlation_heatmap(df, feature_cols[:15])  # top 15 for readability

    # Step 4: Chronological split
    X_train, X_test, y_train, y_test, _ = chronological_split(df, feature_cols)

    # Step 5: Benchmark all models
    results_df, predictions = run_benchmark(X_train, X_test, y_train, y_test, feature_cols)

    # Step 6: Print winner
    best = results_df.iloc[0]
    print(f"\n WINNER: {best['Model']} | MAPE={best['MAPE%']:.2f}%"
          f" | DirAcc={best['Dir Acc%']:.1f}%")

    return results_df


if __name__ == "__main__":
    main()
