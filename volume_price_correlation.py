"""
volume_price_correlation.py - Standalone tool to compute and visualize the 
stationary log-return correlation and lead-lag cross-correlation between 
stock price and trading volume.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Theme Configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})

BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    "500112_combined.csv",
    "500209_combined.csv",
    "500247_combined.csv",
    "500510_combined.csv",
    "500696_combined.csv",
    "532174_combined.csv",
    "532215_combined.csv",
]
OUTPUT_DIR = BASE_DIR / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_and_plot_correlation(file_name: str):
    path = BASE_DIR / file_name
    if not path.exists():
        print(f"[Error] File not found: {path}")
        return None

    prefix = Path(file_name).stem

    # 1. Load & Sort chronologically
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%B-%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Extract price & volume
    price = df["Close Price"]
    volume = df["No.of Shares"]

    # 3. Compute stationary features (Log returns & changes)
    log_return = np.log(price / price.shift(1))
    log_vol_change = np.log(volume / volume.shift(1))

    # Mask out NaNs and Infs (often occur in volume if volume is 0, though rare in large equities)
    valid_mask = (
        log_return.notna()
        & log_vol_change.notna()
        & ~np.isinf(log_return)
        & ~np.isinf(log_vol_change)
    )
    
    clean_return = log_return[valid_mask]
    clean_vol_change = log_vol_change[valid_mask]

    # Compute correlation coefficients
    pearson_corr = clean_return.corr(clean_vol_change, method="pearson")
    spearman_corr = clean_return.corr(clean_vol_change, method="spearman")

    # 4. Compute Lead-Lag Cross-Correlation (Lags: -5 to +5)
    # Lag k: Volume(t) vs Return(t+k). 
    # k > 0: Volume today vs Future Return (Volume leads)
    # k < 0: Volume today vs Past Return (Price leads)
    lags = np.arange(-5, 6)
    ccf_values = []
    for lag in lags:
        if lag >= 0:
            shifted_return = clean_return.shift(-lag)
        else:
            shifted_return = clean_return.shift(-lag)  # pandas shift shifts index forward/backward
        
        # Align series after shifting
        combined = pd.concat([clean_vol_change, shifted_return], axis=1).dropna()
        combined.columns = ["VolChange", "RetShifted"]
        # Drop inf values if any
        combined = combined[~np.isinf(combined).any(axis=1)]
        corr = combined["VolChange"].corr(combined["RetShifted"], method="pearson")
        ccf_values.append(corr)

    ccf_df = pd.DataFrame({"Lag": lags, "Correlation": ccf_values})

    # 5. Generate Beautiful Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot A: Scatter Plot with Regression Line
    sns.regplot(
        x=clean_vol_change,
        y=clean_return,
        ax=axes[0],
        scatter_kws={"alpha": 0.4, "color": "#1f77b4", "edgecolor": "none", "s": 25},
        line_kws={"color": "#d62728", "linewidth": 2},
    )
    axes[0].set_title(f"Log Return vs. Log Volume Change\nPearson: {pearson_corr:+.3f} | Spearman: {spearman_corr:+.3f}")
    axes[0].set_xlabel("Daily Log Volume Change")
    axes[0].set_ylabel("Daily Log Price Return")

    # Plot B: Lead-Lag Cross-Correlation Bar Chart
    colors = ["#2ca02c" if val >= 0 else "#ff7f0e" for val in ccf_values]
    axes[1].bar(ccf_df["Lag"], ccf_df["Correlation"], color=colors, alpha=0.85, width=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    
    # 95% Confidence Interval limit line (approximate)
    n = len(clean_return)
    ci_limit = 1.96 / np.sqrt(n)
    axes[1].axhline(ci_limit, color="gray", linewidth=1.2, linestyle=":", label="95% Significance Limit")
    axes[1].axhline(-ci_limit, color="gray", linewidth=1.2, linestyle=":")
    
    axes[1].set_title("Lead-Lag Cross-Correlation\nVolume(t) vs. Price Return(t+k)")
    axes[1].set_xlabel("Lag (k days)")
    axes[1].set_ylabel("Correlation Coefficient")
    axes[1].set_xticks(lags)
    axes[1].set_xticklabels([f"t{k:+d}" if k != 0 else "t" for k in lags])
    axes[1].legend(loc="upper right")

    plt.suptitle(f"Price-Volume Joint Dynamics Audit — Stock Code: {prefix}", y=0.98)
    plt.tight_layout()

    # Save output
    output_path = OUTPUT_DIR / f"{prefix}_volume_price_correlation.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\n[Finished] Analyzed {file_name}")
    print(f"  - Pearson Correlation: {pearson_corr:+.5f}")
    print(f"  - Spearman Correlation: {spearman_corr:+.5f}")
    print(f"  - Plot saved to: {output_path}")

    return {
        "prefix": prefix,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "max_ccf_lag": ccf_df.loc[ccf_df["Correlation"].abs().idxmax()]["Lag"],
        "max_ccf_val": ccf_df.loc[ccf_df["Correlation"].abs().idxmax()]["Correlation"]
    }


def main():
    print("=" * 60)
    print("      Price-Volume Joint Dynamics Analysis Engine")
    print("=" * 60)
    
    summary_results = []
    for file_name in DATA_PATHS:
        res = calculate_and_plot_correlation(file_name)
        if res:
            summary_results.append(res)
            
    print("\n" + "=" * 60)
    print(f"{'Stock Code':<20} | {'Pearson':<9} | {'Spearman':<9} | {'Max CCF Lag':<12}")
    print("-" * 60)
    for res in summary_results:
        lag_str = f"t{int(res['max_ccf_lag']):+d}" if res['max_ccf_lag'] != 0 else "t"
        print(f"{res['prefix']:<20} | {res['pearson']:+9.4f} | {res['spearman']:+9.4f} | {lag_str:<12} ({res['max_ccf_val']:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
