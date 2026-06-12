"""
Feature engineering pipeline for stock price forecasting.
Computes lag features, rolling statistics, price returns, technical indicators, and calendar features.
"""

import pandas as pd
import numpy as np
from config import TARGET_COL, DATE_COL, LAG_STEPS, ROLLING_WINDOWS


COLS_FOR_LAGS = [
    "Open Price", "High Price", "Low Price", "Close Price",
    "WAP", "Spread High-Low", "Spread Close-Open",
    "No.of Shares", "No. of Trades",
]


def add_lag_features(df):
    for col in COLS_FOR_LAGS:
        if col in df.columns:
            for lag in LAG_STEPS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    """Rolling mean, std, min, and max on Close (using today's close; no extra shift
    needed because the target is already shift(-1))."""
    for w in ROLLING_WINDOWS:
        base = df[TARGET_COL]
        df[f"close_mean_{w}d"] = base.rolling(w).mean()
        df[f"close_std_{w}d"]  = base.rolling(w).std()
        df[f"close_min_{w}d"]  = base.rolling(w).min()
        df[f"close_max_{w}d"]  = base.rolling(w).max()
        # Rolling range as a fraction of rolling mean (volatility-normalized)
        df[f"close_range_pct_{w}d"] = (
            (df[f"close_max_{w}d"] - df[f"close_min_{w}d"])
            / (df[f"close_mean_{w}d"] + 1e-9)
        )
    # Volume rolling features
    if "No.of Shares" in df.columns:
        vol = df["No.of Shares"]
        for w in ROLLING_WINDOWS:
            df[f"volume_mean_{w}d"] = vol.rolling(w).mean()
            df[f"volume_ratio_{w}d"] = df["No.of Shares"] / (
                vol.rolling(w).mean() + 1e-9
            )
    return df


def add_return_features(df):
    """Return features use today's close (no extra shift; target is already
    shift(-1), so using today's data is safe)."""
    close = df[TARGET_COL]
    df["daily_return"]    = close.pct_change()
    df["log_return"]      = np.log(close / close.shift(1))
    # Rolling return (momentum signal)
    df["return_3d"]       = close.pct_change(3)
    df["return_5d"]       = close.pct_change(5)
    df["return_10d"]      = close.pct_change(10)
    # Return volatility (realized vol)
    df["return_std_5d"]   = df["daily_return"].rolling(5).std()
    df["return_std_10d"]  = df["daily_return"].rolling(10).std()
    return df


def _rsi(series, period=14):
    """RSI: Relative Strength Index. Measures momentum. 0-100 scale.
    Above 70 = overbought, below 30 = oversold."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _ema(series, span):
    """Exponential Moving Average: gives more weight to recent prices."""
    return series.ewm(span=span, adjust=False).mean()


def add_technical_indicators(df):
    close = df[TARGET_COL]  # Use today's close (target is shift(-1), no leakage)

    # Simple Moving Averages
    df["SMA_5"]  = close.rolling(5).mean()
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()

    # Exponential Moving Averages
    df["EMA_5"]  = _ema(close, 5)
    df["EMA_10"] = _ema(close, 10)
    df["EMA_20"] = _ema(close, 20)

    # RSI (14-day)
    df["RSI_14"] = _rsi(close, 14)

    # MACD = EMA12 - EMA26. Signal line = EMA9 of MACD.
    ema12         = _ema(close, 12)
    ema26         = _ema(close, 26)
    df["MACD"]         = ema12 - ema26
    df["MACD_signal"]  = _ema(df["MACD"], 9)
    df["MACD_hist"]    = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands: price relative to its rolling mean +/- 2 std
    bb_mean        = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_upper"] = bb_mean + 2 * bb_std
    df["BB_lower"] = bb_mean - 2 * bb_std
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]          # volatility proxy
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_width"] + 1e-9)

    # Price vs SMA ratios (where is price relative to trend?)
    df["price_vs_SMA5"]  = close / (df["SMA_5"]  + 1e-9)
    df["price_vs_SMA20"] = close / (df["SMA_20"] + 1e-9)
    df["price_vs_SMA50"] = close / (df["SMA_50"] + 1e-9)

    # SMA crossover signals (momentum regime)
    df["SMA5_vs_SMA20"]  = df["SMA_5"] / (df["SMA_20"] + 1e-9)
    df["SMA10_vs_SMA50"] = df["SMA_10"] / (df["SMA_50"] + 1e-9)
    df["EMA5_vs_EMA20"]  = df["EMA_5"] / (df["EMA_20"] + 1e-9)

    # ATR (Average True Range) - volatility indicator
    if "High Price" in df.columns and "Low Price" in df.columns:
        high = df["High Price"]
        low  = df["Low Price"]
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(14).mean()
        df["ATR_pct"] = df["ATR_14"] / (close + 1e-9)  # normalized ATR

    # High-Low spread ratio
    df["HL_ratio"] = df["Spread High-Low"] / (close + 1e-9)

    # Stochastic Oscillator (%K)
    if "High Price" in df.columns and "Low Price" in df.columns:
        low14  = df["Low Price"].rolling(14).min()
        high14 = df["High Price"].rolling(14).max()
        df["stoch_K"] = (close - low14) / (high14 - low14 + 1e-9) * 100
        df["stoch_D"] = df["stoch_K"].rolling(3).mean()

    return df


def add_calendar_features(df):
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["month"]     = df[DATE_COL].dt.month
    df["quarter"]   = df[DATE_COL].dt.quarter
    return df


def add_target(df):
    """Next-day close price. This is what we predict."""
    df["Target"] = df[TARGET_COL].shift(-1)
    return df


def select_features(df, feature_cols, max_features=None):
    """
    Select the top-K features using mutual information regression.
    K is capped at train_rows / 5 (minimum 15) to prevent overfitting
    on small datasets. If the dataset is large enough, all features pass.
    """
    from sklearn.feature_selection import mutual_info_regression
    from config import TRAIN_RATIO

    n_train = int(len(df) * TRAIN_RATIO)
    if max_features is None:
        max_features = max(15, n_train // 5)

    if len(feature_cols) <= max_features:
        print(f"[features] Keeping all {len(feature_cols)} features "
              f"(under cap of {max_features})")
        return feature_cols

    # Compute MI on training portion only to avoid leakage
    X_train = df[feature_cols].iloc[:n_train]
    y_train = df["Target"].iloc[:n_train]

    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    selected = mi_series.head(max_features).index.tolist()
    print(f"[features] Selected {len(selected)}/{len(feature_cols)} features "
          f"by mutual information (cap={max_features})")
    print(f"[features] Top 10: {selected[:10]}")
    return selected


def build_features(df):
    """
    Master function. Applies all feature engineering in order.
    Returns (df_with_features, list_of_feature_column_names).
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_return_features(df)
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    df = add_target(df)

    df = df.dropna().copy().reset_index(drop=True)

    # Exclude only metadata columns; keep raw OHLC/Volume as direct features
    # so today's price levels are visible to the model.
    exclude = [DATE_COL, "Target",
               "Total Turnover (Rs.)", "Deliverable Quantity",
               "% Deli. Qty to Traded Qty"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Adaptive feature selection: prevent overfitting on small datasets
    feature_cols = select_features(df, feature_cols)

    print(f"[features] {len(df)} usable rows, {len(feature_cols)} features (final)")
    return df, feature_cols


if __name__ == "__main__":
    from config import DATA_PATHS, BASE_DIR
    from pathlib import Path
    from data_loader import load_and_clean

    p = Path(DATA_PATHS[0])
    if not p.is_absolute():
        p = BASE_DIR / p
    data = load_and_clean(p)
    data, fcols = build_features(data)
    print("Sample features:", fcols[:10])
