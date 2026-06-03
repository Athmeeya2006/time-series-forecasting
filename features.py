"""
features.py - All feature engineering. One job only.

Features built:
  1. Lag features        - what the price/volume WAS n days ago
  2. Rolling stats       - trend and volatility over a window
  3. Return features     - percentage change, log return
  4. Technical indicators - RSI, MACD, Bollinger Bands, EMA, SMA
  5. Calendar features   - day of week, month
  6. Target              - next-day close price (shift -1)
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
    """Rolling mean and std on Close, shifted by 1 to prevent leakage."""
    for w in ROLLING_WINDOWS:
        base = df[TARGET_COL].shift(1)
        df[f"close_mean_{w}d"] = base.rolling(w).mean()
        df[f"close_std_{w}d"]  = base.rolling(w).std()
    return df


def add_return_features(df):
    df["daily_return"] = df[TARGET_COL].pct_change()
    df["log_return"]   = np.log(df[TARGET_COL] / df[TARGET_COL].shift(1))
    # Rolling return (momentum signal)
    df["return_3d"]    = df[TARGET_COL].pct_change(3)
    df["return_5d"]    = df[TARGET_COL].pct_change(5)
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
    close = df[TARGET_COL]

    # Simple Moving Averages
    df["SMA_5"]  = close.shift(1).rolling(5).mean()
    df["SMA_10"] = close.shift(1).rolling(10).mean()
    df["SMA_20"] = close.shift(1).rolling(20).mean()

    # Exponential Moving Averages
    df["EMA_5"]  = _ema(close.shift(1), 5)
    df["EMA_10"] = _ema(close.shift(1), 10)

    # RSI (14-day)
    df["RSI_14"] = _rsi(close.shift(1), 14)

    # MACD = EMA12 - EMA26. Signal line = EMA9 of MACD.
    ema12         = _ema(close.shift(1), 12)
    ema26         = _ema(close.shift(1), 26)
    df["MACD"]         = ema12 - ema26
    df["MACD_signal"]  = _ema(df["MACD"], 9)
    df["MACD_hist"]    = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands: price relative to its rolling mean +/- 2 std
    bb_mean        = close.shift(1).rolling(20).mean()
    bb_std         = close.shift(1).rolling(20).std()
    df["BB_upper"] = bb_mean + 2 * bb_std
    df["BB_lower"] = bb_mean - 2 * bb_std
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]          # volatility proxy
    df["BB_pct"]   = (close.shift(1) - df["BB_lower"]) / (df["BB_width"] + 1e-9)

    # Price vs SMA ratios (where is price relative to trend?)
    df["price_vs_SMA5"]  = close.shift(1) / (df["SMA_5"]  + 1e-9)
    df["price_vs_SMA20"] = close.shift(1) / (df["SMA_20"] + 1e-9)

    # High-Low spread ratio
    df["HL_ratio"] = df["Spread High-Low"] / (close.shift(1) + 1e-9)

    return df


def add_calendar_features(df):
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["month"]     = df[DATE_COL].dt.month
    return df


def add_target(df):
    """Next-day close price. This is what we predict."""
    df["Target"] = df[TARGET_COL].shift(-1)
    return df


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

    df = df.dropna().reset_index(drop=True)

    # Everything except raw OHLC, Date, Target is a feature
    exclude = [DATE_COL, "Target"] + [
        "Open Price", "High Price", "Low Price", "Close Price",
        "WAP", "No.of Shares", "No. of Trades",
        "Total Turnover (Rs.)", "Deliverable Quantity",
        "% Deli. Qty to Traded Qty", "Spread High-Low", "Spread Close-Open"
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"[features] {len(df)} usable rows, {len(feature_cols)} features")
    return df, feature_cols


if __name__ == "__main__":
    from data_loader import load_and_clean
    data = load_and_clean()
    data, fcols = build_features(data)
    print("Sample features:", fcols[:10])
