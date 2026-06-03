"""
data_loader.py - Load, clean, and validate the CSV. One job only.
"""

import pandas as pd
from config import DATA_PATH, DATE_COL, DATE_FORMAT, TARGET_COL


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # 1. Parse dates and sort oldest -> newest (raw data is reversed)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 2. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 3. Force all numeric columns to float
    numeric_cols = [c for c in df.columns if c != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop fully duplicate rows
    df = df.drop_duplicates(subset=DATE_COL).reset_index(drop=True)

    print(f"[data_loader] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"[data_loader] Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    missing = df.isnull().sum()
    if missing.any():
        print("[data_loader] Missing values:\n", missing[missing > 0])
    else:
        print("[data_loader] No missing values.")

    return df


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


if __name__ == "__main__":
    data = load_and_clean()
    print(get_summary(data))
