"""
data_loader.py - Load, clean, and validate the CSV. One job only.
"""

import pandas as pd
from pathlib import Path
from config import BASE_DIR, DATA_PATHS, DATE_COL, DATE_FORMAT, TARGET_COL


def load_and_clean(data_path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # 1. Strip whitespace from column names before validation/parsing
    df.columns = df.columns.str.strip()

    required_cols = {DATE_COL, TARGET_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"{data_path} is missing required columns: {missing}")

    # 2. Parse dates and sort oldest -> newest (raw data is reversed)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 3. Force all non-date columns to numeric
    numeric_cols = [c for c in df.columns if c != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop fully duplicate rows
    df = df.drop_duplicates(subset=DATE_COL).reset_index(drop=True)

    # 5. Detect and remove file-stitch boundary artifacts
    #    A gap > 5 trading days indicates missing data between stitched files.
    #    The boundary row has a corrupt Target (multi-week move labelled as
    #    next-day), and the preceding 50 rows have rolling features that mix
    #    prices from across the gap.
    MAX_GAP_DAYS = 5
    day_gaps = df[DATE_COL].diff().dt.days
    boundary_rows = day_gaps[day_gaps > MAX_GAP_DAYS].index.tolist()

    if boundary_rows:
        print(f"[data_loader] WARNING: {len(boundary_rows)} large date gap(s) "
              f"found at row(s) {boundary_rows}")
        drop_idx = set()
        for br in boundary_rows:
            # Drop boundary row (wrong target) + 50 rows before (corrupt rolling)
            drop_idx.update(range(max(0, br - 50), br + 1))
        df = df.drop(index=list(drop_idx)).reset_index(drop=True)
        print(f"[data_loader] Dropped {len(drop_idx)} rows with corrupted "
              f"targets/features, {len(df)} remaining.")

    print(f"[data_loader] Loaded {data_path}")
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
    import sys

    paths = sys.argv[1:] or DATA_PATHS
    if not paths:
        raise SystemExit("No CSV files found.")

    resolved = []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = BASE_DIR / p
        resolved.append(p)

    for path in resolved:
        data = load_and_clean(path)
        print(get_summary(data))
