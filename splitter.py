"""
splitter.py - Chronological train/test split. No shuffle. One job only.
"""

import pandas as pd
from config import TRAIN_RATIO


def chronological_split(df: pd.DataFrame, feature_cols: list[str]):
    """
    80% oldest rows = train, 20% newest rows = test.
    Returns X_train, X_test, y_train, y_test, split_idx.
    """
    split_idx = int(len(df) * TRAIN_RATIO)

    X = df[feature_cols]
    y = df["Target"]

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test  = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test  = y.iloc[split_idx:].reset_index(drop=True)

    print(f"[splitter] Train: {len(X_train)} rows | Test: {len(X_test)} rows | Split at index {split_idx}")
    return X_train, X_test, y_train, y_test, split_idx
