# cleaning: duplicates, missing values
from __future__ import annotations
import pandas as pd
import logging
from src.data.io_utils import save_csv

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Normalize raw events to a consistent, analysis-ready shape.

    1- Coerce timestamps and registration to datetimes
    2- Drop rows with missing/blank userId or page
    3- Normalize string columns (trim, lower where appropriate)
    4- Provide helper columns used later by labeling/feature blocks:
        ts (datetime), event_date (date);
    """
    df = df.copy()

    # --- 1) Basic type fixes
    for col in ("userId", "sessionId"):
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' in raw data.")
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].ne("")]

    # timestamps in ms -> datetime
    for col in ("ts", "registration"):
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' (ms since epoch) in raw data.")
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")

    # --- 2) Drop unusable rows / columns
    # drop duplicates records
    df = df.drop_duplicates()

    # drop rows without userId or ts
    df = df.dropna(subset=["userId", "ts"])  # drop NaN in both
    df = df[df["userId"].astype(str).ne("")]  # drop empty strings

    # drop columns (not predictive)
    before = set(df.columns)
    df = df.drop(columns=drop_cols, errors="ignore")
    removed = before - set(df.columns)
    # sanity check: confirm expected columns were removed
    assert removed == set(
        drop_cols
    ), f"Unexpected removed cols: {removed} vs {drop_cols}"
    logger.debug("Removed columns: %s", removed)

    # --- 3) Normalize/standardize key categoricals
    for col in ("page", "level", "gender", "userAgent", "location", "method", "auth"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        else:
            df[col] = "unknown"

    # --- 4) Lightweight NA handling (numerics -> median, categoricals -> 'unknown')
    # Note: heavy imputations happen later on the aggregated user table.
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].fillna("unknown")

    # --- 5) Helper columns used later by labels & features
    if "ts" in df.columns:
        df["event_date"] = df["ts"].dt.normalize()

    # (length) keep numeric; clip tiny/huge outliers very conservatively
    if "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        df["length"] = df["length"].clip(
            lower=0
        )  # leave upper as-is; model can learn tails

    # Sort for any later rolling/session logic
    df = df.sort_values(["userId", "sessionId", "ts"], kind="mergesort")

    # --- 6) Save cleaned file to data/processed
    save_csv(df, "customer_churn_cleaned", "data/processed")

    return df
