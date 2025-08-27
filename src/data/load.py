# Get raw data into a DataFrame.
import pandas as pd


def load_raw_events(path: str = "data/raw/customer_churn_mini.json") -> pd.DataFrame:
    """Read newline-delimited JSON log (one JSON object per line)."""
    return pd.read_json(path, lines=True)
