# scripts/export_example.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
SCHEMA_PATH = Path("models/artifacts/input_schema.json")
OUT_DIR = Path("examples")
OUT_PATH = OUT_DIR / "example_payload.json"


def find_latest_processed() -> Path:
    cands = sorted(
        PROCESSED_DIR.glob("customer_churn_model_features_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(
            f"No files like {PROCESSED_DIR}/customer_churn_model_features_*.csv"
        )
    return cands[0]


def make_example_from_latest() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "Expected input schema at models/artifacts/input_schema.json. "
            "Run training first."
        )
    expected_cols = json.loads(SCHEMA_PATH.read_text())

    latest = find_latest_processed()
    df = pd.read_csv(latest)

    # Drop potential labels if present
    for tgt in ("explicit_churn", "churn"):
        if tgt in df.columns:
            df = df.drop(columns=[tgt])

    if df.empty:
        raise RuntimeError(f"{latest} has no rows after dropping targets")

    # Select a row: pick a random index
    idx = int(np.random.randint(len(df)))
    row = df.iloc[idx].to_dict()

    # Strictly keep only expected features, fill missing with 0
    example = {k: row.get(k, 0) for k in expected_cols}
    return {
        "note": "Hydrated from latest processed features file; keys/order match schema.",
        "source_file": str(latest),
        "row_index": idx,
        "example": example,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = make_example_from_latest()
    OUT_PATH.write_text(json.dumps(payload["example"], indent=2))
    print(f"[export_example] wrote {OUT_PATH} from {payload['source_file']}")


if __name__ == "__main__":
    main()
