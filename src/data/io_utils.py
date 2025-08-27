# Save file
from pathlib import Path
import pandas as pd
from datetime import datetime


def save_csv(df: pd.DataFrame, name: str, path: str = "data/processed"):
    processed_dir = Path(path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = processed_dir / f"{name}_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO][io_utils] {out_path} file has been saved.")

    return out_path
