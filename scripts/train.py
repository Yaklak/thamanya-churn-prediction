import argparse, json, yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.load import load_raw_events            # JSONL ingestion
from src.data.preprocess import clean                # dedupe + fillna
from src.features.build_features import build_user_features   # aggregate + churn label
from src.features.preprocess import preprocess       # ColumnTransformer + (X,y)
from src.models.pipeline import make_pipeline
from src.models.train_utils import fit_and_tune
from src.models.registry import save_artifacts
from src.utils.metrics import evaluate_all

def main(cfg):
    # 1) Load raw events (JSON Lines)
    data_path = cfg.get("data", {}).get("path", "data/raw/customer_churn_mini.json")
    df_raw = load_raw_events(data_path)

    # 2) Clean raw rows
    df_raw = clean(df_raw)

    # 3) Build aggregated user features + churn
    df_feat = build_user_features(df_raw)

    # (optional) light clean after aggregation
    df_feat = clean(df_feat)

    # 4) Model preprocessing -> X, y, fe
    X, y, fe = preprocess(
        df_feat,
        target=cfg["target"],
        drop_cols=cfg["features"]["drop"],
    )

    # 5) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_seed"]
    )

    # 6) Build + fit
    pipe = make_pipeline(cfg, fe)
    pipe = fit_and_tune(pipe, X_tr, y_tr, cfg)

    # 7) Evaluate + save artifacts
    metrics = evaluate_all(pipe, X_te, y_te, cfg["metrics"])
    print("Test metrics:", metrics)
    save_artifacts(pipe, fe, metrics)

    # Save API input schema (columns BEFORE OHE)
    schema_path = Path("models/artifacts/input_schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(list(X.columns)))
    print(f"Wrote input schema to {schema_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)