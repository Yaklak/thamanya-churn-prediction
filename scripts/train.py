import argparse, yaml
from sklearn.model_selection import train_test_split
from src.data.load import load_dataset
from src.data.preprocess import clean
from src.features.build_features import build
from src.models.pipeline import make_pipeline
from src.models.train_utils import fit_and_tune
from src.models.registry import save_artifacts
from src.utils.metrics import evaluate_all
from pathlib import Path
import json

def main(cfg):
    df = clean(load_dataset())
    X, y, fe = build(df, target=cfg["target"], drop_cols=cfg["features"]["drop"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_seed"]
    )
    pipe = make_pipeline(cfg, fe)
    pipe = fit_and_tune(pipe, X_tr, y_tr, cfg)
    metrics = evaluate_all(pipe, X_te, y_te, cfg["metrics"])
    print("Test metrics:", metrics)
    save_artifacts(pipe, fe, metrics)

    # Save the feature columns the model expects at inference
    schema_path = Path("models/artifacts/input_schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema = list(X_tr.columns)  # columns before encoding (what your API must send)
    schema_path.write_text(json.dumps(schema))
    print(f"Wrote input schema to {schema_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
