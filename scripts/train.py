# orchestrates: load data → preprocess → build pipeline(s) → fit & tune → evaluate → save artifacts (+ best)

import argparse
import json
from copy import deepcopy
from pathlib import Path

import joblib
import yaml  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Optional: XGBoost (will be skipped if not available)
try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False

from src.data.load import load_raw_events  # JSONL ingestion
from src.data.preprocess import clean  # basic cleaning
from src.features.build_features import build_user_features  # aggregate + churn label
from src.features.preprocess import preprocess  # ColumnTransformer + (X,y)
from src.models.pipeline import make_pipeline
from src.models.registry import save_artifacts
from src.models.train_utils import fit_and_tune
from src.utils.metrics import evaluate_all


def _save_artifacts_compat(pipe, metrics, model_name, fe=None):
    """Call registry.save_artifacts with either the new or old signature.
    - New: save_artifacts(pipe, metrics, model_name=...)
    - Old: save_artifacts(pipe, fe, metrics)
    """
    try:
        return save_artifacts(pipe, metrics, model_name=model_name)
    except TypeError:
        if fe is None:
            raise
        return save_artifacts(pipe, fe, metrics)


# Rank models for tie-breaking: xgboost > logreg > random_forest > others (e.g., decision_tree)
def _priority(model_name: str) -> int:
    name = model_name.lower()
    if name == "xgboost":
        return 3
    elif name == "logreg":
        return 2
    elif name == "random_forest":
        return 1
    else:
        return 0


def main(cfg):
    # 1) Load raw events (JSON Lines)
    data_path = cfg.get("data", {}).get("path", "data/raw/customer_churn_mini.json")
    df_raw = load_raw_events(data_path)

    # 2) Clean raw rows
    df_raw = clean(df_raw, drop_cols=cfg["features"].get("drop", []))

    # 3) Build aggregated user features + churn (notebook-aligned). Do NOT pass target.
    df_feat = build_user_features(
        df=df_raw,
        inactivity_days=cfg["features"].get("inactivity_days", 30),
        target=cfg.get("target", "churn"),
    )

    # 4) Model preprocessing -> X, y, fe
    X, y, fe = preprocess(
        df=df_feat,
        target=cfg["target"],
        drop_cols=cfg["features"].get("drop", []),
        scale_numeric=True,  # helpful for Logistic Regression; neutral for trees
    )

    # 5) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg.get("test_size", 0.2),
        stratify=y,
        random_state=cfg.get("random_seed", 42),
    )

    # 6) Define candidate estimators (conditionally include XGBoost)
    candidates = {
        "logreg": LogisticRegression(max_iter=500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
    }
    if _HAS_XGB and XGBClassifier is not None:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        print("[WARN] xgboost not available; skipping XGBoost candidate.")

    trained = {}  # name -> {"pipe": pipe, "metrics": metrics}

    # 7) Train/evaluate each model (sequential; can be parallelized later)
    for name, est in candidates.items():
        fe_local = deepcopy(fe)  # each model gets a fresh copy of preprocessing
        pipe = make_pipeline(cfg, fe_local, model_override=est)
        pipe = fit_and_tune(
            pipe, X_tr, y_tr, cfg
        )  # runs GridSearchCV if cfg['tuning'] provided
        metrics = evaluate_all(
            pipe, X_te, y_te, cfg.get("metrics", ["roc_auc", "average_precision", "f1"])
        )
        rounded_metrics = {
            k: (round(v, 2) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
        }
        print(f"\n[RESULT] {name} metrics:", rounded_metrics)
        trained[name] = {"pipe": pipe, "metrics": metrics, "fe": fe_local}

    # 8) Persist per-model artifacts
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    results = {}
    for name, pack in trained.items():
        results[name] = pack["metrics"]
        _save_artifacts_compat(
            pack["pipe"], pack["metrics"], model_name=name, fe=pack["fe"]
        )  # per-model run dir

    Path("models/artifacts/metrics_multi.json").write_text(
        json.dumps(results, indent=2)
    )

    # 9) Select BEST model by primary metric (roc_auc) and copy to models/artifacts/best/
    primary = cfg.get("primary_metric", "roc_auc")
    best_name = max(
        results,
        key=lambda k: (
            results[k].get(primary, float("-inf")),
            _priority(k),
        ),
    )
    best_pipe = trained[best_name]["pipe"]
    best_metrics = trained[best_name]["metrics"]

    best_dir = Path("models/artifacts/best")
    best_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, best_dir / "model.joblib")
    (best_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2))
    print(
        f"\n---------------------------\n"
        f"[RESULT] BEST model: {best_name} → saved to {best_dir}\n"
        f"Primary metric ({primary}): {best_metrics.get(primary, 0):.2f}"
    )

    # 10) Save API input schema (columns BEFORE OHE)
    schema_path = Path("models/artifacts/input_schema.json")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(list(X.columns)))
    print(f"Wrote input schema to {schema_path}")

    # Auto-create a real example payload for README/tests
    try:
        from scripts.export_example import main as export_example_main

        export_example_main()
    except Exception as e:
        print(f"[WARN] Could not export example payload: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
