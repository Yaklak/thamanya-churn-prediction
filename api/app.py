import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Churn API", version="0.1.0")

BEST_DIR = Path("models/artifacts/best")
MODEL_PATH = BEST_DIR / "model.joblib"
METRICS_PATH = BEST_DIR / "metrics.json"
SCHEMA_PATH = Path("models/artifacts/input_schema.json")

model = None
expected_cols = None
cached_metrics = None


@app.on_event("startup")
def _load():
    global model, expected_cols, cached_metrics
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    expected_cols = None
    cached_metrics = None
    if SCHEMA_PATH.exists():
        expected_cols = json.loads(SCHEMA_PATH.read_text())
    if METRICS_PATH.exists():
        try:
            cached_metrics = json.loads(METRICS_PATH.read_text())
        except Exception:
            cached_metrics = None


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "expects": expected_cols,
    }


@app.get("/model/info")
def model_info():
    if model is None:
        return {
            "loaded": False,
            "detail": "No model loaded. Train first, then ensure models/artifacts/best/model.joblib exists.",
        }

    # Model class name (e.g., 'RandomForestClassifier')
    try:
        clf = getattr(model, "named_steps", {}).get("clf", None)
        model_class = (
            clf.__class__.__name__ if clf is not None else model.__class__.__name__
        )
    except Exception:
        model_class = "unknown"

    # Timestamp from filesystem
    try:
        ts = MODEL_PATH.stat().st_mtime
        from datetime import datetime

        timestamp = datetime.fromtimestamp(ts).isoformat()
    except Exception:
        timestamp = None

    # Metrics from METRICS_PATH (written by training loop)
    info = {
        "loaded": True,
        "model_class": model_class,
        "metrics": cached_metrics,
        "artifact_path": str(MODEL_PATH),
        "timestamp": timestamp,
    }
    return info


@app.get("/model/schema")
def model_schema():
    return {
        "expected_columns": expected_cols,
        "note": "Send a JSON object whose keys exactly match expected_columns.",
    }


@app.post("/predict")
def predict(x: dict):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first, then restart the API.",
        )

    # Build DataFrame from raw dict
    X = pd.DataFrame([x])

    # Validate input columns match what the model saw during training
    if expected_cols:
        missing = [c for c in expected_cols if c not in X.columns]
        extra = [c for c in X.columns if c not in expected_cols]
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "Input columns mismatch",
                    "missing": missing,
                    "extra": extra,
                    "expected_order": expected_cols,
                },
            )
        # Reorder columns to training order
        X = X[expected_cols]

    try:
        # model is a full sklearn Pipeline -> handles encoding internally
        p = model.predict_proba(X)[:, 1][0]
        return {"churn_probability": float(p)}
    except Exception as e:
        # Surface the real reason instead of a generic 500
        tb = traceback.format_exc(limit=2)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e.__class__.__name__}: {e}\n{tb}",
        )
