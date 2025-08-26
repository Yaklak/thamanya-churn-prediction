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


@app.get("/model/example")
def model_example():
    """
    Return a valid example payload whose keys exactly match the input schema.
    Values are sensible defaults; edit as needed before calling /predict.
    """
    if expected_cols is None:
        raise HTTPException(
            status_code=500,
            detail="No input schema found. Train first to generate models/artifacts/input_schema.json.",
        )

    # Start with zeros for every expected feature
    example = {c: 0 for c in expected_cols}

    # Helper to set a value only if the column exists in the schema
    def setv(k, v):
        if k in example:
            example[k] = v

    # Populate a few reasonable defaults (only applied if present in schema)
    setv("events", 120)
    setv("sessions", 10)
    setv("days_active", 25)
    setv("tenure_days", 180)
    setv("recency_days", 2)
    setv("songs_played", 450)
    setv("unique_songs", 300)
    setv("unique_artists", 220)
    setv("total_song_length", 95000)
    setv("avg_song_len", 211.1)
    setv("status_200", 118)
    setv("status_307", 1)
    setv("status_404", 1)
    setv("error_events", 2)
    setv("error_rate", 0.016)
    setv("gender_usage_f_ratio", 0.45)
    setv("gender_usage_m_ratio", 0.55)
    setv("level_free", 0.3)
    setv("level_paid", 0.7)
    setv("paid_ratio", 0.7)
    setv("os_usage_ios_ratio", 0.2)
    setv("os_usage_linux_ratio", 0.1)
    setv("os_usage_mac_ratio", 0.3)
    setv("os_usage_windows_ratio", 0.4)
    setv("nextsong_success_ratio", 0.98)
    setv("home_success_ratio", 0.95)
    setv("downgrade_success_ratio", 1.0)
    setv("logout_failed_ratio", 0.0)
    setv("settings_success_ratio", 1.0)
    setv("add_friend_failed_ratio", 0.0)
    setv("thumbs_up_failed_ratio", 0.01)
    setv("add_to_playlist_success_ratio", 0.9)
    setv("thumbs_down_failed_ratio", 0.02)
    setv("help_success_ratio", 1.0)
    setv("error_failed_ratio", 0.0)
    setv("cancel_failed_ratio", 0.0)
    setv("cancellation_confirmation_success_ratio", 1.0)
    setv("about_success_ratio", 1.0)
    setv("roll_advert_success_ratio", 0.95)
    setv("save_settings_failed_ratio", 0.0)
    setv("upgrade_success_ratio", 0.85)
    setv("submit_upgrade_failed_ratio", 0.0)
    setv("submit_downgrade_failed_ratio", 0.0)

    return {
        "note": "Edit values as needed and POST to /predict. Keys and order already match the trained model schema.",
        "example": example,
    }
