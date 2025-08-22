from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib, pandas as pd, json, traceback

app = FastAPI(title="Churn API", version="0.1.0")

MODEL_PATH = Path("models/registry/latest_model.joblib")
SCHEMA_PATH = Path("models/artifacts/input_schema.json")

model = None
expected_cols = None

@app.on_event("startup")
def _load():
    global model, expected_cols
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    expected_cols = None
    if SCHEMA_PATH.exists():
        expected_cols = json.loads(SCHEMA_PATH.read_text())

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

# Pydantic model with broad types; we'll validate keys at runtime.
class Payload(BaseModel):
    events: int
    unique_songs: int
    unique_artists: int
    avg_song_len: float
    tenure_days: int
    plan_tier: str = Field(..., examples=["paid", "free", "basic"])
    gender: str = Field(..., examples=["M","F","U"])

@app.post("/predict")
def predict(x: Payload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first, then restart the API.")

    X = pd.DataFrame([x.dict()])

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

    try:
        # model is a full sklearn Pipeline -> handles encoding internally
        p = model.predict_proba(X)[:, 1][0]
        return {"churn_probability": float(p)}
    except Exception as e:
        # Surface the real reason instead of a generic 500
        tb = traceback.format_exc(limit=2)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e.__class__.__name__}: {e}\n{tb}")