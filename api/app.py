# api/app.py
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

BEST_DIR = Path("models/artifacts/best")
MODEL_PATH = BEST_DIR / "model.joblib"
METRICS_PATH = BEST_DIR / "metrics.json"
SCHEMA_PATH = Path("models/artifacts/input_schema.json")

# module-level fallbacks (still kept for robustness)
model = None
expected_cols: Optional[List[str]] = None
cached_metrics = None


def _load_from_disk():
    m = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    cols = None
    mets = None
    if SCHEMA_PATH.exists():
        try:
            cols = json.loads(SCHEMA_PATH.read_text())
        except Exception:
            cols = None
    if METRICS_PATH.exists():
        try:
            mets = json.loads(METRICS_PATH.read_text())
        except Exception:
            mets = None
    return m, cols, mets


def _get_schema() -> List[str]:
    """Prefer app.state.input_schema, then module globals (input_schema/expected_cols)."""
    try:
        val = getattr(app.state, "input_schema", None)
    except Exception:
        val = None
    if not val:
        # Allow tests (or callers) to patch a simple module-level name
        val = globals().get("input_schema", None)
    if not val:
        val = globals().get("expected_cols", []) or []
    return list(val)


def _get_model():
    """Prefer app.state.model, then module global."""
    try:
        val = getattr(app.state, "model", None)
    except Exception:
        val = None
    if val is None:
        val = globals().get("model", None)
    return val


def _get_metrics():
    try:
        val = getattr(app.state, "metrics", None)
    except Exception:
        val = None
    if val is None:
        val = globals().get("cached_metrics", None)
    return val


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    m, cols, mets = _load_from_disk()

    # set both app.state (preferred) and module globals (fallback)
    app.state.model = m
    app.state.input_schema = cols or []
    app.state.metrics = mets

    globals()["model"] = m
    globals()["expected_cols"] = cols or []
    globals()["cached_metrics"] = mets

    yield
    # --- shutdown: nothing needed ---


app = FastAPI(title="Churn API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    schema = _get_schema()
    return {
        "status": "ok",
        "model_loaded": _get_model() is not None,
        "expects": schema,
    }


@app.get("/model/info")
def model_info():
    m = _get_model()
    if m is None:
        return {
            "loaded": False,
            "detail": "No model loaded. Train first, then ensure models/artifacts/best/model.joblib exists.",
        }
    try:
        clf = getattr(m, "named_steps", {}).get("clf", None)
        model_class = (
            clf.__class__.__name__ if clf is not None else m.__class__.__name__
        )
    except Exception:
        model_class = "unknown"

    try:
        from datetime import datetime

        timestamp = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
    except Exception:
        timestamp = None

    return {
        "loaded": True,
        "model_class": model_class,
        "metrics": _get_metrics(),
        "artifact_path": str(MODEL_PATH),
        "timestamp": timestamp,
    }


@app.get("/model/schema")
def model_schema():
    cols = _get_schema()
    return {"expected_columns": cols, "count": len(cols)}


@app.post("/predict")
def predict(
    x: dict,
    align: bool = Query(
        True, description="If true, auto-align payload to expected schema (default)."
    ),
):
    m = _get_model()
    if m is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first, then restart the API.",
        )

    expected = _get_schema()
    X = pd.DataFrame([x])

    if expected:
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        if (missing or extra) and not align:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "Input columns mismatch",
                    "missing": missing,
                    "extra": extra,
                    "expected_order": expected,
                },
            )
        # align if asked (fill missing with 0, drop extras)
        if align:
            for c in missing:
                X[c] = 0
            X = X[[c for c in expected]]  # drop extras & reorder
        else:
            X = X[expected]

    # Keep as DataFrame so ColumnTransformer can use column names
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_df = X  # keep DataFrame; many pipelines require column names

    # Try multiple inference strategies without failing the API
    # We attempt each method in order; if one fails, we try the next.
    # Only if all attempts fail do we return a default with a warning.
    prob: Optional[float] = None
    warnings: List[str] = []

    # 1) predict_proba -> take positive class
    if hasattr(m, "predict_proba"):
        try:
            prob = float(np.asarray(m.predict_proba(X_df))[:, -1][0])
        except Exception as e:
            warnings.append(f"predict_proba failed: {type(e).__name__}")

    # 2) decision_function -> sigmoid
    if prob is None and hasattr(m, "decision_function"):
        try:
            raw = float(np.asarray(m.decision_function(X_df))[0])
            prob = 1.0 / (1.0 + np.exp(-raw))
        except Exception as e:
            warnings.append(f"decision_function failed: {type(e).__name__}")

    # 3) predict -> cast to {0,1} probability
    if prob is None and hasattr(m, "predict"):
        try:
            pred = float(np.asarray(m.predict(X_df))[0])
            prob = float(np.clip(pred, 0.0, 1.0))
        except Exception as e:
            warnings.append(f"predict failed: {type(e).__name__}")

    # 4) Callable model
    if prob is None and callable(m):
        try:
            out = m(X_df)
            pred = float(np.asarray(out)[0])
            prob = float(np.clip(pred, 0.0, 1.0))
        except Exception as e:
            warnings.append(f"callable model failed: {type(e).__name__}")

    # Finalize response
    if prob is None:
        # Last-resort fallback: don't fail the request; return 0.0 with a warning.
        return {
            "prediction": 0.0,  # backward-compat alias
            "churn_probability": 0.0,
            "warning": "All inference methods failed; returned default 0.0",
            "details": warnings[:3],
        }

    return {
        "prediction": prob,  # backward-compat alias expected by tests
        "churn_probability": prob,
        "warnings": warnings or None,
    }


@app.get("/model/example")
def model_example(minimal: bool = False, idx: int = 0):
    """
    Valid example payload.
    - minimal=true -> zeros skeleton
    - otherwise -> prefer examples/example_payload.json
                   else call exporter to generate it
                   else fallback to zeros skeleton
    """
    expected = _get_schema()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="No input schema found. Train first to generate models/artifacts/input_schema.json.",
        )

    zeros_example = {c: 0 for c in expected}
    if minimal:
        return {
            "note": "Minimal example with zeros for all features.",
            "mode": "minimal",
            "example": zeros_example,
        }

    # 1) Try pre-generated example
    try:
        example_path = Path("examples/example_payload.json")
        if example_path.exists():
            loaded = json.loads(example_path.read_text())
            return {
                "note": "Loaded from examples/example_payload.json",
                "mode": "file",
                "example": {k: loaded.get(k, 0) for k in expected},
            }
    except Exception:
        pass

    # 2) Try regenerate via exporter, then load
    try:
        from scripts.export_example import main as export_example_main

        export_example_main()  # writes examples/example_payload.json (random record)
        example_path = Path("examples/example_payload.json")
        if example_path.exists():
            loaded = json.loads(example_path.read_text())
            return {
                "note": "Generated via scripts/export_example.py and loaded from examples/example_payload.json",
                "mode": "generated",
                "example": {k: loaded.get(k, 0) for k in expected},
            }
    except Exception:
        pass

    # 3) Fallback
    return {
        "note": "Fell back to minimal example.",
        "mode": "fallback_minimal",
        "example": zeros_example,
    }
