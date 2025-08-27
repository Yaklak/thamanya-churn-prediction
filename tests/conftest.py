import pytest
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient
import joblib
import numpy as np


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).parent.parent


@pytest.fixture
def tmp_path(tmpdir) -> Path:
    """Create a temporary directory for testing."""
    return Path(tmpdir)


@pytest.fixture
def raw_data_path(tmp_path: Path) -> Path:
    """Create a dummy raw data file and return its path."""
    raw_data = [
        {
            "ts": 1630000000000,
            "userId": "1",
            "sessionId": "1",
            "page": "Home",
            "auth": "Logged In",
            "method": "GET",
            "status": 200,
            "level": "free",
            "itemInSession": 0,
            "location": "New York, NY",
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "lastName": "Smith",
            "firstName": "John",
            "registration": 1620000000000,
            "gender": "M",
        },
        {
            "ts": 1630000001000,
            "userId": "1",
            "sessionId": "1",
            "page": "nextsong",
            "auth": "Logged In",
            "method": "PUT",
            "status": 200,
            "level": "free",
            "itemInSession": 1,
            "location": "New York, NY",
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "lastName": "Smith",
            "firstName": "John",
            "registration": 1620000000000,
            "gender": "M",
            "song": "Song A",
            "artist": "Artist A",
            "length": 240.0,
        },
        {
            "ts": 1630000002000,
            "userId": "",
            "sessionId": "2",
            "page": "Home",
            "auth": "Guest",
            "method": "GET",
            "status": 200,
            "level": "free",
            "itemInSession": 0,
            "location": "Los Angeles, CA",
            "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "lastName": None,
            "firstName": None,
            "registration": None,
            "gender": None,
        },
        {
            "ts": 1626544004000,
            "userId": "2",
            "sessionId": "3",
            "page": "Home",
            "auth": "Logged In",
            "method": "GET",
            "status": 200,
            "level": "paid",
            "itemInSession": 0,
            "location": "Chicago, IL",
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1",
            "lastName": "Doe",
            "firstName": "Jane",
            "registration": 1610000000000,
            "gender": "F",
        },
    ]
    raw_data_file = tmp_path / "log.json"
    with open(raw_data_file, "w") as f:
        for item in raw_data:
            f.write(json.dumps(item) + "\n")
    return raw_data_file


@pytest.fixture
def raw_events_df() -> pd.DataFrame:
    """Create a sample raw events DataFrame."""
    data = [
        {
            "userId": "1",
            "ts": 1630000000000,
            "sessionId": 1,
            "page": "Home",
            "auth": "Logged In",
            "method": "GET",
            "status": 200,
            "level": "free",
            "itemInSession": 0,
            "location": "New York, NY",
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "lastName": "Smith",
            "firstName": "John",
            "registration": 1620000000000,
            "gender": "M",
        },
        {
            "userId": "1",
            "ts": 1630000001000,
            "sessionId": 1,
            "page": "nextsong",
            "auth": "Logged In",
            "method": "PUT",
            "status": 200,
            "level": "free",
            "itemInSession": 1,
            "location": "New York, NY",
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "lastName": "Smith",
            "firstName": "John",
            "registration": 1620000000000,
            "gender": "M",
            "song": "Song A",
            "artist": "Artist A",
            "length": 240.0,
        },
        {
            "userId": "2",
            "ts": 1626544004000,
            "sessionId": 3,
            "page": "Home",
            "auth": "Logged In",
            "method": "GET",
            "status": 200,
            "level": "paid",
            "itemInSession": 0,
            "location": "Chicago, IL",
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)",
            "lastName": "Doe",
            "firstName": "Jane",
            "registration": 1610000000000,
            "gender": "F",
        },
        {
            "userId": "1",
            "ts": 1630000004000,
            "sessionId": 1,
            "page": "Thumbs Up",
            "auth": "Logged In",
            "method": "PUT",
            "status": 200,
            "level": "free",
            "itemInSession": 2,
            "location": "New York, NY",
            "userAgent": "Mozilla/5.0 (Windows NT 10..0; Win64; x64)",
            "lastName": "Smith",
            "firstName": "John",
            "registration": 1620000000000,
            "gender": "M",
        },
    ]
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms")
    df["event_date"] = df["ts"].dt.date
    return df


@pytest.fixture
def fe_transformer() -> ColumnTransformer:
    """Create a dummy feature engineering transformer."""
    return ColumnTransformer([], remainder="passthrough")


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Create a test client for the API, with a dummy model."""
    from api.app import app as fastapi_app
    from api import app as api_module

    # Create dummy model artifacts
    dummy_schema = ["a", "b", "c"]
    dummy_model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [("passthrough", "passthrough", ["a", "b", "c"])], remainder="drop"
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    )
    dummy_metrics = {"roc_auc": 0.9, "f1": 0.8}

    # Fit the dummy model
    X = pd.DataFrame(np.random.rand(10, 3), columns=dummy_schema)
    y = pd.Series(np.random.randint(0, 2, 10))
    dummy_model.fit(X, y)

    # Save the dummy artifacts
    best_dir = tmp_path / "models" / "artifacts" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy_model, best_dir / "model.joblib")
    with open(best_dir / "metrics.json", "w") as f:
        json.dump(dummy_metrics, f)

    schema_path = tmp_path / "models" / "artifacts" / "input_schema.json"
    with open(schema_path, "w") as f:
        json.dump(dummy_schema, f)

    # Monkeypatch the paths in the app
    original_model_path = api_module.MODEL_PATH
    original_schema_path = api_module.SCHEMA_PATH
    api_module.MODEL_PATH = best_dir / "model.joblib"
    api_module.SCHEMA_PATH = schema_path

    # Reload the app to pick up the new paths
    api_module._load()

    yield TestClient(fastapi_app)

    # Restore the original paths
    api_module.MODEL_PATH = original_model_path
    api_module.SCHEMA_PATH = original_schema_path
    api_module._load()
