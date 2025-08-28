import pytest
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient
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

    # Fit the dummy model
    X = pd.DataFrame(np.random.rand(10, 3), columns=dummy_schema)
    y = pd.Series(np.random.randint(0, 2, 10))
    dummy_model.fit(X, y)

    # Build client (startup runs with app lifespan)
    test_client = TestClient(fastapi_app)

    # Patch state after startup to inject dummy model & schema
    api_module.app.state.model = dummy_model
    api_module.app.state.input_schema = dummy_schema

    # Also mirror on module-level variables for any legacy access paths (runtime use only)
    setattr(api_module, "model", dummy_model)
    setattr(api_module, "input_schema", dummy_schema)

    yield test_client

    # No teardown necessary; TestClient handles app shutdown.
