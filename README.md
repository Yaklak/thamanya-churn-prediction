# Churn Prediction

Production-ready churn prediction pipeline with feature engineering, **multi-model training** (Logistic Regression, Decision Tree, Random Forest), **model selection & registry**, and a **FastAPI** service for real-time scoring. Includes schema-aware payload validation, example payloads, pre-commit hooks, and Docker packaging.

## Features

- **End-to-end pipeline**
  - `src/data` → load/clean events
  - `src/features` → feature engineering + modeling preprocess
  - `scripts/train.py` → trains 3 models, evaluates, **selects best**, saves to registry
- **Model registry**
  - Saves each run to `models/artifacts/<modelname>_<timestamp>/`
  - Copies the best model to `models/artifacts/best/`
  - Persists the **input schema** used during training: `models/artifacts/input_schema.json`
- **FastAPI service**
  - `/health` – service & schema status
  - `/model/info` – which model is loaded + metrics
  - `/model/schema` – expected feature columns
  - `/model/example` – example payload matching schema
  - `/predict` – returns `prediction` and `probability`
- **Tooling**
  - `Makefile` for repetitive tasks
  - Pre-commit hooks (ruff, black, isort, mypy, pytest)
  - Dockerfile with OpenMP support for XGBoost

## Project Structure

```
.
├── api/
│   └── app.py                # FastAPI app, endpoints, model loading
├── configs/
│   └── training.yaml         # Training configuration
├── data/                     # (local) raw/processed data
├── models/
│   └── artifacts/            # Saved runs and best/ model
├── notebooks/                # Jupyter exploration
├── scripts/
│   └── train.py              # Training entrypoint
├── src/                      # data, features, utils
├── tests/                    # pytest tests
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Setup (Local)

**pip**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Train Models

```
make train
```

## Serve the API

```
make serve
```

## API Documentation

### Endpoints

- `GET /health`: Health check endpoint.
- `GET /model/info`: Get information about the loaded model.
- `GET /model/schema`: Get the input schema of the model.
- `GET /model/example`: Get an example payload for the model.
- `POST /predict`: Make a prediction.

### `GET /health`

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "expects": [
    "events",
    "sessions",
    ...
  ]
}
```

### `GET /model/info`

**Response:**

```json
{
  "loaded": true,
  "model_class": "DecisionTreeClassifier",
  "metrics": {
    "roc_auc": 1.0,
    "average_precision": 1.0,
    "f1": 1.0,
    "threshold": 0.5,
    "positives_rate": 0.14444444444444443
  },
  "artifact_path": "models/artifacts/best/model.joblib",
  "timestamp": "2025-08-27T13:57:30.123456"
}
```

### `GET /model/schema`

**Response:**

```json
{
  "expected_columns": [
    "events",
    "sessions",
    ...
  ],
  "note": "Send a JSON object whose keys exactly match expected_columns."
}
```

### `GET /model/example`

**Response:**

```json
{
  "note": "Loaded from examples/example_payload.json",
  "mode": "file",
  "source_file": "examples/example_payload.json",
  "example": {
    "events": 0,
    "sessions": 0,
    ...
  }
}
```

### `POST /predict`

**Request:**

```json
{
  "events": 0,
  "sessions": 0,
  ...
}
```

**Response:**

```json
{
  "churn_probability": 0.5
}
```

### Quickstart: Real Example Payload

After running `make train`, a real schema-aligned example is automatically generated at:

```
examples/example_payload.json
```

You can send it directly to the API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/example_payload.json | jq
```

You can also fetch examples dynamically from the API:

```bash
# Real row from latest processed features
curl 'http://127.0.0.1:8000/model/example' | jq

# Minimal all-zeros skeleton
curl 'http://127.0.0.1:8000/model/example?minimal=true' | jq

# Pick a specific row (e.g., row 5)
curl 'http://127.0.0.1:8000/model/example?idx=5' | jq
```

## Data

The dataset used in this project is a synthetic dataset of user events from a music streaming service. The dataset is in JSONL format and contains the following fields:

- `ts`: timestamp of the event
- `userId`: ID of the user
- `sessionId`: ID of the session
- `page`: page visited by the user
- `auth`: authentication status of the user
- `method`: HTTP method of the request
- `status`: HTTP status of the response
- `level`: level of the user (free or paid)
- `itemInSession`: item number in the session
- `location`: location of the user
- `userAgent`: user agent of the user
- `lastName`: last name of the user
- `firstName`: first name of the user
- `registration`: registration timestamp of the user
- `gender`: gender of the user
- `song`: name of the song
- `artist`: name of the artist
- `length`: length of the song

## Models

The project trains three different models:

- **Logistic Regression:** A simple linear model that is easy to interpret.
- **Decision Tree:** A non-linear model that is easy to visualize.
- **Random Forest:** An ensemble of decision trees that is more robust and accurate than a single decision tree.

The best model is selected based on the ROC AUC score on the test set. The best model is then saved to the `models/artifacts/best` directory.

## Docker

```
docker build -t yaklak/thamanya-churn:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD/models/artifacts:/app/models/artifacts" \
  yaklak/thamanya-churn:latest
```

## Tests

```
pytest -q
```

## Troubleshooting

- **Perfect scores:** check for data leakage or small test sets.
- **Payload rejected:** use `/model/schema` or `/model/example`.
- **XGBoost import error on macOS:** install `brew install libomp` or use Docker.

## License

MIT
