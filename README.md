# Churn Prediction

Production-ready churn prediction pipeline with feature engineering, **multi-model training** (Logistic Regression, Decision Tree, Random Forest, XGBoost), **model selection & registry** with tie-breaking priority for XGBoost, and a **FastAPI** service for real-time scoring. Includes schema-aware payload validation, example payloads, pre-commit hooks, and Docker packaging.

[![CI](https://github.com/Yaklak/thamanya-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Yaklak/thamanya-churn-prediction/actions/workflows/ci.yml)

## Features

- **End-to-end pipeline**
  - `src/data` → load/clean events
  - `src/features` → feature engineering + modeling preprocess
  - `scripts/train.py` → trains 4 models, evaluates, **selects best with tie-breaking priority for XGBoost**, saves to registry
- **Model registry**
  - Saves each run to `models/artifacts/<modelname>_<timestamp>/`
  - Copies the best model to `models/artifacts/best/`
  - Persists the **input schema** used during training: `models/artifacts/input_schema.json`
- **FastAPI service**
  - `/health` – service & schema status
  - `/model/info` – which model is loaded + metrics
  - `/model/schema` – expected feature columns
  - `/model/example` – example payload matching schema, supports random real rows
  - `/predict` – returns `prediction` and `probability`
- **Data**
  - Processed data files saved as CSVs in `data/processed/*.csv`
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
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
make install
make install-dev
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
- `GET /model/example`: Get an example payload for the model, supports random real rows.
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

**Query Parameters:**

- `minimal` (boolean, optional): If set to `true`, returns a minimal example payload with all feature values set to zero. Useful for understanding the required schema without real data.
- `idx` (integer, optional): Specifies a row index from the processed dataset to use as the example payload. If provided, returns the real feature values from that row. If both `minimal` and `idx` are provided, `idx` takes precedence.

If no query parameters are provided, the endpoint attempts to load an example payload from `examples/example_payload.json`. If this file is not available, it falls back to returning a random real row from the processed data.

**Response:**

```json
{
  "note": "Loaded from examples/example_payload.json or random real rows",
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

This endpoint accepts a JSON payload matching the expected input schema and returns a churn prediction with the associated probability.

**Request:**

```json
{
  "events": 10,
  "sessions": 3,
  "total_listens": 25,
  "avg_session_length": 300,
  "is_premium_user": 1,
  "days_since_last_session": 5,
  ...
}
```

**Response:**

```json
{
  "churn_probability": 0.15
}
```

### Detailed `/predict` Example

**Sample Request Payload:**

```json
{
  "events": 12,
  "sessions": 4,
  "total_listens": 30,
  "avg_session_length": 350,
  "is_premium_user": 0,
  "days_since_last_session": 7,
  "num_artists_listened": 5,
  "num_songs_listened": 20,
  "avg_song_length": 210,
  "num_skips": 2
}
```

**Sample Response:**

```json
{
  "churn_probability": 0.72
}
```

**Interpretation:**

The model predicts a 72% probability that the user will churn (stop using the service). This high value suggests the user is at risk and may benefit from retention efforts. Lower probabilities indicate lower risk of churn.

### Error Handling

- **422 Unprocessable Entity:** Returned when the request payload does not conform to the expected schema. This can happen if required fields are missing, extra fields are present, or data types are incorrect. The response includes details about the validation errors.

  **Example:**

  ```json
  {
    "detail": [
      {
        "loc": ["body", "events"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
  ```

- **500 Internal Server Error:** Returned when an unexpected error occurs on the server, such as issues loading the model or processing the request. The response may include a message indicating the error.

  **Note:** If you encounter repeated 500 errors, check the server logs for details and ensure the model is properly loaded.

### Interactive Docs

An interactive API documentation interface is available at the `/docs` endpoint. This interface allows you to explore the API endpoints, view request/response schemas, and make test calls directly from your browser.

## Quickstart: Real Example Payload

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

The project trains four different models:

- **Logistic Regression:** A simple linear model that is easy to interpret.
- **Decision Tree:** A non-linear model that is easy to visualize.
- **Random Forest:** An ensemble of decision trees that is more robust and accurate than a single decision tree.
- **XGBoost:** A powerful gradient boosting model that often achieves state-of-the-art performance.

The best model is selected based on the ROC AUC score on the test set, with tie-breaking priority given to XGBoost. The best model is then saved to the `models/artifacts/best` directory.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **ROC AUC:** Measures the ability of the model to distinguish between classes.
- **Average Precision:** Summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold.
- **F1 Score:** Harmonic mean of precision and recall.
- **Accuracy:** Overall correctness of the model.
- **Precision:** Proportion of positive identifications that were actually correct.
- **Recall:** Proportion of actual positives that were identified correctly.
- **Lift/Gain:** Measures the effectiveness of the model at identifying positive cases compared to random selection.

## Docker

```
docker build -t yaklak/thamanya-churn:latest -f Dockerfile .
docker run --rm -p 8000:8000 \
  -v "$PWD/models/artifacts:/app/models/artifacts" \
  yaklak/thamanya-churn:latest
```

## Quickstart with Docker

```bash
# 1. Build the Docker image
docker build -t yaklak/thamanya-churn:latest -f Dockerfile .

# 2. Run the container with port mapping and volume mount
docker run --rm -p 8000:8000 \
  -v "$PWD/models/artifacts:/app/models/artifacts" \
  yaklak/thamanya-churn:latest

# 3. Check health endpoint
curl http://127.0.0.1:8000/health

# 4. Make a prediction using example payload
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/example_payload.json
```

## CI/CD

This project includes GitHub Actions workflows:

- **CI:** Runs pre-commit checks, pytest tests, and builds the Docker image on every push.
- **release-docker:** Automatically builds and publishes Docker images to Docker Hub when a new tag is pushed.

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
