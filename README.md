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
  - Pre-commit hooks (black, isort, flake8, pytest)
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
├── environment.yml
└── README.md
```

## Setup (Local)

**Conda**

```
conda env create -f environment.yml
conda activate thamanya
```

**pip**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Models

```
make train
```

## Serve the API

```
make serve
```


### Endpoints

- `GET /health`
- `GET /model/info`
- `GET /model/schema`
- `GET /model/example`
- `POST /predict`

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
