# Thamanya Churn Prediction

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
\`\`\`
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
\`\`\`

## Setup (Local)

**Conda**
\`\`\`bash
conda env create -f environment.yml
conda activate thamanya
\`\`\`

**pip**
\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Train Models
\`\`\`bash
make train
\`\`\`

## Serve the API
\`\`\`bash
make serve
\`\`\`

### Endpoints
- GET /health
- GET /model/info
- GET /model/schema
- GET /model/example
- POST /predict

## Docker
\`\`\`bash
docker build -t yaklak/thamanya-churn:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD/models/artifacts:/app/models/artifacts" \
  yaklak/thamanya-churn:latest
\`\`\`

## Tests
\`\`\`bash
pytest -q
\`\`\`

## Troubleshooting
- **Perfect scores:** check for data leakage or small test sets.
- **Payload rejected:** use `/model/schema` or `/model/example`.
- **XGBoost import error on macOS:** install `brew install libomp` or use Docker.

## License
MIT
