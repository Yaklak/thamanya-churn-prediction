# Customer Churn Prediction – Project Documentation

## Executive Summary

This project delivers an end-to-end **customer churn prediction system**. It started with exploratory data analysis (EDA) in Jupyter notebooks to understand user behavior and identify key features driving churn. From there, the project evolved into a production-ready ML pipeline that includes:

- **Feature Engineering**: Transforming raw event logs into enriched user-level metrics.
- **Model Training & Evaluation**: Comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers across metrics like AUC, F1, Precision/Recall, and Lift.
- **Best Model Selection**: Automatically identifying and persisting the best model artifacts.
- **FastAPI Service**: Serving real-time churn predictions and exposing endpoints for health checks, model metadata, schema, and example payloads.
- **Automation & CI/CD**: Makefile-driven commands, pre-commit hooks, GitHub Actions for testing and Docker builds, and Dockerization for reproducible deployment.

The system is designed for **reliability, reproducibility, and scalability**. It allows evaluators or operators to quickly train new models, inspect model performance, and serve predictions via a REST API.

---

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
2. [Feature Engineering](#feature-engineering)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training & Evaluation](#model-training--evaluation)
5. [Model Registry & Best Model Selection](#model-registry--best-model-selection)
6. [FastAPI Service](#fastapi-service)
7. [Automation & Tooling](#automation--tooling)
8. [CI/CD and Dockerization](#cicd-and-dockerization)
9. [API Documentation](#api-documentation)
10. [Quickstart](#quickstart)

---

## Exploratory Data Analysis (EDA)

- Conducted in **Jupyter notebooks** using raw event logs (`ts`, `userId`, `page`, `song`, etc.).
- Explored data distribution, user engagement patterns, churn indicators, and feature correlations.
- Identified missing data and inconsistencies (e.g., empty `userId` rows).
- Produced visuals for:
  - User activity distribution over time.
  - Correlation between listening behavior and churn.
  - Differences in usage between free vs. paid users.

---

## Feature Engineering

Implemented in `src/features/build_features.py` and `src/features/preprocess.py`:

- Aggregated raw logs into **user-level features**:
  - Activity metrics: `events`, `sessions`, `days_active`, `recency_days`.
  - Music interaction: `songs_played`, `unique_songs`, `unique_artists`, `avg_song_len`.
  - Error/status breakdown: `status_200`, `status_307`, `status_404`, `error_events`, `error_rate`.
  - Demographics & subscription: `gender_usage`, `level_free`, `level_paid`, `paid_ratio`.
  - Device & OS usage: iOS, Windows, Mac, Linux ratios.
  - State-level features for geographical analysis.
  - Page interaction success/failure ratios (e.g., `nextsong_success_ratio`, `logout_failed_ratio`).

- Target labels (`churn`, `explicit_churn`) were defined based on user activity and cancellation events.

---

## Data Preprocessing

- Implemented in `src/data/preprocess.py`.
- Steps included:
  - Removing duplicates.
  - Handling missing values (median for numeric, `"unknown"` for categorical).
  - Converting timestamps into tenure and recency features.
- Final processed files are stored in `data/processed/`.

---

## Model Training & Evaluation

Implemented in `scripts/train.py`:

- Models compared:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost

- Evaluation metrics:
  - **ROC-AUC**: ranking quality of predicted probabilities.
  - **Average Precision (PR-AUC)**: quality of precision-recall balance.
  - **F1 Score**: balance of precision and recall.
  - **Accuracy**: overall proportion of correct predictions.
  - **Precision & Recall**: additional class-level performance indicators.
  - **Lift/Gain**: measures business value by comparing model-targeting vs. random selection.

- Automated **multi-model comparison loop** with selection of best model based on primary metric (`roc_auc`) with priority order:
  1. XGBoost
  2. Logistic Regression
  3. Random Forest
  4. Decision Tree

---

## Model Registry & Best Model Selection

- Each model’s trained artifact is saved in `models/artifacts/<model_name_timestamp>/`.
- Metrics for each model are logged in JSON alongside the artifact.
- The **best model** is copied into `models/artifacts/best/model.joblib` for serving.
- Input schema (`input_schema.json`) is generated from training features and stored alongside artifacts.

---

## FastAPI Service

Implemented in `api/app.py`:

- Endpoints:
  - `GET /health` → service status.
  - `POST /predict` → predict churn probability from feature payload.
  - `GET /model/info` → metadata of currently loaded model (class, metrics, timestamp).
  - `GET /model/schema` → expected input feature schema.
  - `GET /model/example` → example input payload (random record from latest processed CSV).

- Auto-loads best model at startup from `models/artifacts/best/model.joblib`.

---

## Automation & Tooling

- **Makefile**:
  - `make install` → install dependencies.
  - `make train` → run training pipeline.
  - `make serve` → start FastAPI app.
  - `make drift` → run data drift detection (via `evidently`).
  - `make docker-build` → build Docker image.
  - `make docker-run` → run container locally.

- **Pre-commit hooks**: black, isort, flake8, pytest.
- **Testing**: Unit tests added under `tests/`.

---

## CI/CD and Dockerization

- **GitHub Actions** workflows:
  - `ci.yml` → runs pre-commit, pytest, and Docker build.
  - `release-docker.yml` → builds & pushes Docker image to registry on tagged releases.

- **Dockerfile**:
  - Multi-stage build for lightweight production images.
  - Final container runs FastAPI via `uvicorn`.
  - Models mounted via `/app/models`.

---

## API Documentation

### `POST /predict` Example

Request:
```json
{
  "events": 400,
  "sessions": 20,
  "days_active": 15,
  "tenure_days": 120,
  "recency_days": 5,
  "songs_played": 350,
  "unique_songs": 180,
  "unique_artists": 120,
  "total_song_length": 80000,
  "avg_song_len": 230,
  "status_200": 300,
  "status_307": 10,
  "status_404": 2,
  "error_events": 12,
  "error_rate": 0.03,
  "gender_usage_f_ratio": 0.4,
  "gender_usage_m_ratio": 0.6,
  "level_free": 0,
  "level_paid": 1,
  "paid_ratio": 1.0,
  "os_usage_ios_ratio": 0.1,
  "os_usage_linux_ratio": 0.0,
  "os_usage_mac_ratio": 0.3,
  "os_usage_windows_ratio": 0.6,
  "nextsong_success_ratio": 0.95,
  "home_success_ratio": 1.0,
  "downgrade_success_ratio": 0.8,
  "logout_failed_ratio": 0.0,
  "settings_success_ratio": 0.9,
  "add_friend_failed_ratio": 0.0,
  "thumbs_up_failed_ratio": 0.0,
  "add_to_playlist_success_ratio": 1.0,
  "thumbs_down_failed_ratio": 0.0,
  "help_success_ratio": 1.0,
  "error_failed_ratio": 0.0,
  "cancel_failed_ratio": 0.0,
  "cancellation_confirmation_success_ratio": 1.0,
  "about_success_ratio": 0.9,
  "roll_advert_success_ratio": 1.0,
  "save_settings_failed_ratio": 0.0,
  "upgrade_success_ratio": 1.0,
  "submit_upgrade_failed_ratio": 0.0,
  "submit_downgrade_failed_ratio": 0.0
}
```

Response:
```json
{
  "churn_probability": 0.27,
  "churn_prediction": false,
  "model_version": "xgboost_2024-06-01T12:34:56",
  "explanation": "The user has a low probability of churn based on recent activity and engagement metrics."
}
```

---

## Quickstart

### Running Locally

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   make install
   ```
3. Train the model:
   ```bash
   make train
   ```
4. Start the FastAPI service:
   ```bash
   make serve
   ```
5. Access the API at `http://localhost:8000`.

### Running with Docker

1. Build the Docker image:
   ```bash
   make docker-build
   ```
2. Run the Docker container:
   ```bash
   make docker-run
   ```
3. The API will be available at `http://localhost:8000`.

---

## Conclusion

This project presents a comprehensive customer churn prediction system that integrates data exploration, feature engineering, model training, and deployment into a seamless pipeline. Leveraging robust evaluation metrics and automated best model selection ensures high-quality predictions. The FastAPI service provides a scalable and user-friendly interface for real-time inference. Combined with CI/CD workflows and Dockerization, the system supports reproducibility, maintainability, and ease of deployment, making it a valuable tool for business stakeholders to proactively manage customer retention.
