# Sequence of Execution

This document details the step-by-step execution flow of the Iris ML project, from infrastructure provisioning to model inference.

## 1. CI/CD Pipeline Execution (GitHub Actions)
Every time code is pushed to the repository, the following sequence occurs in `.github/workflows/cicd.yml`.

### Phase 1: Validation (All Environments)
1.  **Lint & Unit Tests**: Running `pytest` on the Python codebase to ensure logic is correct.
2.  **Terraform Validate**: Ensuring infrastructure code is syntactically valid.

### Phase 2: Deploy to Dev
This phase runs automatically on push to main/develop branches.

1.  **Infrastructure Provisioning (Terraform)**
    *   **IAM**: Creates a Service Account for Cloud Run.
    *   **BigQuery**: Creates the `iris_dev` dataset and `iris_raw` table (empty schema provided).
    *   **GCS**: Creates buckets for data and Terraform state.
    *   **Artifact Registry**: Creates a Docker repository.
    *   **Cloud Run**: Creates/Updates the service (scaled down for dev).

2.  **Data Pipeline Execution (Python Scripts)**
    *   **`src/bq_ingestion.py`**: Loads the raw Iris dataset (from sklearn) into BigQuery table `iris_raw`. Note: This uses `WRITE_TRUNCATE`, so it refreshes data on each run.
    *   **`src/load_data_from_bq.py`**: Queries BigQuery `iris_raw` and exports it as a CSV file to GCS (`gs://.../data/raw/iris.csv`).
    *   **`src/clean_raw_data.py`**: Reads raw CSV from GCS, performs cleaning (deduplication, null removal), and saves to `gs://.../data/clean/iris_clean.csv`.
    *   **`src/transform_data.py`**:
        *   Reads cleaned CSV.
        *   Fits a `StandardScaler`.
        *   Saves the **Scaler Artifact** to `gs://.../artifacts/scalers/scaler.pkl` (critical for inference).
        *   Saves transformed data to `gs://.../data/transformed/iris_transformed.csv`.
    *   **`src/train.py`**:
        *   Reads transformed data.
        *   Trains a RandomForest model.
        *   Saves the **Model Artifact** to both:
            *   `gs://.../artifacts/models/{GIT_SHA}/model.pkl` (Versioned)
            *   `gs://.../artifacts/models/latest/model.pkl` (Latest pointer)

3.  **Application Containerization**
    *   **Docker Build**: Builds the `app.py` image. *Note: The model is NOT baked into the image.*
    *   **Docker Push**: Pushes the image to Artifact Registry (tagged with commit SHA).

4.  **Service Deployment**
    *   **Cloud Run Deploy**: Deploys the new image. The service starts up.
    *   **Startup Logic (`app.py` / `inference.py`)**:
        *   On container start (or first request), the app downloads `model.pkl` and `scaler.pkl` from GCS using the `latest` or configured prefix.
        *   The model and scaler are cached in memory.

5.  **Integration Testing**
    *   Runs live HTTP requests against the deployed Dev URL to verify `/predict` works.

### Phase 3: Promote to QA (Manual Approval)
1.  **Terraform Apply (QA)**: Provisions QA infrastructure.
2.  **Image Promotion**: Pulls the Docker image from Dev registry, re-tags it, and pushes to QA registry. *No rebuild occurs.*
3.  **Deploy to QA**: Deploys the exact same image to Cloud Run (QA environment).
4.  **QA Validation**: Runs integration tests against the QA URL.

### Phase 4: Promote to Prod (Manual Approval)
1.  **Terraform Apply (Prod)**: Provisions Prod infrastructure (High availability settings).
2.  **Image Promotion**: Promotes image from QA to Prod registry.
3.  **Deploy to Prod**: Deploys to Prod Cloud Run.
4.  **Smoke Test**: Verifies a single prediction on the live Prod endpoint.

---

## 2. Runtime Request Flow (Inference)
When a user makes a POST request to `/predict`:

1.  **FastAPI (`src/app.py`)**: Receives the JSON payload.
2.  **Inference (`src/inference.py`)**:
    *   **Lazy Loading**: Checks if model/scaler are loaded in memory. If not, downloads them from GCS (latency hit on first request/cold start).
    *   **Preprocessing**: Uses the loaded `scaler` to transform input features.
    *   **Feature Engineering**: Calculates `petal_area` and `sepal_area` on the fly (same logic as training pipeline).
    *   **Prediction**: Passes engineered features to the RandomForest model.
3.  **Response**: Returns class name and probabilities.
