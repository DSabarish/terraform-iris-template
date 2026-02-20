# Iris ML — Production-Ready ML Project on GCP

A complete end-to-end ML project following MLOps best practices:
**BigQuery → GCS → Clean → Transform → Train → Containerize → Cloud Run**

## Architecture

```
BigQuery (raw)
    ↓ load_data_from_bq.py
GCS: data/raw/
    ↓ clean_raw_data.py
GCS: data/clean/
    ↓ transform_data.py
GCS: data/transformed/ + artifacts/scalers/scaler.pkl
    ↓ train.py
GCS: artifacts/models/{sha}/model.pkl
    ↓ Docker → Artifact Registry
Cloud Run: FastAPI /predict endpoint
```

## Project Structure

```
iris_ml_project/
├── src/
│   ├── bq_ingestion.py        # Load Iris data into BigQuery
│   ├── load_data_from_bq.py   # Export BQ → GCS CSV
│   ├── clean_raw_data.py      # Clean raw CSV → clean CSV
│   ├── transform_data.py      # Scale + feature engineering
│   ├── train.py               # Train RandomForest, save .pkl to GCS
│   ├── inference.py           # Load model/scaler from GCS, predict()
│   └── app.py                 # FastAPI app (deployed to Cloud Run)
├── terraform/
│   ├── main.tf / variables.tf / outputs.tf
│   ├── modules/
│   │   ├── iam/               # Service accounts + IAM bindings
│   │   ├── bigquery/          # Dataset + schema (no data)
│   │   ├── gcs/               # Data & model bucket
│   │   ├── artifact_registry/ # Docker repository
│   │   └── cloud_run/         # Cloud Run service
│   └── envs/
│       ├── dev/  ← terraform.tfvars + backend.conf
│       ├── qa/
│       └── prod/
├── tests/
│   ├── test_pipeline.py       # Unit tests
│   └── integration/test_api.py # Integration tests vs live endpoint
├── .github/workflows/cicd.yml  # Full CI/CD pipeline with manual approvals
├── Dockerfile
└── requirements.txt
```

## Prerequisites

- GCP Projects: one each for dev, qa, prod
- Terraform state GCS buckets created manually (bootstrap):
  ```bash
  gsutil mb gs://your-project-dev-tf-state
  gsutil mb gs://your-project-qa-tf-state
  gsutil mb gs://your-project-prod-tf-state
  ```
- GitHub Secrets configured (see CI/CD section)

## Initial Setup

### 1. Edit tfvars

Update `project_id` in each environment:
- `terraform/envs/dev/terraform.tfvars`
- `terraform/envs/qa/terraform.tfvars`
- `terraform/envs/prod/terraform.tfvars`

Also update `bucket` in each `backend.conf`.

### 2. Bootstrap Terraform (first time)

```bash
cd terraform
terraform init -backend-config=envs/dev/backend.conf
terraform apply -var-file=envs/dev/terraform.tfvars
```

### 3. GitHub Secrets

Add these secrets in GitHub → Settings → Secrets:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_DEV` | Dev project ID |
| `GCP_PROJECT_QA` | QA project ID |
| `GCP_PROJECT_PROD` | Prod project ID |
| `GCP_SA_KEY_DEV` | CI/CD service account JSON key (dev) |
| `GCP_SA_KEY_QA` | CI/CD service account JSON key (qa) |
| `GCP_SA_KEY_PROD` | CI/CD service account JSON key (prod) |
| `GCS_BUCKET_DEV` | GCS bucket name (dev) |
| `GCS_BUCKET_QA` | GCS bucket name (qa) |
| `GCS_BUCKET_PROD` | GCS bucket name (prod) |

### 4. Enable Manual Approvals

In GitHub → Settings → Environments, create `qa` and `prod` environments
and add required reviewers. This gates the CI/CD pipeline at each stage.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars
export GCP_PROJECT=your-dev-project
export GCS_BUCKET=your-dev-bucket
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json

# Run the pipeline manually
python src/bq_ingestion.py --project_id $GCP_PROJECT --dataset_id iris_dev --table_id iris_raw
python src/load_data_from_bq.py --project_id $GCP_PROJECT --dataset_id iris_dev --table_id iris_raw --gcs_bucket $GCS_BUCKET
python src/clean_raw_data.py --project_id $GCP_PROJECT --gcs_bucket $GCS_BUCKET
python src/transform_data.py --project_id $GCP_PROJECT --gcs_bucket $GCS_BUCKET
python src/train.py --project_id $GCP_PROJECT --gcs_bucket $GCS_BUCKET

# Run the API locally
GCS_BUCKET=$GCS_BUCKET \
MODEL_PREFIX=artifacts/models/latest \
SCALER_PREFIX=artifacts/scalers \
python src/app.py
```

## API Usage

### Health check
```bash
curl https://<your-cloud-run-url>/health
```

### Single prediction
```bash
curl -X POST https://<your-cloud-run-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Response:
# {"class_id": 0, "class_name": "setosa", "probabilities": {"setosa": 0.98, ...}}
```

### Batch prediction
```bash
curl -X POST https://<your-cloud-run-url>/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
  ]}'
```

## Running Tests

```bash
# Unit tests
pytest tests/test_pipeline.py -v

# Integration tests (against live endpoint)
pytest tests/integration/ -v --base-url=https://iris-ml-api-dev-xxx.run.app
```

## Key Design Decisions

- **No model embedded in image**: Model is loaded from GCS at startup, enabling model updates without rebuilding the container.
- **Immutable image promotion**: The same Docker image SHA is promoted Dev → QA → Prod. No rebuilds.
- **Schema-first BigQuery**: Table schema defined by Terraform; data loaded by CI/CD.
- **Scaler persisted separately**: The fitted scaler is saved to GCS and loaded alongside the model to prevent training/serving skew.
- **Least privilege IAM**: Separate service accounts for CI/CD and Cloud Run with minimal permissions.
