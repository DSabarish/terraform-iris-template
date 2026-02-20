# Vertex AI Model Registry + Deployment Guide

This guide explains the complete flow: **GCS → Vertex AI Model Registry → Docker → Cloud Run** with your HTML frontend.

## Architecture Flow

```
GitHub Actions Pipeline:
  1. Train model → Save to GCS (artifacts/models/<sha>/model.pkl)
  2. Register model → Vertex AI Model Registry
  3. Build Docker image → Artifact Registry
  4. Deploy to Cloud Run → Links to Vertex AI Model Registry
  5. HTML frontend → Served from Cloud Run
```

## Prerequisites

### 1. Additional GitHub Secrets

Add these to **Settings → Secrets and variables → Actions**:

- **`VERTEX_REGION`** (optional): Vertex AI region, defaults to `us-central1`
- **`CLOUD_RUN_SA_EMAIL`**: Service account email for Cloud Run deployment
  - Example: `iris-ml-github@iris-100.iam.gserviceaccount.com`
  - This SA needs:
    - `roles/run.admin` (to deploy)
    - `roles/aiplatform.user` (to access Vertex AI Model Registry)
    - `roles/storage.objectViewer` (to read model/scaler from GCS)

### 2. Artifact Registry Repository

Create a Docker repository in Artifact Registry:

```bash
gcloud artifacts repositories create iris-ml \
  --repository-format=docker \
  --location=us-central1 \
  --project=iris-100
```

### 3. Enable APIs

Ensure these APIs are enabled:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  --project=iris-100
```

## How It Works

### Step 1: Training (GitHub Actions)

The `pipeline` job runs `src/pipeline.py`, which:
- Trains the model
- Saves to `gs://<bucket>/artifacts/models/<commit-sha>/model.pkl`
- Also saves scaler to `gs://<bucket>/artifacts/scalers/scaler.pkl`

### Step 2: Model Registration (GitHub Actions)

The `register_and_deploy` job runs `scripts/register_model_vertex.py`, which:
- Registers the model in **Vertex AI Model Registry**
- Uses the GCS directory as `artifact_uri`
- Creates a model with display name like `iris-classifier-<commit-sha>`

### Step 3: Docker Build (GitHub Actions)

The `build_image` job:
- Builds Docker image from `Dockerfile`
- Tags with commit SHA and `latest`
- Pushes to Artifact Registry: `us-central1-docker.pkg.dev/<project>/iris-ml/iris-ml-api:<sha>`

### Step 4: Cloud Run Deployment (GitHub Actions)

The `register_and_deploy` job runs `scripts/deploy_cloud_run_vertex.py`, which:
- Deploys the Docker image to Cloud Run
- Sets environment variables:
  - `VERTEX_MODEL_DISPLAY_NAME`: Points to the registered model
  - `VERTEX_REGION`: Vertex AI region
  - `GOOGLE_CLOUD_PROJECT`: Project ID
- Your `src/inference.py` automatically loads from Vertex AI Model Registry
- HTML frontend (`frontend/index.html`) is served at `/`

## Manual Deployment (Optional)

If you want to deploy manually instead of via GitHub Actions:

### 1. Register Model

```bash
python scripts/register_model_vertex.py \
  --project_id iris-100 \
  --region us-central1 \
  --model_gcs_uri gs://iris-100-ml-bucket/artifacts/models/latest/model.pkl \
  --scaler_gcs_uri gs://iris-100-ml-bucket/artifacts/scalers/scaler.pkl \
  --display_name iris-classifier-latest \
  --description "Latest Iris classifier model"
```

### 2. Build Docker Image

```bash
PROJECT_ID=iris-100
REGION=us-central1
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/iris-ml/iris-ml-api:latest"

gcloud builds submit --tag $IMAGE_URI .
```

### 3. Deploy to Cloud Run

```bash
python scripts/deploy_cloud_run_vertex.py \
  --project_id iris-100 \
  --region us-central1 \
  --service_name iris-ml-api \
  --image_uri us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest \
  --model_display_name iris-classifier-latest \
  --service_account iris-ml-github@iris-100.iam.gserviceaccount.com
```

## Accessing Your Deployment

After deployment, Cloud Run provides a URL like:
```
https://iris-ml-api-xxxxx.us-central1.run.app
```

- **Frontend (HTML)**: `https://iris-ml-api-xxxxx.us-central1.run.app/`
- **API Health**: `https://iris-ml-api-xxxxx.us-central1.run.app/health`
- **API Predict**: `POST https://iris-ml-api-xxxxx.us-central1.run.app/predict`

## How the App Loads the Model

Your `src/inference.py` checks environment variables in this order:

1. **`VERTEX_MODEL_DISPLAY_NAME`** (set by Cloud Run deployment)
   - Looks up the model in Vertex AI Model Registry
   - Downloads model/scaler from the model's `artifact_uri` (GCS)

2. **`VERTEX_MODEL_RESOURCE_NAME`** (alternative)
   - Direct resource name: `projects/123/locations/us-central1/models/456`

3. **Fallback to GCS** (if no Vertex vars)
   - Uses `GCS_BUCKET`, `MODEL_PREFIX`, `SCALER_PREFIX`

## Benefits of Vertex AI Model Registry

- **Versioning**: Each commit creates a new model version
- **Lineage**: Track which commit/model version is deployed
- **Management**: View all models in Vertex AI Console
- **Rollback**: Easy to switch to previous model versions
- **Monitoring**: Vertex AI provides model monitoring capabilities

## Troubleshooting

### Model registration fails

- Ensure service account has `roles/aiplatform.admin` or `roles/aiplatform.user`
- Check that GCS paths exist and are accessible

### Cloud Run can't load model

- Verify `VERTEX_MODEL_DISPLAY_NAME` matches a registered model
- Check Cloud Run service account has `roles/aiplatform.user`
- Check Cloud Run logs: `gcloud run services logs read iris-ml-api --region=us-central1`

### Docker build fails

- Ensure Artifact Registry repository exists
- Check service account has `roles/artifactregistry.writer`
