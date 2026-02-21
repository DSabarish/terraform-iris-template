# Iris ML — Project Setup & GitHub Actions Architecture

## Overview

This project implements a full end-to-end ML pipeline for Iris flower classification,
deployed on Google Cloud Platform. The entire workflow — from data ingestion to model
deployment — is automated via GitHub Actions.

---

## Project Structure

```
terraform-iris-template/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions pipeline
├── src/
│   ├── pipeline.py                 # Entrypoint: runs all pipeline steps
│   ├── bq_ingestion.py             # Loads Iris data into BigQuery
│   ├── load_data_from_bq.py        # Exports BQ table to GCS as CSV
│   ├── clean_raw_data.py           # Cleans raw CSV
│   ├── transform_data.py           # Feature engineering + scaling
│   ├── train.py                    # Model training + saves artifacts to GCS
│   ├── app.py                      # FastAPI app served on Cloud Run
│   └── inference.py                # Model + scaler loading and prediction logic
├── scripts/
│   ├── register_model_vertex.py    # Registers model in Vertex AI Model Registry
│   └── deploy_cloud_run_vertex.py  # Deploys FastAPI app to Cloud Run
├── tests/
│   ├── test_core_pipeline.py
│   └── test_pipeline.py
├── frontend/
│   └── index.html                  # Static frontend served by FastAPI
├── Dockerfile                      # Container definition for Cloud Run
└── requirements.txt
```

---

## GCP Services Used

| Service | Purpose |
|---|---|
| **BigQuery** | Stores raw Iris dataset |
| **Google Cloud Storage (GCS)** | Stores raw CSV, cleaned data, transformed data, model artifacts, scaler |
| **Artifact Registry** | Stores Docker images |
| **Vertex AI Model Registry** | Registers and versions trained models |
| **Cloud Run** | Hosts the FastAPI inference API |

---

## GitHub Actions Pipeline

The pipeline is defined in `.github/workflows/ci.yml` and consists of **4 jobs**
that run sequentially on every push to `dev-branch-iris`.

```
push to dev-branch-iris
        │
        ▼
   ┌─────────┐
   │  test   │  ← runs on all pushes including PRs
   └────┬────┘
        │ needs: test
        ▼
   ┌──────────┐
   │ pipeline │  ← only on dev-branch-iris
   └────┬─────┘
        │ needs: pipeline
        ▼
   ┌─────────────┐
   │ build_image │
   └──────┬──────┘
          │ needs: pipeline + build_image
          ▼
   ┌────────────────────┐
   │ register_and_deploy│
   └────────────────────┘
```

---

## Job-by-Job Breakdown

### Job 1: `test`
**Runs on:** every push and pull request  
**Purpose:** runs unit tests before any GCP resources are touched

```yaml
- pytest tests/test_core_pipeline.py tests/test_pipeline.py -v
```

**Decision:** tests run unconditionally (no `if:` guard) so PRs are also validated
before merging.

---

### Job 2: `pipeline`
**Runs on:** push to `dev-branch-iris` only  
**Purpose:** runs the full ML training pipeline on GCP

Steps in order:
1. Writes `GCP_SA_KEY` secret to `gcp-sa-key.json` for authentication
2. Runs `src/pipeline.py` which chains:
   - `bq_ingestion.py` → loads 150 Iris rows into BigQuery (`iris_dataset.iris_raw`)
   - `load_data_from_bq.py` → exports BQ table to `gs://BUCKET/data/raw/iris_raw.csv`
   - `clean_raw_data.py` → cleans and writes to `gs://BUCKET/data/clean/`
   - `transform_data.py` → scales features, writes to `gs://BUCKET/data/transformed/` and saves `scaler.pkl` to `gs://BUCKET/artifacts/scalers/`
   - `train.py` → trains RandomForest, saves `model.pkl` to `gs://BUCKET/artifacts/models/<git-sha>/`

**Key decision — GCS bucket name normalization:**  
The `GCS_BUCKET` secret was found to sometimes contain `gs://` prefix or trailing
slashes. Both `pipeline.py` and `load_data_from_bq.py` normalize the bucket name
by stripping `gs://` and splitting on `/` before passing to the GCS client.

---

### Job 3: `build_image`
**Runs on:** push to `dev-branch-iris`, after `pipeline` succeeds  
**Purpose:** builds the Docker image and pushes to Artifact Registry

```
us-central1-docker.pkg.dev/PROJECT/iris-ml/iris-ml-api:SHORT_SHA
us-central1-docker.pkg.dev/PROJECT/iris-ml/iris-ml-api:latest
```

**Key decision — REGION hardcoded, not from secret:**  
`REGION` is set directly in the job `env` block as `us-central1` rather than
from a `VERTEX_REGION` secret. Using a secret caused GitHub Actions to redact
the value (`***`) wherever it appeared in strings, breaking variable interpolation
in shell scripts.

**Key decision — base64 encoding for cross-job output:**  
The image URI contains the GCP project ID, which matches the `GCP_PROJECT` secret.
GitHub Actions' secret scanner silently drops any job output that contains a secret
value, logging `Skip output since it may contain secret`. To work around this, the
image URI is base64-encoded before writing to `$GITHUB_OUTPUT`:

```bash
IMAGE_URI_B64=$(echo -n "$IMAGE_URI" | base64 -w 0)
echo "image_uri_b64=$IMAGE_URI_B64" >> $GITHUB_OUTPUT
```

And decoded in the next job:
```bash
IMAGE_URI=$(echo -n "$IMAGE_URI_B64" | base64 -d)
echo "IMAGE_URI=$IMAGE_URI" >> $GITHUB_ENV
```

**Key decision — `$GITHUB_OUTPUT` not `$GITHUB_ENV` for cross-job values:**  
`$GITHUB_ENV` only persists within the same job. To pass values between jobs,
`$GITHUB_OUTPUT` must be used together with a job-level `outputs:` block.
Values from `needs.*.outputs` must then be injected via a step-level `env:`
block — using them directly inside `run:` scripts via `${{ }}` syntax causes
a workflow parse error.

---

### Job 4: `register_and_deploy`
**Runs on:** push to `dev-branch-iris`, after `pipeline` and `build_image` succeed  
**Purpose:** registers the trained model in Vertex AI and deploys the API to Cloud Run

#### Step: Decode IMAGE_URI
Receives the base64-encoded image URI from `build_image` via step-level env,
decodes it, and writes to `$GITHUB_ENV` for subsequent steps.

#### Step: Register model in Vertex AI Model Registry
Calls `scripts/register_model_vertex.py` which:
- Calls `aiplatform.Model.upload()` with the GCS artifact directory and the
  Docker image URI as the serving container
- Returns the Vertex AI model resource name

**Key decision — logging to stderr only:**  
All `logging.basicConfig()` calls in scripts use `stream=sys.stderr` so that
only the final `print(resource_name)` goes to stdout. This ensures shell
command substitution `MODEL_RESOURCE=$(python ...)` captures a clean single-line
value. Previously, INFO log lines were going to stdout and polluting
`MODEL_RESOURCE`, causing `$GITHUB_ENV` writes to fail with
`Invalid format` errors.

**Key decision — `tail -1` as safety net:**  
Even with stderr logging, the model resource is extracted with:
```bash
MODEL_RESOURCE=$(python scripts/register_model_vertex.py ... 2>&1 1>/tmp/out.txt)
MODEL_RESOURCE=$(cat /tmp/out.txt | tail -1)
```
This guarantees only the last line (the resource name) is used regardless of
any other output.

#### Step: Deploy to Cloud Run
Calls `scripts/deploy_cloud_run_vertex.py` which runs:
```bash
gcloud run deploy iris-ml-api \
  --image IMAGE_URI \
  --set-env-vars VERTEX_MODEL_DISPLAY_NAME=...,GCS_BUCKET=...,SCALER_PREFIX=...
```

**Key decision — pass `GCS_BUCKET` to Cloud Run:**  
The scaler (`scaler.pkl`) is stored separately from the model at
`artifacts/scalers/scaler.pkl`. The Vertex AI artifact URI only contains
`model.pkl`. The Cloud Run container needs `GCS_BUCKET` as an env var so
`inference.py` can load the scaler directly from GCS rather than trying to
find it in the Vertex artifact directory (which would crash on startup).

---

## Secrets Required

Set these in your repo under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `GCP_PROJECT` | GCP project ID (e.g. `iris-100`) |
| `GCP_SA_KEY` | Full JSON content of the GitHub Actions service account key |
| `GCS_BUCKET` | GCS bucket name (bare name, no `gs://`) |
| `CLOUD_RUN_SA_EMAIL` | Email of the Cloud Run runtime service account |

---

## IAM Permissions Required

### GitHub Actions Service Account (`GCP_SA_KEY`)
Must have these roles in the GCP project:

| Role | Purpose |
|---|---|
| `roles/bigquery.dataEditor` | Create datasets and load data |
| `roles/bigquery.jobUser` | Run BQ queries |
| `roles/storage.objectAdmin` | Read/write GCS objects |
| `roles/artifactregistry.writer` | Push Docker images |
| `roles/aiplatform.user` | Register models in Vertex AI |
| `roles/run.admin` | Deploy Cloud Run services |
| `roles/iam.serviceAccountUser` | Act as the Cloud Run service account |

### Cloud Run Service Account (`CLOUD_RUN_SA_EMAIL`)
Must have:

| Role | Purpose |
|---|---|
| `roles/storage.objectViewer` | Download model and scaler from GCS |
| `roles/aiplatform.user` | Query Vertex AI Model Registry at runtime |

---

## Inference Architecture

At runtime, the Cloud Run container:

1. On startup (`lifespan`), calls `load_model()` and `load_scaler()`
2. `load_model()` — tries Vertex AI first (via `VERTEX_MODEL_DISPLAY_NAME`),
   falls back to direct GCS path (`GCS_BUCKET/MODEL_PREFIX/model.pkl`)
3. `load_scaler()` — always loads directly from `GCS_BUCKET/SCALER_PREFIX/scaler.pkl`
4. `/predict` — scales input features, appends engineered features
   (petal area, sepal area), runs RandomForest prediction, returns class + probabilities

**Key decision — `gca_resource.artifact_uri` not `model.artifact_uri`:**  
`aiplatform.Model.list()` returns lightweight proxy objects. The `artifact_uri`
attribute is only populated on the underlying proto object accessed via
`.gca_resource.artifact_uri`. Accessing it directly on the SDK wrapper raises
`AttributeError: 'Model' object has no attribute 'artifact_uri'`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `frontend/index.html` |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/model-info` | Returns model config from env vars |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |

---

## Viewing the Deployed App

```bash
# Get the Cloud Run URL
gcloud run services describe iris-ml-api \
  --region us-central1 \
  --project iris-100 \
  --format "value(status.url)"
```

Then open in browser:
- `https://YOUR-URL/` — Frontend UI
- `https://YOUR-URL/docs` — Swagger API docs
- `https://YOUR-URL/health` — Health check
