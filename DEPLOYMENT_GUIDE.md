# Deployment Guide — Iris ML Project (GitHub Actions-first)

This guide explains how to deploy and retrain the Iris ML pipeline
using **GitHub Actions + GCP**, with a clean, GitHub-triggered flow.

The old local-only and Terraform-based instructions have been removed /
replaced. The **source of truth** for automation is:

- Workflow: `.github/workflows/ci.yml`
- Pipeline entrypoint: `src/pipeline.py`

---

## 1. One-time GCP setup

You still need a project, dataset, and bucket once.

- Create / choose a GCP project (e.g. `iris-100`) and enable billing.
- Enable APIs (can be done once from Cloud Console, or via `gcloud`):
  - BigQuery
  - Cloud Storage

Create a bucket and dataset (names are examples; pick your own):

```bash
PROJECT_ID=iris-100
REGION=us-central1
BUCKET=iris-100-ml-bucket

gcloud config set project "$PROJECT_ID"
gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET"
bq mk --dataset --location="$REGION" "${PROJECT_ID}:iris_dataset"
```

---

## 2. Create a service account and key for GitHub

Create a service account with BigQuery + Storage access, and generate a
JSON key that will be stored **only** in GitHub Secrets.

```bash
PROJECT_ID=iris-100
SA_NAME=iris-ml-github
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Iris ML GitHub Actions"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Create a key file locally once:

```bash
gcloud iam service-accounts keys create gcp-sa-key.json \
  --iam-account="${SA_EMAIL}"
```

Open `gcp-sa-key.json` in a text editor and copy the **entire JSON**
into a GitHub Secret (see next section), then **delete the local file**
or store it securely. `.gitignore` already ignores `gcp-sa-key.json`
and `*-sa-key.json`.

---

## 3. Configure GitHub Secrets

In your GitHub repository:

1. Go to **Settings → Secrets and variables → Actions**.
2. Add these **repository secrets**:

| Secret        | Example value              | Description                          |
|---------------|---------------------------|--------------------------------------|
| `GCP_PROJECT` | `iris-100`                | GCP project ID                       |
| `GCS_BUCKET`  | `iris-100-ml-bucket`      | Bucket for data, scaler, and models |
| `GCP_SA_KEY`  | (JSON from `gcp-sa-key`)  | Service account key contents         |

You do not commit any keys; the workflow writes the JSON into
`gcp-sa-key.json` at runtime only.

---

## 4. How GitHub Actions runs the pipeline

Workflow: `.github/workflows/ci.yml`

- On every push / PR to `main`:
  - Job **`test`** installs dependencies and runs `pytest -v`.
- On pushes to `main` (after tests pass):
  - Job **`pipeline`**:
    - Writes the `GCP_SA_KEY` JSON to `gcp-sa-key.json`.
    - Sets `GOOGLE_APPLICATION_CREDENTIALS` to that file.
    - Runs:

      ```bash
      python src/pipeline.py \
        --project_id "$GCP_PROJECT" \
        --dataset_id "iris_dataset" \
        --table_id "iris_raw" \
        --gcs_bucket "$GCS_BUCKET"
      ```

`src/pipeline.py` orchestrates:

1. Load Iris data into BigQuery (`iris_dataset.iris_raw`).
2. Export to GCS (`data/raw/iris_raw.csv`).
3. Clean raw data → `data/clean/iris_clean.csv`.
4. Transform + scaler → `data/transformed/...` and `artifacts/scalers/scaler.pkl`.
5. Train model → `artifacts/models/<sha>/model.pkl` and `artifacts/models/latest/model.pkl`.

All of this now runs in GitHub’s runner, not on your local machine.

---

## 5. Optional: Run the same pipeline locally

If you want to replicate what GitHub does from your laptop:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-sa-key.json
export GCP_PROJECT=iris-100
export GCS_BUCKET=iris-100-ml-bucket

python src/pipeline.py \
  --project_id "$GCP_PROJECT" \
  --dataset_id "iris_dataset" \
  --table_id "iris_raw" \
  --gcs_bucket "$GCS_BUCKET"
```

This will produce the same GCS layout that CI does.

---

## 6. Serving the model (Cloud Run) — optional

Serving is **not** wired into GitHub Actions in this simplified setup,
but you can still deploy the existing Dockerfile + FastAPI app to
Cloud Run once the model is in GCS.

High level:

1. Build and push the image (locally with Docker or via Cloud Build).
2. Deploy a Cloud Run service pointing to that image.
3. Set env vars:
   - `GCS_BUCKET`
   - `MODEL_PREFIX=artifacts/models/latest`
   - `SCALER_PREFIX=artifacts/scalers`

The `src/app.py` and `src/inference.py` pieces already handle loading
the scaler and model from GCS.

---

## 7. Retraining later

To retrain with fresh data:

1. Push a new commit to `main` (code changes, or even an empty commit).
2. GitHub Actions runs:
   - Tests (`pytest -v`).
   - Full pipeline via `src/pipeline.py`.
3. A new model is written to:
   - `gs://<bucket>/artifacts/models/latest/model.pkl`

If you have Cloud Run pointing at the `latest` prefix, you can trigger
a re-deploy or just let the service reload the model depending on how
you’ve configured it.

---

## 8. (Optional) Run the pipeline as a Vertex AI Custom Job

If you want the training pipeline to run on managed GCP compute instead
of a GitHub runner, you can use `scripts/run_vertex_custom_job.py` to
submit a **Vertex AI Custom Job** that reuses the same Docker image as
your API (it already contains the code and dependencies).

High-level:

1. Build and push the Docker image defined in `Dockerfile` to Artifact
   Registry (similar to how you would for Cloud Run).
2. Note the image URI, for example:

   ```text
   us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest
   ```

3. Ensure the Vertex service account has BigQuery + Storage access.
4. Run, from your machine:

   ```bash
   python scripts/run_vertex_custom_job.py \
     --project_id iris-100 \
     --region us-central1 \
     --gcs_bucket iris-100-ml-bucket \
     --image_uri us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest \
     --service_account iris-ml-github@iris-100.iam.gserviceaccount.com
   ```

Vertex AI will start a container from that image and override its
command to:

```bash
python pipeline.py --project_id ... --dataset_id ... --table_id ... --gcs_bucket ...
```

producing the same GCS outputs as the GitHub Actions pipeline.

You can wire this into GitHub Actions later by calling the script from
CI instead of running `pipeline.py` directly.

---

## 9. Troubleshooting

- **Pipeline job fails with BigQuery or GCS permissions error**
  - Ensure the service account used for `GCP_SA_KEY` has:
    - `roles/bigquery.dataEditor`
    - `roles/bigquery.jobUser`
    - `roles/storage.objectAdmin`

- **`GOOGLE_APPLICATION_CREDENTIALS` issues locally**
  - Make sure the path points to a valid JSON key and that the project
    inside that key has access to the dataset and bucket you’re using.

- **Tests fail in CI but pass locally**
  - Ensure local Python version is close to CI’s (3.11), and that
    you ran `pip install -r requirements.txt` from a clean venv.
