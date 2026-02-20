# Run steps — GitHub Actions flow (short version)

This file is now a **quick cheat sheet** for running the project with
the new GitHub Actions setup. For full details, see `README.md` and
`DEPLOYMENT_GUIDE.md`.

---

## Once per project (GCP)

- Create / pick a project (e.g. `iris-100`) and enable billing.
- Enable BigQuery + Storage APIs.
- Create:
  - BigQuery dataset `iris_dataset` in your region.
  - GCS bucket (e.g. `iris-100-ml-bucket`).
- Create a service account with:
  - `roles/bigquery.dataEditor`
  - `roles/bigquery.jobUser`
  - `roles/storage.objectAdmin`
- Generate a JSON key for that SA and copy its contents.

---

## Once per repo (GitHub)

In GitHub → **Settings → Secrets and variables → Actions**:

- Add `GCP_PROJECT` → your project ID (e.g. `iris-100`).
- Add `GCS_BUCKET` → your bucket (e.g. `iris-100-ml-bucket`).
- Add `GCP_SA_KEY` → paste the full JSON from the SA key.

Push this repo to GitHub (main branch name must match the workflow).

---

## On each push to `main`

GitHub Actions will:

1. **Run tests**

   ```bash
   pytest -v
   ```

2. **Run the full pipeline** (after tests pass):

   ```bash
   python src/pipeline.py \
     --project_id "$GCP_PROJECT" \
     --dataset_id "iris_dataset" \
     --table_id "iris_raw" \
     --gcs_bucket "$GCS_BUCKET"
   ```

Outputs go to your GCS bucket:

- `artifacts/scalers/scaler.pkl`
- `artifacts/models/latest/model.pkl`

---

## Optional: Local dev quick commands

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest -v
```

Run the pipeline locally (same as CI):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-sa-key.json
export GCP_PROJECT=iris-100
export GCS_BUCKET=iris-100-ml-bucket

python src/pipeline.py \
  --project_id "$GCP_PROJECT" \
  --dataset_id "iris_dataset" \
  --table_id "iris_raw" \
  --gcs_bucket "$GCS_BUCKET"
```
