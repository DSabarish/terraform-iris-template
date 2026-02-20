## Iris ML — Clean CI/CD with GitHub Actions

This repository contains an end-to-end Iris ML project:
**BigQuery → GCS → Clean → Transform → Train → Serve with FastAPI**,
with a simple GitHub Actions workflow as the only CI/CD entrypoint.

Terraform and Vertex custom jobs have been removed so you can push this
repo to GitHub and let Actions do the heavy lifting.

### Project Structure (simplified)

```
src/
  __init__.py
  bq_ingestion.py       # Load Iris data into BigQuery
  load_data_from_bq.py  # Export BQ → GCS CSV
  clean_raw_data.py     # Clean raw CSV → clean CSV
  transform_data.py     # Scale + feature engineering + scaler.pkl
  train.py              # Train RandomForest, save model/metrics to GCS
  inference.py          # Load model/scaler from GCS or Vertex, predict()
  app.py                # FastAPI app exposing /health and /predict
  pipeline.py           # Single “master” pipeline entrypoint

tests/
  test_core_pipeline.py # Three focused unit tests

.github/workflows/ci.yml  # Tests + GCP pipeline job (no Terraform)
Dockerfile
requirements.txt
frontend/index.html        # Optional simple UI for the API
```

### GitHub Actions CI/CD

The workflow in `.github/workflows/ci.yml` has **two jobs**:

- **test**: installs dependencies and runs `pytest -v`.
- **pipeline**: after tests pass on `main`, runs the full GCP pipeline
  by calling:

  ```bash
  python src/pipeline.py \
    --project_id "$GCP_PROJECT" \
    --dataset_id "iris_dataset" \
    --table_id "iris_raw" \
    --gcs_bucket "$GCS_BUCKET"
  ```

Configure these **repository secrets** in GitHub → Settings → Secrets and variables:

- **`GCP_PROJECT`**: GCP project ID that contains BigQuery + GCS.
- **`GCS_BUCKET`**: GCS bucket to store data, scaler, and models.
- **`GCP_SA_KEY`**: JSON contents of a service account key with
  BigQuery and Storage access (paste the raw JSON as the secret value).

> The workflow writes `GCP_SA_KEY` into `gcp-sa-key.json` at runtime and
> points `GOOGLE_APPLICATION_CREDENTIALS` to this file, so nothing
> sensitive is committed to the repo.

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

To run the **pipeline locally** (same as GitHub Actions):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
export GCP_PROJECT=your-project-id
export GCS_BUCKET=your-bucket

python src/pipeline.py \
  --project_id "$GCP_PROJECT" \
  --dataset_id "iris_dataset" \
  --table_id "iris_raw" \
  --gcs_bucket "$GCS_BUCKET"
```

To run the **API locally** after you have a trained model in GCS:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
export GCS_BUCKET=your-bucket
export MODEL_PREFIX=artifacts/models/latest
export SCALER_PREFIX=artifacts/scalers

cd src
python app.py
```

### Running Tests

```bash
pytest -v
```

The current tests cover:

- core cleaning logic (`clean_data`)
- transformation + feature engineering (`transform`)
- model training + metrics (`train_and_evaluate`)

