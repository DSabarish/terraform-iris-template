# Configuration (cfg)

Single source of truth for the Iris ML project.

## Main file

- **`base.yaml`** — All defaults: app, GCP, BigQuery, GCS paths, model schema, Vertex AI, Cloud Run, logging.

## Usage in code

```python
from cfg import get_config

config = get_config()
project_id = config["gcp"]["project_id"]
bucket = config["gcs"]["bucket"]
paths = config["gcs"]["paths"]  # raw, clean, transformed, scaler, model
feature_cols = config["model"]["feature_cols"]
```

## Environment overrides

These env vars override `base.yaml` when set (e.g. in CI or Cloud Run):

- `GCP_PROJECT` / `GOOGLE_CLOUD_PROJECT` → `gcp.project_id`
- `GCS_BUCKET` → `gcs.bucket`
- `VERTEX_REGION` / `REGION` → `gcp.region`
- `MODEL_PREFIX` → `gcs.paths.model_latest`
- `SCALER_PREFIX` → `gcs.paths.scaler`
- `GIT_COMMIT_SHA` → `pipeline.version_key`
- `PORT` → `app.port`
- `ENVIRONMENT` → `app.environment`

So you can keep secrets in GitHub / Cloud Run and still use one config file.
