"""
Run Iris training pipeline locally (no Vertex AI quota needed).
--------------------------------------------------------------
Runs: BQ ingest → export to GCS → clean → transform → train.
Writes model to GCS at artifacts/models/latest/model.pkl and
scaler to artifacts/scalers/scaler.pkl.

Use this when you don't have Vertex AI custom training quota.
Then deploy Cloud Run with GCS only (see VERTEX_AI_SETUP.md "Without Vertex training quota").

Requires: Python 3.10+, gcloud auth (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth application-default login),
          PROJECT_ID and GCS_BUCKET with BQ + GCS access.

Usage (PowerShell):
  $env:PROJECT_ID = "iris-100"
  $env:GCS_BUCKET = "iris-bucket-100"
  python scripts/run_training_local.py

Usage (Bash):
  export PROJECT_ID=iris-100 GCS_BUCKET=iris-bucket-100
  python scripts/run_training_local.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(PROJECT_ROOT)


def run(cmd: list, env: dict | None = None) -> None:
    env = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    project_id = os.environ.get("PROJECT_ID")
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not project_id or not gcs_bucket:
        print("Set PROJECT_ID and GCS_BUCKET (and optionally DATASET_ID).", file=sys.stderr)
        sys.exit(1)
    dataset_id = os.environ.get("DATASET_ID", "iris_dataset")

    print("Running Iris pipeline locally (no Vertex quota).")
    print(f"  PROJECT_ID={project_id} GCS_BUCKET={gcs_bucket} DATASET_ID={dataset_id}")

    # 1) Ingest Iris into BigQuery
    run([
        sys.executable, str(SRC / "bq_ingestion.py"),
        "--project_id", project_id,
        "--dataset_id", dataset_id,
        "--table_id", "iris_raw",
    ])

    # 2) Export BQ → GCS raw
    run([
        sys.executable, str(SRC / "load_data_from_bq.py"),
        "--project_id", project_id,
        "--dataset_id", dataset_id,
        "--table_id", "iris_raw",
        "--gcs_bucket", gcs_bucket,
    ])

    # 3) Clean
    run([
        sys.executable, str(SRC / "clean_raw_data.py"),
        "--project_id", project_id,
        "--gcs_bucket", gcs_bucket,
    ])

    # 4) Transform (writes scaler to artifacts/scalers/scaler.pkl)
    run([
        sys.executable, str(SRC / "transform_data.py"),
        "--project_id", project_id,
        "--gcs_bucket", gcs_bucket,
    ])

    # 5) Train (writes to artifacts/models/<sha>/ and artifacts/models/latest/model.pkl)
    run([
        sys.executable, str(SRC / "train.py"),
        "--project_id", project_id,
        "--gcs_bucket", gcs_bucket,
        "--model_prefix", "artifacts/models",
    ], env={"GIT_COMMIT_SHA": "local"})

    print("Done. Model: gs://{}/artifacts/models/latest/model.pkl".format(gcs_bucket))
    print("Scaler:   gs://{}/artifacts/scalers/scaler.pkl".format(gcs_bucket))
    print("Deploy Cloud Run with GCS only (no VERTEX_* vars); see VERTEX_AI_SETUP.md.")


if __name__ == "__main__":
    main()
