"""
pipeline.py — Single entrypoint for full Iris pipeline on GCP. Uses cfg/base.yaml.
BigQuery → GCS (raw) → GCS (clean) → GCS (transformed + scaler) → GCS (model).
"""

import argparse
import logging

from cfg import get_config
from src import bq_ingestion, load_data_from_bq, clean_raw_data, transform_data, train

_config = get_config()
logging.basicConfig(level=logging.INFO, format=_config["logging"]["format"])
logger = logging.getLogger(__name__)


def _normalize_bucket_name(bucket: str) -> str:
    """Normalize GCS bucket name: remove gs:// prefix, strip whitespace."""
    bucket = bucket.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    if "/" in bucket:
        bucket = bucket.split("/")[0]
    if not bucket:
        raise ValueError("GCS bucket name cannot be empty")
    return bucket


def run_pipeline(
    project_id: str,
    dataset_id: str = "iris_dataset",
    table_id: str = "iris_raw",
    gcs_bucket: str = "",
) -> None:
    if not gcs_bucket:
        raise ValueError("gcs_bucket must be provided")
    
    # Normalize bucket name
    gcs_bucket = _normalize_bucket_name(gcs_bucket)

    logger.info("Starting Iris pipeline for project=%s, bucket=%s", project_id, gcs_bucket)

    # 1) Load Iris into BigQuery
    bq_ingestion.load_iris_to_bq(project_id=project_id, dataset_id=dataset_id, table_id=table_id)

    p = _config["gcs"]["paths"]
    load_data_from_bq.export_bq_to_gcs(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        gcs_bucket=gcs_bucket,
        gcs_prefix=p["raw"],
    )
    clean_raw_data.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        raw_prefix=p["raw"],
        clean_prefix=p["clean"],
    )
    transform_data.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        clean_prefix=p["clean"],
        transformed_prefix=p["transformed"],
        scaler_prefix=p["scaler"],
    )
    train.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        transformed_prefix=p["transformed"],
        model_prefix=p["model"],
    )

    logger.info("Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    bq = _config["bigquery"]
    parser = argparse.ArgumentParser(description="Run the full Iris ML pipeline on GCP")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--dataset_id", default=bq["dataset_id"], help="BigQuery dataset ID")
    parser.add_argument("--table_id", default=bq["table_id"], help="BigQuery table ID")
    parser.add_argument("--gcs_bucket", required=True, help="GCS bucket for data & models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        gcs_bucket=args.gcs_bucket,
    )

