"""
pipeline.py
-----------
Single entrypoint to run the full Iris pipeline on GCP:

BigQuery → GCS (raw) → GCS (clean) → GCS (transformed + scaler) → GCS (model).

This is what GitHub Actions should call instead of running each step manually.
"""

import argparse
import logging

from src import bq_ingestion, load_data_from_bq, clean_raw_data, transform_data, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    project_id: str,
    dataset_id: str = "iris_dataset",
    table_id: str = "iris_raw",
    gcs_bucket: str = "",
) -> None:
    if not gcs_bucket:
        raise ValueError("gcs_bucket must be provided")

    logger.info("Starting Iris pipeline for project=%s, bucket=%s", project_id, gcs_bucket)

    # 1) Load Iris into BigQuery
    bq_ingestion.load_iris_to_bq(project_id=project_id, dataset_id=dataset_id, table_id=table_id)

    # 2) Export BQ → GCS (raw CSV)
    load_data_from_bq.export_bq_to_gcs(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        gcs_bucket=gcs_bucket,
        gcs_prefix="data/raw",
    )

    # 3) Clean raw data
    clean_raw_data.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        raw_prefix="data/raw",
        clean_prefix="data/clean",
    )

    # 4) Transform data + save scaler
    transform_data.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        clean_prefix="data/clean",
        transformed_prefix="data/transformed",
        scaler_prefix="artifacts/scalers",
    )

    # 5) Train model + save artifacts
    train.main(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        transformed_prefix="data/transformed",
        model_prefix="artifacts/models",
    )

    logger.info("Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Iris ML pipeline on GCP")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--dataset_id", default="iris_dataset", help="BigQuery dataset ID")
    parser.add_argument("--table_id", default="iris_raw", help="BigQuery table ID")
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

