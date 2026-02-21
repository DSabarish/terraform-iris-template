"""
src/load_data_from_bq.py

load_data_from_bq.py
--------------------
Exports raw Iris data from BigQuery to GCS as a CSV file.

Usage:
    python load_data_from_bq.py \
        --project_id <GCP_PROJECT_ID> \
        --dataset_id iris_dataset \
        --table_id iris_raw \
        --gcs_bucket <BUCKET_NAME> \
        --gcs_prefix data/raw
"""

import argparse
import io
import logging
import re

from google.cloud import bigquery, storage
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# GCS bucket names: 3-63 chars, lowercase letters, numbers, hyphens, dots only
_VALID_BUCKET_RE = re.compile(r'^[a-z0-9][a-z0-9\-_.]{1,61}[a-z0-9]$')


def _normalize_bucket_name(bucket: str) -> str:
    """Normalize and validate a GCS bucket name."""
    bucket = bucket.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    # Strip any accidental path components (e.g. "my-bucket/some/path")
    bucket = bucket.split("/")[0]
    if not bucket:
        raise ValueError("GCS bucket name cannot be empty after normalization")
    if not _VALID_BUCKET_RE.match(bucket):
        raise ValueError(
            f"Invalid GCS bucket name after normalization: '{bucket}'. "
            "Bucket names must be 3-63 chars, lowercase, and contain only "
            "letters, numbers, hyphens, and dots."
        )
    return bucket


def _normalize_prefix(prefix: str) -> str:
    """Strip leading/trailing slashes from a GCS object prefix."""
    return prefix.strip().strip("/")


def export_bq_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    gcs_bucket: str,
    gcs_prefix: str,
) -> str:
    """
    Reads the raw iris table from BigQuery and writes it to GCS as a CSV.
    Returns the GCS URI of the exported file.
    """
    gcs_bucket = _normalize_bucket_name(gcs_bucket)
    gcs_prefix = _normalize_prefix(gcs_prefix)

    # Log the resolved bucket name so misconfigured secrets are immediately visible
    logger.info("Resolved GCS bucket: '%s'", gcs_bucket)

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    logger.info("Reading data from BigQuery table: %s", table_ref)
    query = f"SELECT * FROM `{table_ref}`"
    df: pd.DataFrame = client.query(query).to_dataframe()
    logger.info("Retrieved %d rows from BigQuery", len(df))

    gcs_path = f"{gcs_prefix}/iris_raw.csv"
    gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

    logger.info("Exported raw data to: %s", gcs_uri)
    return gcs_uri


def parse_args():
    parser = argparse.ArgumentParser(description="Export BQ table to GCS CSV")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", default="iris_dataset")
    parser.add_argument("--table_id", default="iris_raw")
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--gcs_prefix", default="data/raw")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_bq_to_gcs(
        args.project_id,
        args.dataset_id,
        args.table_id,
        args.gcs_bucket,
        args.gcs_prefix,
    )