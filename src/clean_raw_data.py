"""
clean_raw_data.py
-----------------
Reads raw iris CSV from GCS, cleans it, and saves clean_data.csv back to GCS.
Uses cfg/base.yaml for schema and path defaults.
"""

import argparse
import io
import logging

from google.cloud import storage
import pandas as pd

from cfg import get_config

_config = get_config()
logging.basicConfig(level=logging.INFO, format=_config["logging"]["format"])
logger = logging.getLogger(__name__)

FEATURE_COLS = _config["model"]["feature_cols"]
TARGET_COL = _config["model"]["target_col"]
VALID_TARGETS = set(_config["model"]["valid_targets"])


def read_csv_from_gcs(storage_client: storage.Client, bucket_name: str, blob_path: str) -> pd.DataFrame:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    content = blob.download_as_text()
    return pd.read_csv(io.StringIO(content))


def write_csv_to_gcs(storage_client: storage.Client, bucket_name: str, blob_path: str, df: pd.DataFrame) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")
    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info("Written %d rows to %s", len(df), uri)
    return uri


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    logger.info("After dedup: %d rows (dropped %d)", len(df), original_len - len(df))

    # Drop rows with nulls in feature/target columns
    required_cols = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=required_cols)
    logger.info("After dropna: %d rows", len(df))

    # Cast feature columns to float
    for col in FEATURE_COLS:
        df[col] = df[col].astype(float)

    # Cast target to int
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Validate target values
    invalid_targets = df[~df[TARGET_COL].isin(VALID_TARGETS)]
    if not invalid_targets.empty:
        logger.warning("Dropping %d rows with invalid target values", len(invalid_targets))
        df = df[df[TARGET_COL].isin(VALID_TARGETS)]

    logger.info("Clean dataset shape: %s", df.shape)
    return df.reset_index(drop=True)


def main(project_id: str, gcs_bucket: str, raw_prefix: str, clean_prefix: str) -> None:
    storage_client = storage.Client(project=project_id)
    raw_csv = _config["gcs"]["filenames"]["raw_csv"]
    clean_csv = _config["gcs"]["filenames"]["clean_csv"]

    raw_blob_path = f"{raw_prefix}/{raw_csv}"
    logger.info("Reading raw data from gs://%s/%s", gcs_bucket, raw_blob_path)
    df_raw = read_csv_from_gcs(storage_client, gcs_bucket, raw_blob_path)
    logger.info("Raw data shape: %s", df_raw.shape)

    df_clean = clean_data(df_raw)

    clean_blob_path = f"{clean_prefix}/{clean_csv}"
    write_csv_to_gcs(storage_client, gcs_bucket, clean_blob_path, df_clean)


def parse_args():
    p = _config["gcs"]["paths"]
    parser = argparse.ArgumentParser(description="Clean raw Iris data")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--raw_prefix", default=p["raw"])
    parser.add_argument("--clean_prefix", default=p["clean"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.project_id, args.gcs_bucket, args.raw_prefix, args.clean_prefix)
