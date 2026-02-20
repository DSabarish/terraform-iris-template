"""
transform_data.py
-----------------
Reads clean Iris CSV from GCS, applies scaling / feature engineering,
and saves a transformed CSV back to GCS.

Transformations:
  - StandardScaler on feature columns
  - Saves the fitted scaler as scaler.pkl to GCS (used during inference)

Usage:
    python transform_data.py \
        --project_id <GCP_PROJECT_ID> \
        --gcs_bucket <BUCKET_NAME> \
        --clean_prefix data/clean \
        --transformed_prefix data/transformed \
        --scaler_prefix artifacts/scalers
"""

import argparse
import io
import logging
import pickle

from google.cloud import storage
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "target"


def read_csv_from_gcs(storage_client: storage.Client, bucket_name: str, blob_path: str) -> pd.DataFrame:
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    return pd.read_csv(io.StringIO(blob.download_as_text()))


def upload_bytes_to_gcs(storage_client: storage.Client, bucket_name: str, blob_path: str, data: bytes) -> str:
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(data)
    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info("Uploaded to %s", uri)
    return uri


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_out = df.copy()
    df_out[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # Feature engineering: petal area approximation
    df_out["petal_area"] = df_out["petal_length"] * df_out["petal_width"]
    df_out["sepal_area"] = df_out["sepal_length"] * df_out["sepal_width"]

    logger.info("Transformed data shape: %s", df_out.shape)
    return df_out, scaler


def main(
    project_id: str,
    gcs_bucket: str,
    clean_prefix: str,
    transformed_prefix: str,
    scaler_prefix: str,
) -> None:
    storage_client = storage.Client(project=project_id)

    clean_blob_path = f"{clean_prefix}/iris_clean.csv"
    logger.info("Reading clean data from gs://%s/%s", gcs_bucket, clean_blob_path)
    df_clean = read_csv_from_gcs(storage_client, gcs_bucket, clean_blob_path)

    df_transformed, scaler = transform(df_clean)

    # Save transformed CSV
    buf = io.StringIO()
    df_transformed.to_csv(buf, index=False)
    upload_bytes_to_gcs(
        storage_client, gcs_bucket,
        f"{transformed_prefix}/iris_transformed.csv",
        buf.getvalue().encode(),
    )

    # Save scaler artifact
    scaler_bytes = pickle.dumps(scaler)
    upload_bytes_to_gcs(
        storage_client, gcs_bucket,
        f"{scaler_prefix}/scaler.pkl",
        scaler_bytes,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Transform Iris data")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--clean_prefix", default="data/clean")
    parser.add_argument("--transformed_prefix", default="data/transformed")
    parser.add_argument("--scaler_prefix", default="artifacts/scalers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.project_id,
        args.gcs_bucket,
        args.clean_prefix,
        args.transformed_prefix,
        args.scaler_prefix,
    )
