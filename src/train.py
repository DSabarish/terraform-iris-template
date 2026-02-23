"""
train.py
--------
Trains a RandomForest classifier on the transformed Iris dataset,
saves model and metrics to GCS. Uses cfg/base.yaml (GIT_COMMIT_SHA for versioning).
"""

import argparse
import io
import json
import logging
import os
import pickle

from google.cloud import storage
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from cfg import get_config

_config = get_config()
logging.basicConfig(level=logging.INFO, format=_config["logging"]["format"])
logger = logging.getLogger(__name__)

FEATURE_COLS = _config["model"]["feature_cols_full"]
TARGET_COL = _config["model"]["target_col"]
TARGET_NAMES = _config["model"]["target_names"]


def read_csv_from_gcs(storage_client: storage.Client, bucket: str, path: str) -> pd.DataFrame:
    blob = storage_client.bucket(bucket).blob(path)
    return pd.read_csv(io.StringIO(blob.download_as_text()))


def upload_bytes(storage_client: storage.Client, bucket: str, path: str, data: bytes) -> str:
    blob = storage_client.bucket(bucket).blob(path)
    blob.upload_from_string(data)
    return f"gs://{bucket}/{path}"


def train_and_evaluate(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    t = _config["model"]["train"]
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t["test_size"], random_state=t["random_state"], stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=t["n_estimators"], random_state=t["random_state"], n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    logger.info("Model accuracy: %.4f", accuracy)
    return model, metrics


def main(project_id: str, gcs_bucket: str, transformed_prefix: str, model_prefix: str) -> None:
    commit_sha = _config["pipeline"]["version_key"]
    storage_client = storage.Client(project=project_id)
    gcs_f = _config["gcs"]["filenames"]

    logger.info("Loading transformed data...")
    df = read_csv_from_gcs(storage_client, gcs_bucket, f"{transformed_prefix}/{gcs_f['transformed_csv']}")

    logger.info("Training model...")
    model, metrics = train_and_evaluate(df)

    model_bytes = pickle.dumps(model)
    model_path = f"{model_prefix}/{commit_sha}/{gcs_f['model_pkl']}"
    model_uri = upload_bytes(storage_client, gcs_bucket, model_path, model_bytes)
    logger.info("Model saved to: %s", model_uri)

    latest_path = f"{model_prefix}/latest/{gcs_f['model_pkl']}"
    upload_bytes(storage_client, gcs_bucket, latest_path, model_bytes)

    metrics_bytes = json.dumps(metrics, indent=2).encode()
    metrics_path = f"{model_prefix}/{commit_sha}/{gcs_f['metrics_json']}"
    upload_bytes(storage_client, gcs_bucket, metrics_path, metrics_bytes)
    upload_bytes(storage_client, gcs_bucket, f"{model_prefix}/latest/{gcs_f['metrics_json']}", metrics_bytes)

    # Save model URI for downstream steps
    print(model_uri)


def parse_args():
    p = _config["gcs"]["paths"]
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--transformed_prefix", default=p["transformed"])
    parser.add_argument("--model_prefix", default=p["model"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.project_id, args.gcs_bucket, args.transformed_prefix, args.model_prefix)
