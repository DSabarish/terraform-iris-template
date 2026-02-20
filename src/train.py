"""
train.py
--------
Trains a RandomForest classifier on the transformed Iris dataset,
evaluates it, and saves the model artifact to GCS.

The model is versioned using the GIT_COMMIT_SHA environment variable.

Usage:
    python train.py \
        --project_id <GCP_PROJECT_ID> \
        --gcs_bucket <BUCKET_NAME> \
        --transformed_prefix data/transformed \
        --model_prefix artifacts/models
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_area", "sepal_area"]
TARGET_COL = "target"
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def read_csv_from_gcs(storage_client: storage.Client, bucket: str, path: str) -> pd.DataFrame:
    blob = storage_client.bucket(bucket).blob(path)
    return pd.read_csv(io.StringIO(blob.download_as_text()))


def upload_bytes(storage_client: storage.Client, bucket: str, path: str, data: bytes) -> str:
    blob = storage_client.bucket(bucket).blob(path)
    blob.upload_from_string(data)
    return f"gs://{bucket}/{path}"


def train_and_evaluate(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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
    commit_sha = os.getenv("GIT_COMMIT_SHA", "local")
    storage_client = storage.Client(project=project_id)

    logger.info("Loading transformed data...")
    df = read_csv_from_gcs(storage_client, gcs_bucket, f"{transformed_prefix}/iris_transformed.csv")

    logger.info("Training model...")
    model, metrics = train_and_evaluate(df)

    # Save model
    model_bytes = pickle.dumps(model)
    model_path = f"{model_prefix}/{commit_sha}/model.pkl"
    model_uri = upload_bytes(storage_client, gcs_bucket, model_path, model_bytes)
    logger.info("Model saved to: %s", model_uri)

    # Also save as 'latest' for easy reference
    latest_path = f"{model_prefix}/latest/model.pkl"
    upload_bytes(storage_client, gcs_bucket, latest_path, model_bytes)

    # Save metrics
    metrics_bytes = json.dumps(metrics, indent=2).encode()
    metrics_path = f"{model_prefix}/{commit_sha}/metrics.json"
    upload_bytes(storage_client, gcs_bucket, metrics_path, metrics_bytes)
    upload_bytes(storage_client, gcs_bucket, f"{model_prefix}/latest/metrics.json", metrics_bytes)

    # Save model URI for downstream steps
    print(model_uri)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--gcs_bucket", required=True)
    parser.add_argument("--transformed_prefix", default="data/transformed")
    parser.add_argument("--model_prefix", default="artifacts/models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.project_id, args.gcs_bucket, args.transformed_prefix, args.model_prefix)
