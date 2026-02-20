"""
inference.py
------------
Loads the model and scaler from GCS and provides a predict() function
used by the FastAPI app.

Environment Variables (set at runtime via Cloud Run):
  - GCS_BUCKET: GCS bucket name
  - MODEL_PREFIX: path prefix in bucket (e.g. artifacts/models/latest)
  - SCALER_PREFIX: path prefix in bucket (e.g. artifacts/scalers)
  - GCP_PROJECT_ID: GCP project

This module is imported by app.py.
"""

import io
import logging
import os
import pickle
from functools import lru_cache
from typing import List

from google.cloud import storage
import numpy as np

logger = logging.getLogger(__name__)

TARGET_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _download_pkl(bucket_name: str, blob_path: str) -> object:
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    data = blob.download_as_bytes()
    return pickle.loads(data)


@lru_cache(maxsize=1)
def load_model():
    bucket = os.environ["GCS_BUCKET"]
    model_prefix = os.environ.get("MODEL_PREFIX", "artifacts/models/latest")
    model_path = f"{model_prefix}/model.pkl"
    logger.info("Loading model from gs://%s/%s", bucket, model_path)
    return _download_pkl(bucket, model_path)


@lru_cache(maxsize=1)
def load_scaler():
    bucket = os.environ["GCS_BUCKET"]
    scaler_prefix = os.environ.get("SCALER_PREFIX", "artifacts/scalers")
    scaler_path = f"{scaler_prefix}/scaler.pkl"
    logger.info("Loading scaler from gs://%s/%s", bucket, scaler_path)
    return _download_pkl(bucket, scaler_path)


def predict(features: List[float]) -> dict:
    """
    features: [sepal_length, sepal_width, petal_length, petal_width]
    Returns: {"class_id": int, "class_name": str, "probabilities": dict}
    """
    scaler = load_scaler()
    model = load_model()

    arr = np.array(features, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    # Append engineered features
    petal_area = arr_scaled[0][2] * arr_scaled[0][3]
    sepal_area = arr_scaled[0][0] * arr_scaled[0][1]
    arr_full = np.append(arr_scaled, [[petal_area, sepal_area]], axis=1)

    class_id = int(model.predict(arr_full)[0])
    proba = model.predict_proba(arr_full)[0].tolist()

    return {
        "class_id": class_id,
        "class_name": TARGET_NAMES[class_id],
        "probabilities": {TARGET_NAMES[i]: round(p, 4) for i, p in enumerate(proba)},
    }
