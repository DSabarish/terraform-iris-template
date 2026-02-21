"""
inference.py
------------
Loads the model and scaler from GCS.

Environment Variables (set at runtime via Cloud Run):
  GCS_BUCKET:            GCS bucket name
  MODEL_PREFIX:          path to model dir   (default: artifacts/models/latest)
  SCALER_PREFIX:         path to scaler dir  (default: artifacts/scalers)
  VERTEX_MODEL_DISPLAY_NAME: if set, resolve latest model artifact URI from Vertex AI
  VERTEX_REGION:         Vertex AI region    (default: us-central1)
  GOOGLE_CLOUD_PROJECT:  GCP project ID
"""

import logging
import os
import pickle
from functools import lru_cache
from typing import List, Tuple

from google.cloud import storage
import numpy as np

logger = logging.getLogger(__name__)

TARGET_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    uri = uri.rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    rest = uri[5:]
    idx = rest.find("/")
    if idx == -1:
        return rest, ""
    return rest[:idx], rest[idx + 1:]


def _download_pkl(bucket_name: str, blob_path: str) -> object:
    client = storage.Client()
    data = client.bucket(bucket_name).blob(blob_path).download_as_bytes()
    return pickle.loads(data)


@lru_cache(maxsize=1)
def _get_model_gcs_prefix() -> Tuple[str, str]:
    """
    Returns (bucket, prefix) for the model directory.
    Tries Vertex AI first, falls back to GCS env vars.
    """
    display_name = os.environ.get("VERTEX_MODEL_DISPLAY_NAME")
    region = os.environ.get("VERTEX_REGION", "us-central1")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")

    if display_name and project:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=project, location=region)
            models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                order_by="create_time desc",
            )
            if models:
                # Call gca_resource to get the full resource with artifact_uri
                model_resource = models[0].gca_resource
                artifact_uri = model_resource.artifact_uri
                if artifact_uri:
                    bucket, prefix = _parse_gs_uri(artifact_uri)
                    logger.info("Vertex model artifact URI: gs://%s/%s", bucket, prefix)
                    return bucket, prefix
        except Exception as e:
            logger.warning("Vertex AI model lookup failed, falling back to GCS: %s", e)

    # Fallback: direct GCS env vars
    bucket = os.environ["GCS_BUCKET"]
    prefix = os.environ.get("MODEL_PREFIX", "artifacts/models/latest")
    logger.info("Using GCS model path: gs://%s/%s", bucket, prefix)
    return bucket, prefix


@lru_cache(maxsize=1)
def load_model():
    bucket, prefix = _get_model_gcs_prefix()
    path = f"{prefix}/model.pkl" if prefix else "model.pkl"
    logger.info("Loading model from gs://%s/%s", bucket, path)
    return _download_pkl(bucket, path)


@lru_cache(maxsize=1)
def load_scaler():
    # Scaler is always at a dedicated GCS path — never inside the Vertex artifact dir
    bucket = os.environ["GCS_BUCKET"]
    prefix = os.environ.get("SCALER_PREFIX", "artifacts/scalers")
    path = f"{prefix}/scaler.pkl"
    logger.info("Loading scaler from gs://%s/%s", bucket, path)
    return _download_pkl(bucket, path)


def predict(features: List[float]) -> dict:
    scaler = load_scaler()
    model = load_model()

    arr = np.array(features, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

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