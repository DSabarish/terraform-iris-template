"""
inference.py
------------
Loads the model and scaler from GCS or Vertex AI Model Registry.

Environment Variables (set at runtime, e.g. Cloud Run):
  Option A — Vertex AI (registered model):
    - VERTEX_MODEL_DISPLAY_NAME: use latest model with this name
    - VERTEX_REGION: region for Vertex AI (default us-central1)
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - GCS_BUCKET: bucket where scaler.pkl is stored
  Option B — GCS directly:
    - GCS_BUCKET, MODEL_PREFIX, SCALER_PREFIX
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
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _parse_gs_uri(uri: str) -> Tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    uri = uri.rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    rest = uri[5:]
    idx = rest.find("/")
    if idx == -1:
        return rest, ""
    return rest[:idx], rest[idx + 1:]


def _download_pkl(bucket_name: str, blob_path: str) -> object:
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    data = blob.download_as_bytes()
    return pickle.loads(data)


@lru_cache(maxsize=1)
def _get_vertex_artifact_uri() -> str | None:
    """Resolve Vertex AI model display name to GCS artifact_uri."""
    display_name = os.environ.get("VERTEX_MODEL_DISPLAY_NAME")
    region = os.environ.get("VERTEX_REGION", "us-central1")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")

    if display_name and project:
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=region)
        models = aiplatform.Model.list(
            filter=f'display_name="{display_name}"',
            order_by="create_time desc",
        )
        if not models:
            raise ValueError(f"No Vertex AI model found with display_name={display_name!r}")
        uri = models[0].artifact_uri
        logger.info("Vertex model %s -> artifact_uri %s", display_name, uri)
        return uri
    return None


@lru_cache(maxsize=1)
def load_model():
    """Load model.pkl — from Vertex AI artifact URI or directly from GCS."""
    vertex_uri = _get_vertex_artifact_uri()
    if vertex_uri:
        bucket, prefix = _parse_gs_uri(vertex_uri)
        model_path = f"{prefix}/model.pkl" if prefix else "model.pkl"
        logger.info("Loading model from gs://%s/%s", bucket, model_path)
        return _download_pkl(bucket, model_path)

    # Fallback: direct GCS path
    bucket = os.environ["GCS_BUCKET"]
    model_prefix = os.environ.get("MODEL_PREFIX", "artifacts/models/latest")
    model_path = f"{model_prefix}/model.pkl"
    logger.info("Loading model from gs://%s/%s", bucket, model_path)
    return _download_pkl(bucket, model_path)


@lru_cache(maxsize=1)
def load_scaler():
    """Load scaler.pkl — always from the dedicated scaler GCS path.

    The scaler is stored separately from the model at artifacts/scalers/scaler.pkl.
    We never look for it inside the Vertex AI artifact URI because it is not
    uploaded there — only model.pkl is.
    """
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

    # Engineered features
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
