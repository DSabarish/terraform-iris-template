"""
inference.py
------------
Loads the model and scaler from GCS or Vertex AI. Uses cfg/base.yaml (env overrides).
"""

import logging
import os
import pickle
from functools import lru_cache
from typing import List, Tuple

from google.cloud import storage
import numpy as np

from cfg import get_config

logger = logging.getLogger(__name__)


def _target_names_map() -> dict:
    c = get_config()
    return {int(k): v for k, v in c["model"]["target_names_map"].items()}


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
    Tries Vertex AI first, falls back to GCS from config (env overrides).
    """
    c = get_config()
    display_name = os.environ.get("VERTEX_MODEL_DISPLAY_NAME")
    region = os.environ.get("VERTEX_REGION") or c["gcp"].get("region", "us-central1")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID") or c["gcp"].get("project_id")

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

    # Fallback: GCS from config (env vars already merged in get_config)
    c = get_config()
    bucket = c["gcs"].get("bucket") or os.environ.get("GCS_BUCKET", "")
    prefix = c["gcs"]["paths"].get("model_latest", "artifacts/models/latest")
    if not bucket:
        raise ValueError("GCS_BUCKET or gcs.bucket in config must be set for model loading")
    logger.info("Using GCS model path: gs://%s/%s", bucket, prefix)
    return bucket, prefix


@lru_cache(maxsize=1)
def load_model():
    c = get_config()
    model_pkl = c["gcs"]["filenames"]["model_pkl"]
    bucket, prefix = _get_model_gcs_prefix()
    path = f"{prefix}/{model_pkl}" if prefix else model_pkl
    logger.info("Loading model from gs://%s/%s", bucket, path)
    return _download_pkl(bucket, path)


@lru_cache(maxsize=1)
def load_scaler():
    c = get_config()
    bucket = c["gcs"].get("bucket") or os.environ.get("GCS_BUCKET", "")
    prefix = c["gcs"]["paths"].get("scaler", "artifacts/scalers")
    if not bucket:
        raise ValueError("GCS_BUCKET or gcs.bucket in config must be set for scaler loading")
    scaler_pkl = c["gcs"]["filenames"]["scaler_pkl"]
    path = f"{prefix}/{scaler_pkl}"
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

    names = _target_names_map()
    return {
        "class_id": class_id,
        "class_name": names[class_id],
        "probabilities": {names[i]: round(p, 4) for i, p in enumerate(proba)},
    }