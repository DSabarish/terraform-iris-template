"""
Register trained model from GCS to Vertex AI Model Registry.

After training saves model.pkl to GCS, this script registers it in Vertex AI
Model Registry so it can be deployed to Vertex AI Endpoints.

Usage:
    python scripts/register_model_vertex.py \
        --project_id iris-100 \
        --region us-central1 \
        --model_gcs_uri gs://bucket/artifacts/models/latest/model.pkl \
        --scaler_gcs_uri gs://bucket/artifacts/scalers/scaler.pkl \
        --display_name iris-classifier \
        --description "Iris flower classification model"
"""

import argparse
import logging
from datetime import datetime

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def register_model(
    project_id: str,
    region: str,
    model_gcs_uri: str,
    scaler_gcs_uri: str,
    display_name: str,
    description: str = "",
) -> str:
    """
    Register a model in Vertex AI Model Registry.

    Args:
        project_id: GCP project ID
        region: Vertex AI region (e.g., us-central1)
        model_gcs_uri: GCS URI to model.pkl
        scaler_gcs_uri: GCS URI to scaler.pkl
        display_name: Display name for the model in Vertex AI
        description: Optional description

    Returns:
        Full resource name of the registered model
    """
    aiplatform.init(project=project_id, location=region)

    # Create a model directory structure in GCS that Vertex AI expects
    # Vertex AI Model Registry expects the artifact_uri to point to a directory
    # containing model files, not individual files
    import os
    from urllib.parse import urlparse

    parsed = urlparse(model_gcs_uri)
    bucket = parsed.netloc
    model_path = parsed.path.lstrip("/")
    model_dir = "/".join(model_path.split("/")[:-1])  # Remove filename, keep directory

    # Use the model directory as artifact_uri (Vertex AI will look for model files there)
    artifact_uri = f"gs://{bucket}/{model_dir}"

    logger.info("Registering model in Vertex AI Model Registry...")
    logger.info("  Display name: %s", display_name)
    logger.info("  Artifact URI: %s", artifact_uri)
    logger.info("  Model file: %s", model_gcs_uri)
    logger.info("  Scaler file: %s", scaler_gcs_uri)

    # Register the model
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=None,  # We'll use custom container in deployment
        description=description or f"Iris classifier model registered at {datetime.utcnow().isoformat()}",
    )

    logger.info("Model registered successfully!")
    logger.info("  Resource name: %s", model.resource_name)
    logger.info("  Model ID: %s", model.name)

    return model.resource_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register model in Vertex AI Model Registry")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="Vertex AI region, e.g. us-central1")
    parser.add_argument("--model_gcs_uri", required=True, help="GCS URI to model.pkl")
    parser.add_argument("--scaler_gcs_uri", required=True, help="GCS URI to scaler.pkl")
    parser.add_argument("--display_name", required=True, help="Display name for the model")
    parser.add_argument("--description", default="", help="Optional model description")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resource_name = register_model(
        project_id=args.project_id,
        region=args.region,
        model_gcs_uri=args.model_gcs_uri,
        scaler_gcs_uri=args.scaler_gcs_uri,
        display_name=args.display_name,
        description=args.description,
    )
    print(resource_name)
