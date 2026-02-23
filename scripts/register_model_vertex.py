"""
Register trained model from GCS to Vertex AI Model Registry.

Usage:
    python scripts/register_model_vertex.py \
        --project_id iris-100 \
        --region us-central1 \
        --model_gcs_uri gs://bucket/artifacts/models/latest/model.pkl \
        --scaler_gcs_uri gs://bucket/artifacts/scalers/scaler.pkl \
        --display_name iris-classifier \
        --serving_container_image_uri us-central1-docker.pkg.dev/project/iris-ml/iris-ml-api:latest \
        --description "Iris flower classification model"
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfg import get_config
from google.cloud import aiplatform

_config = get_config()

# CRITICAL: stream=sys.stderr ensures ALL log output goes to stderr only.
# stdout must be clean so that shell $(...) captures only the printed resource name.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def register_model(
    project_id: str,
    region: str,
    model_gcs_uri: str,
    scaler_gcs_uri: str,
    display_name: str,
    serving_container_image_uri: str,
    description: str = "",
) -> str:
    aiplatform.init(project=project_id, location=region)

    parsed = urlparse(model_gcs_uri)
    bucket = parsed.netloc
    model_path = parsed.path.lstrip("/")
    model_dir = "/".join(model_path.split("/")[:-1])
    artifact_uri = f"gs://{bucket}/{model_dir}"

    logger.info("Registering model in Vertex AI Model Registry...")
    logger.info("  Display name: %s", display_name)
    logger.info("  Artifact URI: %s", artifact_uri)
    logger.info("  Model file: %s", model_gcs_uri)
    logger.info("  Scaler file: %s", scaler_gcs_uri)
    logger.info("  Serving container: %s", serving_container_image_uri)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        description=description or f"Iris classifier registered at {datetime.utcnow().isoformat()}",
    )

    logger.info("Model registered successfully!")
    logger.info("  Resource name: %s", model.resource_name)
    logger.info("  Model ID: %s", model.name)

    return model.resource_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register model in Vertex AI Model Registry")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model_gcs_uri", required=True)
    parser.add_argument("--scaler_gcs_uri", required=True)
    parser.add_argument("--display_name", required=True)
    parser.add_argument("--serving_container_image_uri", required=True)
    parser.add_argument("--description", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resource_name = register_model(
        project_id=args.project_id,
        region=args.region,
        model_gcs_uri=args.model_gcs_uri,
        scaler_gcs_uri=args.scaler_gcs_uri,
        display_name=args.display_name,
        serving_container_image_uri=args.serving_container_image_uri,
        description=args.description,
    )
    # ONLY the resource name goes to stdout — nothing else
    print(resource_name)