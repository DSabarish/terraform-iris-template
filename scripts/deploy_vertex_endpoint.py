"""
Deploy FastAPI app to Vertex AI Endpoint with custom container.

This deploys the Docker image (built from Dockerfile) to a Vertex AI Endpoint,
linking it to a registered model in Vertex AI Model Registry.

Usage:
    python scripts/deploy_vertex_endpoint.py \
        --project_id iris-100 \
        --region us-central1 \
        --endpoint_name iris-ml-endpoint \
        --model_resource_name projects/123/locations/us-central1/models/456 \
        --image_uri us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest \
        --machine_type n1-standard-2 \
        --min_replica_count 1 \
        --max_replica_count 3
"""

import argparse
import logging
import time

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def deploy_to_vertex_endpoint(
    project_id: str,
    region: str,
    endpoint_name: str,
    model_resource_name: str,
    image_uri: str,
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
) -> str:
    """
    Deploy FastAPI app to Vertex AI Endpoint.

    Args:
        project_id: GCP project ID
        region: Vertex AI region
        endpoint_name: Name for the endpoint
        model_resource_name: Full resource name of registered model
        image_uri: Docker image URI in Artifact Registry
        machine_type: Machine type for the endpoint
        min_replica_count: Minimum number of replicas
        max_replica_count: Maximum number of replicas

    Returns:
        Endpoint resource name
    """
    aiplatform.init(project=project_id, location=region)

    logger.info("Deploying to Vertex AI Endpoint...")
    logger.info("  Endpoint name: %s", endpoint_name)
    logger.info("  Model: %s", model_resource_name)
    logger.info("  Image: %s", image_uri)
    logger.info("  Machine type: %s", machine_type)

    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if endpoints:
        endpoint = endpoints[0]
        logger.info("Using existing endpoint: %s", endpoint.resource_name)
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        logger.info("Created new endpoint: %s", endpoint.resource_name)

    # Get the model
    model = aiplatform.Model(model_resource_name)
    logger.info("Model artifact URI: %s", model.artifact_uri)

    # Deploy model to endpoint with custom container
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{endpoint_name}-deployment",
        machine_spec={
            "machine_type": machine_type,
        },
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        # Custom container configuration
        container_spec={
            "image_uri": image_uri,
            "env": [
                {"name": "VERTEX_MODEL_RESOURCE_NAME", "value": model_resource_name},
                {"name": "VERTEX_REGION", "value": region},
                {"name": "PORT", "value": "8080"},
            ],
            "ports": [{"container_port": 8080}],
        },
        traffic_percentage=100,
    )

    logger.info("Deployment initiated. Waiting for endpoint to be ready...")
    logger.info("  Deployed model: %s", deployed_model.resource_name)

    # Wait for endpoint to be ready
    while True:
        endpoint.reload()
        if endpoint.state == aiplatform.gapic.Endpoint.State.READY:
            logger.info("Endpoint is ready!")
            break
        logger.info("Endpoint state: %s, waiting...", endpoint.state)
        time.sleep(10)

    logger.info("Endpoint URL: %s", endpoint.resource_name)
    logger.info("You can now send requests to the endpoint")

    return endpoint.resource_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy to Vertex AI Endpoint")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="Vertex AI region")
    parser.add_argument("--endpoint_name", required=True, help="Endpoint display name")
    parser.add_argument("--model_resource_name", required=True, help="Full model resource name")
    parser.add_argument("--image_uri", required=True, help="Docker image URI")
    parser.add_argument("--machine_type", default="n1-standard-2", help="Machine type")
    parser.add_argument("--min_replica_count", type=int, default=1, help="Min replicas")
    parser.add_argument("--max_replica_count", type=int, default=3, help="Max replicas")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    endpoint_name = deploy_to_vertex_endpoint(
        project_id=args.project_id,
        region=args.region,
        endpoint_name=args.endpoint_name,
        model_resource_name=args.model_resource_name,
        image_uri=args.image_uri,
        machine_type=args.machine_type,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
    )
    print(endpoint_name)
