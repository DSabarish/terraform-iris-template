"""
Deploy FastAPI app to Cloud Run, configured to use Vertex AI Model Registry.

Usage:
    python scripts/deploy_cloud_run_vertex.py \
        --project_id iris-100 \
        --region us-central1 \
        --service_name iris-ml-api \
        --image_uri us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest \
        --model_display_name iris-classifier-latest \
        --service_account iris-ml-github@iris-100.iam.gserviceaccount.com
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def deploy_to_cloud_run(
    project_id: str,
    region: str,
    service_name: str,
    image_uri: str,
    model_display_name: str,
    service_account: str,
) -> str:
    logger.info("Deploying to Cloud Run...")
    logger.info("  Service name: %s", service_name)
    logger.info("  Image: %s", image_uri)
    logger.info("  Model (Vertex AI): %s", model_display_name)
    logger.info("  Service account: %s", service_account)
    logger.info("  Region: %s", region)
    logger.info("  Project: %s", project_id)

    cmd = [
        "gcloud", "run", "deploy", service_name,
        "--image", image_uri,
        "--region", region,
        "--platform", "managed",
        "--allow-unauthenticated",
        "--service-account", service_account,
        "--set-env-vars",
        f"VERTEX_MODEL_DISPLAY_NAME={model_display_name},VERTEX_REGION={region},GOOGLE_CLOUD_PROJECT={project_id}",
        "--project", project_id,
        "--quiet",
    ]

    logger.info("Running command: %s", " ".join(cmd))

    # Use stdout/stderr passthrough so errors are visible in CI logs
    result = subprocess.run(cmd, text=True, check=False)

    if result.returncode != 0:
        logger.error("gcloud run deploy failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    # Get service URL
    url_cmd = [
        "gcloud", "run", "services", "describe", service_name,
        "--region", region,
        "--project", project_id,
        "--format", "value(status.url)",
    ]
    url_result = subprocess.run(url_cmd, capture_output=True, text=True, check=True)
    service_url = url_result.stdout.strip()

    logger.info("Deployment complete!")
    logger.info("  Service URL: %s", service_url)

    return service_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy to Cloud Run with Vertex AI Model Registry")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--service_name", required=True)
    parser.add_argument("--image_uri", required=True)
    parser.add_argument("--model_display_name", required=True)
    parser.add_argument("--service_account", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    service_url = deploy_to_cloud_run(
        project_id=args.project_id,
        region=args.region,
        service_name=args.service_name,
        image_uri=args.image_uri,
        model_display_name=args.model_display_name,
        service_account=args.service_account,
    )
    print(service_url)