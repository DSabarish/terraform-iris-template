"""
Submit Iris pipeline as a Vertex AI Custom Job.

This runs the SAME steps as src/pipeline.py, but on managed GCP
infrastructure instead of your local machine or GitHub runner.

It assumes you already have a container image in Artifact Registry
that looks like this project's Dockerfile (Python 3.11 + this code
and requirements installed). The job simply overrides the CMD to:

    python pipeline.py --project_id ... --dataset_id ... --table_id ... --gcs_bucket ...

Usage (example):

    python scripts/run_vertex_custom_job.py ^
      --project_id iris-100 ^
      --region us-central1 ^
      --gcs_bucket iris-100-ml-bucket ^
      --image_uri us-central1-docker.pkg.dev/iris-100/iris-ml/iris-ml-api:latest ^
      --service_account iris-ml-github@iris-100.iam.gserviceaccount.com
"""

import argparse
import datetime
import logging
from typing import Optional

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def submit_vertex_custom_job(
    project_id: str,
    region: str,
    gcs_bucket: str,
    image_uri: str,
    service_account: str,
    dataset_id: str = "iris_dataset",
    table_id: str = "iris_raw",
    machine_type: str = "n1-standard-4",
    staging_bucket: Optional[str] = None,
) -> str:
    """Create and run a Vertex AI Custom Job that calls pipeline.py."""

    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    display_name = f"iris-pipeline-custom-job-{ts}"

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "command": ["python", "pipeline.py"],
                "args": [
                    "--project_id",
                    project_id,
                    "--dataset_id",
                    dataset_id,
                    "--table_id",
                    table_id,
                    "--gcs_bucket",
                    gcs_bucket,
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
    )

    logger.info("Submitting Vertex AI Custom Job: %s", display_name)
    job.run(service_account=service_account, sync=True)
    logger.info("Custom Job state: %s", job.state)

    return job.resource_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iris pipeline as a Vertex AI Custom Job")
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="Vertex AI region, e.g. us-central1")
    parser.add_argument("--gcs_bucket", required=True, help="GCS bucket for data and models")
    parser.add_argument("--image_uri", required=True, help="Container image URI in Artifact Registry")
    parser.add_argument("--service_account", required=True, help="Service account email for the job")
    parser.add_argument("--dataset_id", default="iris_dataset", help="BigQuery dataset ID")
    parser.add_argument("--table_id", default="iris_raw", help="BigQuery table ID")
    parser.add_argument(
        "--machine_type",
        default="n1-standard-4",
        help="Vertex AI machine type, e.g. n1-standard-4",
    )
    parser.add_argument(
        "--staging_bucket",
        default=None,
        help="Optional staging bucket for Vertex AI, e.g. gs://my-staging-bucket",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    job_name = submit_vertex_custom_job(
        project_id=args.project_id,
        region=args.region,
        gcs_bucket=args.gcs_bucket,
        image_uri=args.image_uri,
        service_account=args.service_account,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        machine_type=args.machine_type,
        staging_bucket=args.staging_bucket,
    )
    print(job_name)

