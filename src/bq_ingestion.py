"""
bq_ingestion.py — Load Iris data into BigQuery. Uses cfg/base.yaml for dataset/table defaults.
"""

import argparse
import logging

from google.cloud import bigquery
from sklearn.datasets import load_iris
import pandas as pd

from cfg import get_config

_config = get_config()
logging.basicConfig(level=logging.INFO, format=_config["logging"]["format"])
logger = logging.getLogger(__name__)


def load_iris_to_bq(project_id: str, dataset_id: str, table_id: str) -> None:
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    logger.info("Loaded %d rows into %s", len(df), table_ref)


def parse_args():
    bq = _config["bigquery"]
    parser = argparse.ArgumentParser(description="Ingest Iris data into BigQuery")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", default=bq["dataset_id"])
    parser.add_argument("--table_id", default=bq["table_id"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_iris_to_bq(args.project_id, args.dataset_id, args.table_id)
