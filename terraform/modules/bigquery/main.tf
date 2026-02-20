resource "google_bigquery_dataset" "dataset" {
  dataset_id  = "iris_${var.env}"
  project     = var.project_id
  location    = var.location
  description = "Iris dataset for ${var.env} environment"
}

resource "google_bigquery_table" "raw_table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "iris_raw"
  project    = var.project_id
  
  # Schema defined in Terraform to ensure consistency
  schema = <<EOF
[
  {
    "name": "sepal_length",
    "type": "FLOAT",
    "mode": "NULLABLE"
  },
  {
    "name": "sepal_width",
    "type": "FLOAT",
    "mode": "NULLABLE"
  },
  {
    "name": "petal_length",
    "type": "FLOAT",
    "mode": "NULLABLE"
  },
  {
    "name": "petal_width",
    "type": "FLOAT",
    "mode": "NULLABLE"
  },
  {
    "name": "target",
    "type": "INTEGER",
    "mode": "NULLABLE"
  },
  {
    "name": "target_name",
    "type": "STRING",
    "mode": "NULLABLE"
  }
]
EOF
}
