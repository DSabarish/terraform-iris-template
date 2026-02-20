terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state — configured per environment via backend.conf
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─────────────────────────────────────────────────
# Enable Required APIs
# ─────────────────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])
  project                    = var.project_id
  service                    = each.key
  disable_on_destroy         = false
  disable_dependent_services = false
}

# ─────────────────────────────────────────────────
# Service Account for CI/CD
# ─────────────────────────────────────────────────
module "iam" {
  source     = "./modules/iam"
  project_id = var.project_id
  env        = var.env

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────
# BigQuery Dataset + Table
# ─────────────────────────────────────────────────
module "bigquery" {
  source     = "./modules/bigquery"
  project_id = var.project_id
  env        = var.env
  location   = var.bq_location

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────
# GCS Buckets (Data + Models)
# ─────────────────────────────────────────────────
module "gcs" {
  source     = "./modules/gcs"
  project_id = var.project_id
  env        = var.env
  region     = var.region

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────
# Artifact Registry
# ─────────────────────────────────────────────────
module "artifact_registry" {
  source     = "./modules/artifact_registry"
  project_id = var.project_id
  env        = var.env
  region     = var.region

  depends_on = [google_project_service.apis]
}

# ─────────────────────────────────────────────────
# Cloud Run Service
# ─────────────────────────────────────────────────
module "cloud_run" {
  source              = "./modules/cloud_run"
  project_id          = var.project_id
  env                 = var.env
  region              = var.region
  image_uri           = var.image_uri
  gcs_bucket          = module.gcs.data_bucket_name
  model_prefix        = var.model_prefix
  scaler_prefix       = var.scaler_prefix
  service_account_email = module.iam.cloud_run_sa_email

  depends_on = [
    module.iam,
    module.gcs,
    module.artifact_registry,
  ]
}
