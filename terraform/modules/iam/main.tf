resource "google_service_account" "cloud_run_sa" {
  account_id   = "iris-runner-${var.env}"
  display_name = "Cloud Run Service Account (${var.env})"
  project      = var.project_id
}

# Grant Permissions to Cloud Run SA
# 1. Read/Write to GCS (Data/Models) - Granular would be better, but Object Admin is safe for this project scope
resource "google_project_iam_member" "gcs_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# 2. BigQuery User (if app needs to query BQ, which it doesn't seem to, but pipeline does)
# Pipeline uses CI/CD SA, so this might not be needed for Cloud Run SA.
# Keeping it minimal as per "Least privilege IAM" in README.
