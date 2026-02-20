resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "iris-ml-${var.env}"
  description   = "Docker repository for Iris ML API (${var.env})"
  format        = "DOCKER"
  project       = var.project_id
}
