resource "google_storage_bucket" "data_bucket" {
  name          = "${var.project_id}-iris-data-${var.env}"
  location      = var.region
  force_destroy = var.env != "prod" # Protect prod data
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
}
