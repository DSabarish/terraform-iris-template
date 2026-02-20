locals {
  is_prod = var.env == "prod"
}

resource "google_cloud_run_service" "api" {
  name     = "iris-ml-api-${var.env}"
  location = var.region
  project  = var.project_id

  template {
    spec {
      service_account_name = var.service_account_email
      
      containers {
        image = var.image_uri
        
        resources {
          limits = {
            cpu    = local.is_prod ? "2000m" : "1000m"
            memory = local.is_prod ? "2Gi"   : "512Mi"
          }
        }

        env {
          name  = "GCS_BUCKET"
          value = var.gcs_bucket
        }
        env {
          name  = "MODEL_PREFIX"
          value = var.model_prefix
        }
        env {
          name  = "SCALER_PREFIX"
          value = var.scaler_prefix
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = local.is_prod ? "1" : "0"
        "autoscaling.knative.dev/maxScale" = "10"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Allow unauthenticated invocations for Dev/QA (optional, usually Dev is public for extensive testing, Prod might be public or private depending on use case)
# Prompt says: "Prod-specific: ... allow_unauthenticated=false"
# Implicitly, Dev/QA allow it? Or maybe QA doesn't.
# Standard practice: Dev public for ease, Prod private/authenticated (requires IAM).
# Let's follow the prompt strictly: "Prod-specific: allow_unauthenticated=false". 
# This implies Dev/QA ARE allowed.

resource "google_cloud_run_service_iam_member" "public_access" {
  count    = local.is_prod ? 0 : 1
  location = google_cloud_run_service.api.location
  project  = google_cloud_run_service.api.project
  service  = google_cloud_run_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
