
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "env" {
  description = "Environment (dev, qa, prod)"
  type        = string
}

variable "bq_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "image_uri" {
  description = "URI of the Docker image to deploy"
  type        = string
}

variable "model_prefix" {
  description = "GCS prefix for model artifacts"
  type        = string
  default     = "artifacts/models"
}

variable "scaler_prefix" {
  description = "GCS prefix for scaler artifacts"
  type        = string
  default     = "artifacts/scalers"
}
