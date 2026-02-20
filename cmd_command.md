# Run Iris ML pipeline from CMD (no Terraform / no CI/CD)

Use these steps with:
- **Project ID:** `iris-100`
- **GCS bucket:** `iris-bucket-1`
- **BigQuery dataset:** `iris_dataset` (create once in Step 0)

---

## Prerequisites

> link: console.cloud.google.com/iam-admin/serviceaccounts/

1. **Python 3.10+** installed.
2. **GCP auth** (one of):
   - `gcloud auth application-default login`
   - Or set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON path:
     - **CMD:** `set GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"`
     - **PowerShell:** `$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"`
     - To verify in PowerShell: `echo $env:GOOGLE_APPLICATION_CREDENTIALS`
3. **gcloud** CLI installed (for BigQuery dataset creation).

---

## Step 0 — One-time: Create BigQuery dataset

The pipeline loads Iris into BigQuery. The dataset must exist first.

```cmd
gcloud config set project iris-100
bq mk --dataset --location=US iris-100:iris_dataset
```

*(If the dataset already exists, you can skip this.)*

---

## Step 1 — Install Python dependencies

From the **project root** (where `requirements.txt` is):

```cmd
cd c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project
pip install -r requirements.txt
```

---

## Step 2 — Ingest Iris data into BigQuery

Loads the Iris dataset into BigQuery table `iris_raw` in dataset `iris_dataset`.

```cmd
cd c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project
python src\bq_ingestion.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw
```

---

## Step 3 — Export BigQuery to GCS (raw CSV)

Exports the BigQuery table to `gs://iris-bucket-1/data/raw/iris_raw.csv`.

```cmd
python src\load_data_from_bq.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw --gcs_bucket iris-bucket-100
```

---

## Step 4 — Clean raw data

Reads `data/raw/iris_raw.csv`, cleans it, writes `data/clean/iris_clean.csv` in the same bucket.

```cmd
python src\clean_raw_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
```

---

## Step 5 — Transform data and save scaler

Reads clean data, scales features, adds engineered features, writes:
- `data/transformed/iris_transformed.csv`
- `artifacts/scalers/scaler.pkl`

```cmd
python src\transform_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
```

---

## Step 6 — Train model and save to GCS

Trains a RandomForest, saves:
- `artifacts/models/local/model.pkl` (and `metrics.json`)
- `artifacts/models/latest/model.pkl` (and `metrics.json`)

```cmd
python src\train.py --project_id iris-100 --gcs_bucket iris-bucket-100
```

---

## Step 7 — Run the API locally (optional)

Starts FastAPI so you can call `/health` and `/predict`. **You must set credentials and env vars in the same session** (the app loads model/scaler from GCS at startup).

**PowerShell (recommended — run these in order):**

```powershell
# 1) Set GCP credentials (required for GCS)
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"

# 2) Set bucket and paths (use the same bucket you used in train step)
$env:GCS_BUCKET = "iris-bucket-100"
$env:MODEL_PREFIX = "artifacts/models/latest"
$env:SCALER_PREFIX = "artifacts/scalers"

# 3) Go to src and start the API (default port 8080)
cd "c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project\src"
python app.py
```

**One-liner (PowerShell):**

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"; $env:GCS_BUCKET = "iris-bucket-100"; $env:MODEL_PREFIX = "artifacts/models/latest"; $env:SCALER_PREFIX = "artifacts/scalers"; cd "c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project\src"; python app.py
```

**CMD:**

```cmd
set "GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"
set GCS_BUCKET=iris-bucket-100
set MODEL_PREFIX=artifacts/models/latest
set SCALER_PREFIX=artifacts/scalers
cd "c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project\src"
python app.py
```

Then in **another** terminal (app runs on port **8080** by default):

- Health: `curl http://127.0.0.1:8080/health`
- Predict:  
  `curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"`

If you see GCS/permission errors, ensure the service account has access to `iris-bucket-100` and that `artifacts/models/latest/model.pkl` and `artifacts/scalers/scaler.pkl` exist in that bucket (from Step 6).

---

## Step 8 — Frontend (optional)

A single-page HTML in `frontend/index.html` calls all endpoints and shows results.

1. Start the API (Step 7) so it is running on `http://127.0.0.1:8080`.
2. Open the HTML file in a browser:
   - Double-click `frontend/index.html`, or
   - Drag it into a browser tab.
3. Leave **API base URL** as `http://127.0.0.1:8080` (or change it if your API is elsewhere).
4. Use **Check Health**, **Model Info**, **Predict** (single), and **Batch Predict**; results appear below each section.

The API has CORS enabled so the browser can call it from this page.

---

## Summary — copy-paste order (from project root)

**CMD:**
```cmd
set "GOOGLE_APPLICATION_CREDENTIALS=C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"
cd c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project
pip install -r requirements.txt
python src\bq_ingestion.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw
python src\load_data_from_bq.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw --gcs_bucket iris-bucket-100
python src\clean_raw_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
python src\transform_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
python src\train.py --project_id iris-100 --gcs_bucket iris-bucket-100
```

**PowerShell:**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\Selvam Sabarish\Documents\iris-100-41644c1783e6.json"
cd c:\Users\Selvam Sabarish\Desktop\sabs\my_work_on_cnp_projects\DN_Terraform_CICD_Sample_Project
pip install -r requirements.txt
python src\bq_ingestion.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw
python src\load_data_from_bq.py --project_id iris-100 --dataset_id iris_dataset --table_id iris_raw --gcs_bucket iris-bucket-100
python src\clean_raw_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
python src\transform_data.py --project_id iris-100 --gcs_bucket iris-bucket-100
python src\train.py --project_id iris-100 --gcs_bucket iris-bucket-100
```

Then run the API (Step 7) if you want to test predictions locally.
