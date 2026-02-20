"""
app.py
------
FastAPI application serving Iris predictions and optional static frontend.
Deployed on Cloud Run.

Endpoints:
  GET  /               - Frontend (index.html) when frontend dir is present
  GET  /health         - Health check
  GET  /model-info     - Model metadata
  POST /predict        - Single prediction
  POST /predict/batch  - Batch predictions
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator

from inference import predict, load_model, load_scaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup."""
    logger.info("Pre-loading model and scaler...")
    load_model()
    load_scaler()
    logger.info("Model and scaler ready.")
    yield


app = FastAPI(
    title="Iris ML API",
    description="Production-ready Iris flower classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser requests from file:// or any origin (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend at / when frontend dir is present (Docker: COPY frontend/ ./frontend/)
_this_dir = Path(__file__).resolve().parent
_frontend_dir = _this_dir / "frontend" if (_this_dir / "frontend").exists() else _this_dir.parent / "frontend"
if _frontend_dir.exists():
    @app.get("/", tags=["Frontend"])
    def serve_frontend():
        index = _frontend_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend not found")
    logger.info("Serving frontend from %s", _frontend_dir)
else:
    logger.info("No frontend directory found; / will 404.")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20, example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=20, example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=20, example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=20, example=0.2, description="Petal width in cm")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: dict


class BatchPredictionRequest(BaseModel):
    instances: List[IrisFeatures]


@app.get("/health", tags=["Ops"])
def health():
    return {"status": "ok", "service": "iris-ml-api"}


@app.get("/model-info", tags=["Ops"])
def model_info():
    return {
        "model_bucket": os.environ.get("GCS_BUCKET"),
        "model_prefix": os.environ.get("MODEL_PREFIX", "artifacts/models/latest"),
        "scaler_prefix": os.environ.get("SCALER_PREFIX", "artifacts/scalers"),
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def single_predict(features: IrisFeatures):
    try:
        result = predict([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ])
        return result
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Inference"])
def batch_predict(request: BatchPredictionRequest):
    results = []
    for inst in request.instances:
        try:
            result = predict([
                inst.sepal_length,
                inst.sepal_width,
                inst.petal_length,
                inst.petal_width,
            ])
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped by user (CTRL+C)")
        raise SystemExit(0)
