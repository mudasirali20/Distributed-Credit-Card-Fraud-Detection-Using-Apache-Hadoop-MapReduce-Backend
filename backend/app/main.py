from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from app.config import MODEL_PATH, DASHBOARD_IMAGE_PATH, ROOT_DIR
from app.data import data_store
from app.model import model_service
from app.schemas import (
    HealthResponse,
    SummaryResponse,
    PredictRequest,
    PredictResponse,
    BatchPredictResponse,
    FeatureImportanceItem,
    ModelMetricRow,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify and load all data on startup
    try:
        data_store.load_all()
    except Exception as e:
        print(f"CRITICAL ERROR loading data files: {e}")
        raise e

    try:
        model_service.load()
    except Exception as e:
        print(f"CRITICAL ERROR loading ML model: {e}")
        raise e

    print("FastAPI application started successfully. All data and ML model loaded.")
    yield
    print("FastAPI application shutting down.")

app = FastAPI(
    title="Distributed Credit Card Fraud Detection API",
    description="High-performance backend API serving Hadoop MapReduce analytics, live ML model inference, and explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware for seamless local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom validation exception handler for clear 422 messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        errors.append(f"{field}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": errors},
    )

@app.get("/api/health", response_model=HealthResponse)
def get_health():
    is_loaded = model_service.is_loaded()
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_file=MODEL_PATH.name,
    )

@app.get("/api/summary", response_model=SummaryResponse)
def get_summary():
    try:
        return data_store.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")

@app.get("/api/stats/class")
def get_stats_class() -> List[Dict[str, Any]]:
    return data_store.get_hadoop_class_stats()

@app.get("/api/stats/amount")
def get_stats_amount() -> List[Dict[str, Any]]:
    return data_store.get_hadoop_amount_stats()

@app.get("/api/stats/hour")
def get_stats_hour() -> List[Dict[str, Any]]:
    return data_store.get_hadoop_hour_stats()

@app.get("/api/metrics", response_model=List[ModelMetricRow])
def get_metrics():
    return data_store.get_metrics()

@app.get("/api/suspicious")
def get_suspicious() -> List[Dict[str, Any]]:
    return data_store.get_suspicious()

@app.get("/api/samples")
def get_samples() -> List[Dict[str, Any]]:
    return data_store.get_samples()

@app.get("/api/samples/{sample_id}")
def get_sample_by_id(sample_id: int):
    sample = data_store.get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample with ID {sample_id} not found.")
    return sample

@app.get("/api/feature-importance", response_model=List[FeatureImportanceItem])
def get_feature_importance():
    return model_service.get_top_feature_importances(top_n=10)

@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        return model_service.predict_single(payload.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only CSV files are supported.",
        )
    content = await file.read()
    try:
        return model_service.predict_batch(content)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/api/dashboard-image")
def get_dashboard_image():
    if not DASHBOARD_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Dashboard image not found.")
    return FileResponse(DASHBOARD_IMAGE_PATH, media_type="image/png")

# Serve built React frontend in production (Single-service deployment)
from fastapi.staticfiles import StaticFiles

DIST_DIR = ROOT_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Don't intercept API routes (safety check)
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = DIST_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")

