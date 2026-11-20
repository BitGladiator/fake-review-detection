"""
main.py — FastAPI Application for Fake Review Detection
Final-Year Project: "Fake Review Detection using Machine Learning"

Endpoints:
- GET  /api/health                     : Status, loaded models, and RAM usage
- GET  /api/metrics                    : Model validation metrics & YelpNYC metadata
- GET  /api/plots                      : Index of all evaluation charts
- GET  /api/plots/{plot_name}          : Serves high-resolution PNG charts
- POST /api/predict                    : Single-review live multi-model prediction
- POST /api/predict/batch              : CSV / Excel file scanning with chunked inference
- GET  /api/predict/batch/export/{id}  : Downloadable CSV report
- GET  /api/samples                    : 6 curated reviews for 1-click frontend testing
"""

import os
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import psutil
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictResponse,
    SamplesResponse,
    SampleReview,
    HealthResponse,
    MetricsResponse,
)
from ml_service import ml_service, PLOTS_DIR, ARTIFACTS_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Load heavy ML artifacts once into memory at startup.
    Keeps memory footprint low (<350MB) and delivers sub-15ms request latencies.
    """
    logger.info("Initializing Fake Review Detection backend service...")
    try:
        ml_service.load_artifacts()
    except Exception as e:
        logger.error("Failed to load ML artifacts at startup: %s", e)
        raise e

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info("Backend ready. Resident memory: %.1f MB", mem_mb)
    yield
    logger.info("Shutting down backend service.")


app = FastAPI(
    title="Fake Review Detection API",
    description="Machine learning API for detecting deceptive, computer-generated, and spam reviews on YelpNYC.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health, loaded classifiers, and live memory consumption."""
    process = psutil.Process(os.getpid())
    mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    return HealthResponse(
        status="healthy",
        loaded_models=list(ml_service.models.keys()),
        memory_usage_mb=mem_mb,
        device="Apple Silicon (M2 Optimized) / CPU",
        artifacts_loaded=ml_service.is_loaded,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Metrics & Analytics Dashboard
# ---------------------------------------------------------------------------
@app.get("/api/metrics", response_model=MetricsResponse, tags=["Analytics"])
def get_metrics():
    """Serves model performance metrics (Accuracy, ROC-AUC, F1, Confusion Matrices) & YelpNYC metadata."""
    dataset_metadata = {
        "dataset_name": "YelpNYC Cleaned Dataset",
        "total_reviews": 359052,
        "genuine_reviews": 322167,
        "fake_reviews": 36885,
        "spam_prevalence": "10.27%",
        "features": {
            "word_tfidf_features": 12000,
            "char_tfidf_features": 8000,
            "behavioral_features": 13,
            "total_sparse_dimensions": 20013,
        },
        "hardware_safety": "Sparse float32 matrix pipeline tuned for 8GB RAM Apple M2 Air",
    }
    return MetricsResponse(
        metrics=ml_service.metrics_data,
        dataset_metadata=dataset_metadata,
    )


# ---------------------------------------------------------------------------
# Visualizations & Plots
# ---------------------------------------------------------------------------
PLOTS_METADATA = [
    {
        "filename": "01_class_distribution.png",
        "title": "Dataset Class Distribution",
        "description": "Visualizes the extreme 90:10 real-world class imbalance of YelpNYC (322k Genuine vs. 37k Spam).",
    },
    {
        "filename": "02_rating_and_length_eda.png",
        "title": "Rating & Review Length EDA",
        "description": "Exploratory distribution showing spam polarity bi-modality (extreme 1 & 5 stars) and length differences.",
    },
    {
        "filename": "03_model_comparison_metrics.png",
        "title": "Model Comparison Benchmarks",
        "description": "Side-by-side comparison of Accuracy, ROC-AUC, Spam Recall, and Spam F1 across all 9 model permutations.",
    },
    {
        "filename": "04_confusion_matrices_grid.png",
        "title": "3×3 Confusion Matrices Grid",
        "description": "Heatmap grid detailing True Negatives, False Positives, False Negatives, and True Positives across all samplers.",
    },
    {
        "filename": "05_roc_and_pr_curves.png",
        "title": "ROC & Precision-Recall Curves",
        "description": "Overlaid ROC and PR curves showing area-under-curve discrimination on the 71,811-row test split.",
    },
    {
        "filename": "06_top_spam_features.png",
        "title": "Top Spam vs. Genuine Feature Importances",
        "description": "Feature weights detailing behavioral signals, exclamation markers, and lexical tokens.",
    },
]


@app.get("/api/plots", tags=["Visualizations"])
def list_plots():
    """List all available PNG evaluation charts with titles and descriptions."""
    available = []
    for item in PLOTS_METADATA:
        path = os.path.join(PLOTS_DIR, item["filename"])
        available.append({
            **item,
            "exists": os.path.exists(path),
            "url": f"/api/plots/{item['filename']}",
        })
    return {"plots": available}


@app.get("/api/plots/{plot_name}", tags=["Visualizations"])
def get_plot(plot_name: str):
    """Serve a high-resolution PNG evaluation chart generated by model/visualize.py."""
    safe_name = os.path.basename(plot_name)
    if not safe_name.endswith(".png"):
        safe_name += ".png"

    path = os.path.join(PLOTS_DIR, safe_name)
    if not os.path.exists(path):
        valid = [p["filename"] for p in PLOTS_METADATA]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot '{safe_name}' not found. Available charts: {valid}",
        )
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Single-Review Prediction
# ---------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
def predict_review(req: PredictRequest):
    """
    Analyze a single review for deceptive patterns.
    Computes linguistic signals and runs live inference on XGBoost, LinearSVC, and SGD
    to provide both the requested model's verdict and a 3-way consensus breakdown.
    """
    try:
        return ml_service.predict_single(
            text=req.review_text,
            rating=req.rating,
            model_choice=req.model_name,
            user_id=req.user_id,
            product_id=req.product_id,
        )
    except Exception as e:
        logger.error("Single review prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ---------------------------------------------------------------------------
# Batch File Scanning
# ---------------------------------------------------------------------------
@app.post("/api/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch_file(
    file: UploadFile = File(..., description="CSV or Excel file with review text"),
    model_name: str = Form(default="ensemble", description="Model choice: ensemble, xgboost, linearsvc, sgd"),
):
    """
    Upload a CSV or Excel spreadsheet of reviews to perform batch fake-review detection.
    Processes data in memory-safe 5,000-row chunks.
    """
    filename = file.filename or "upload.csv"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a .csv or .xlsx file.",
        )

    try:
        contents = await file.read()
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse spreadsheet: {str(e)}",
        )

    if len(df) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        result = ml_service.predict_batch_df(df, model_choice=model_name)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error("Batch prediction failure: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch error: {str(e)}")


@app.get("/api/predict/batch/export/{export_id}", tags=["Inference"])
def export_batch_results(export_id: str):
    """Download the batch analysis results as a standardized CSV file."""
    csv_data = ml_service.get_export_csv(export_id)
    if not csv_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export session expired or not found. Please re-run the batch file scan.",
        )
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=fake_review_audit_{export_id[:8]}.csv"
        },
    )


# ---------------------------------------------------------------------------
# Curated Test Samples
# ---------------------------------------------------------------------------
CURATED_SAMPLES = [
    SampleReview(
        id=1,
        title="Authentic Romantic Dinner",
        review_text="Great food, friendly staff, and the ambiance was perfect for a date night. Will definitely come back!",
        rating=4.0,
        expected_label="Genuine",
        category="Restaurant Dining",
        explanation="Moderate tone, natural sentence flow, and specific personal context (date night).",
    ),
    SampleReview(
        id=2,
        title="Balanced Food & Service Critique",
        review_text="The pasta was cooked to perfection. Service was a bit slow but worth the wait.",
        rating=4.0,
        expected_label="Genuine",
        category="Italian Dining",
        explanation="Realistic nuanced critique balancing food praise with minor service wait time.",
    ),
    SampleReview(
        id=3,
        title="Moderate Everyday Feedback",
        review_text="Decent place, nothing extraordinary. The burger was good but a little overpriced for the portion size.",
        rating=3.0,
        expected_label="Genuine",
        category="Casual Dining",
        explanation="Unbiased, measured opinion with balanced score and realistic consumer critique.",
    ),
    SampleReview(
        id=4,
        title="Promotional Buzzword Spam",
        review_text="BEST RESTAURANT EVER!!!! AMAZING FOOD!!!! MUST VISIT!!!! 5 STARS!!!! GO NOW!!!!",
        rating=5.0,
        expected_label="Fake",
        category="Promotional Bot",
        explanation="Excessive exclamation marks (>10), all-caps emphasis, and empty superlative hype.",
    ),
    SampleReview(
        id=5,
        title="Repetitive Hyperbolic Spam",
        review_text="Perfect perfect perfect! Everything was absolutely perfect! Best experience of my life! Perfect!",
        rating=5.0,
        expected_label="Fake",
        category="Repetitive Bot",
        explanation="Excessive repetition of the token 'perfect' with hyperbolic praise and zero details.",
    ),
    SampleReview(
        id=6,
        title="Incentivized / Paid Review",
        review_text="I was paid to write this review. The food was okay I guess. Please give 5 stars they asked me to.",
        rating=5.0,
        expected_label="Fake",
        category="Compensated Review",
        explanation="Explicit confession of payment and direct solicitation for a 5-star rating.",
    ),
]


@app.get("/api/samples", response_model=SamplesResponse, tags=["Testing"])
def get_sample_reviews():
    """Returns 6 curated sample reviews (3 genuine, 3 fake) for 1-click frontend testing."""
    return SamplesResponse(samples=CURATED_SAMPLES)
