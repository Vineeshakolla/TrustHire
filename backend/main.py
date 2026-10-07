"""
TrustHire - Main FastAPI Backend Application
Initializes the FastAPI application, configures CORS middleware,
registers health check and prediction routes.
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import HealthResponse
from backend.routes.prediction import router as prediction_router
from backend.routes.analysis import router as analysis_router
from backend.services.prediction_service import PredictionService

app = FastAPI(
    title="TrustHire API",
    description="Machine-learning-based screening of job and internship postings for potentially fraudulent signals.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local development
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(prediction_router)
app.include_router(analysis_router)



@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="API Health & Model Status Check",
    tags=["System"],
)
async def health_check():
    """
    GET /health
    Returns health status and verifies that the TrustHire ML model artifacts are loaded.
    """
    model_loaded = PredictionService.is_model_available()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        service="TrustHire API",
        model_loaded=model_loaded,
    )


@app.get("/", tags=["System"])
async def root():
    """
    GET /
    Root welcome endpoint.
    """
    return {
        "service": "TrustHire API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
