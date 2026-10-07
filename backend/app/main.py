"""
TrustHire Backend - FastAPI Main Application Entry Point
"""

from fastapi import FastAPI
from backend.app.config import PROJECT_NAME, PROJECT_DESCRIPTION, PROJECT_VERSION

app = FastAPI(
    title=PROJECT_NAME,
    description=PROJECT_DESCRIPTION,
    version=PROJECT_VERSION,
)


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify backend service status.
    """
    return {
        "status": "ok",
        "project": PROJECT_NAME
    }
