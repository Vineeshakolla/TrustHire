"""
TrustHire - Prediction Router Module
Defines POST /predict endpoint for single job posting fraud screening.
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas import JobPostingRequest, PredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter(prefix="", tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen Job Posting",
    description="Analyzes job posting attributes using the trained TrustHire machine-learning pipeline to estimate potential fraud risk.",
)
async def predict_job(request: JobPostingRequest):
    """
    POST /predict
    Accepts job posting details, runs screening pipeline, and returns risk assessment.
    """
    try:
        if not PredictionService.is_model_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"success": False, "error": "Model or preprocessing pipeline artifact is unavailable."},
            )

        response = PredictionService.predict(request)
        return response

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "error": str(ve)},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error": "An error occurred during prediction processing."},
        )
