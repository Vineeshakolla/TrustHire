"""
TrustHire - Prediction Service Module
Encapsulates backend service logic connecting FastAPI endpoints to the local ML prediction module.
"""

from typing import Dict, Any, List
from backend.config import MODEL_PATH, PREPROCESSING_PATH
from backend.schemas import JobPostingRequest, PredictionResponse, SignalItem
from ml.prediction.predict import predict_job_posting, load_prediction_artifacts


class PredictionService:
    """
    Service class handling model availability verification and prediction execution.
    """

    @staticmethod
    def is_model_available() -> bool:
        """
        Verifies whether trained model and preprocessing pipeline artifacts are available and loadable.
        """
        try:
            if not MODEL_PATH.exists() or not PREPROCESSING_PATH.exists():
                return False
            load_prediction_artifacts()
            return True
        except Exception:
            return False

    @staticmethod
    def predict(req: JobPostingRequest) -> PredictionResponse:
        """
        Executes job posting screening using the underlying ML pipeline and transparent Risk Fusion.
        """
        job_data = {
            "title": req.title,
            "description": req.description,
            "company_profile": req.company_profile,
            "requirements": req.requirements,
            "benefits": req.benefits,
            "location": req.location,
            "department": req.department,
            "salary_range": req.salary_range,
            "employment_type": req.employment_type,
            "required_experience": req.required_experience,
            "required_education": req.required_education,
            "industry": req.industry,
            "function": req.function,
            "telecommuting": req.telecommuting,
            "has_company_logo": req.has_company_logo,
            "has_questions": req.has_questions,
        }

        # Invoke core ML prediction module
        ml_result = predict_job_posting(job_data, threshold=req.threshold or 0.50)

        # Format feature signals into SignalItem Pydantic models
        formatted_signals: List[SignalItem] = []

        # Process positive risk signals
        for sig in ml_result.get("top_positive_signals", []):
            formatted_signals.append(
                SignalItem(
                    feature=sig["feature"],
                    direction="risk_associated",
                    importance=round(float(sig["contribution"]), 4),
                )
            )

        # Process negative legitimate signals
        for sig in ml_result.get("top_negative_signals", []):
            formatted_signals.append(
                SignalItem(
                    feature=sig["feature"],
                    direction="legitimate_associated",
                    importance=round(abs(float(sig["contribution"])), 4),
                )
            )

        return PredictionResponse(
            success=True,
            prediction=ml_result["prediction_label"],
            risk_score=ml_result["risk_score"],
            risk_percentage=ml_result["risk_percentage"],
            risk_level=ml_result["risk_level"],
            ml_risk_score=ml_result.get("ml_risk_score"),
            ml_risk_percentage=ml_result.get("ml_risk_percentage"),
            final_risk_score=ml_result.get("final_risk_score"),
            final_risk_percentage=ml_result.get("final_risk_percentage"),
            trusthire_assessment=ml_result.get("trusthire_assessment"),
            nlp_risk_penalty=ml_result.get("nlp_risk_penalty"),
            threshold=ml_result["threshold_used"],
            model=ml_result["model_name"],
            signals=formatted_signals,
            warning_signals=ml_result.get("warning_signals", []),
            disclaimer=ml_result["disclaimer"],
        )
