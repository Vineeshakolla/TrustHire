"""
TrustHire - AI Job Information Extraction & Intelligent Analysis API Routes
Defines POST /analyze, POST /analyze/intelligent, POST /analyze/ask, and POST /analyze/full endpoints.
"""

from fastapi import APIRouter, status, HTTPException
from ml.ai.schemas import (
    JobAnalysisRequest, JobAnalysisResponse,
    IntelligentAnalysisResponse, JobQARequest, JobQAResponse, FullAnalysisResponse
)
from ml.ai.information_extractor import extract_job_information
from ml.ai.job_analyzer import perform_intelligent_analysis
from ml.ai.job_qa import answer_job_question
from backend.services.prediction_service import PredictionService
from backend.schemas import JobPostingRequest

router = APIRouter(tags=["AI Job Information Extraction & Analysis"])


@router.post(
    "/analyze",
    response_model=JobAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="AI/NLP Job Information Extraction",
    description="Accepts a job posting and returns a structured representation of extracted fields, missing information analysis, and extraction metadata.",
)
async def analyze_job_posting(payload: JobAnalysisRequest):
    """
    POST /analyze
    Executes local NLP information extraction across job posting text blocks.
    """
    try:
        extracted_info = extract_job_information(
            title=payload.title,
            description=payload.description,
            company_profile=payload.company_profile,
            requirements=payload.requirements,
            benefits=payload.benefits,
            salary_range=payload.salary_range,
            location=payload.location,
            department=payload.department,
            employment_type=payload.employment_type,
            required_experience=payload.required_experience,
            required_education=payload.required_education,
            industry=payload.industry,
            function=payload.function
        )

        metadata = {
            "extractor_version": "1.0.0",
            "extraction_layer": "Local NLP Layer (Regex + Pattern + Context Skill Dictionary)",
            "external_api_used": False,
            "disclaimer": "This module provides NLP-based information extraction and does not replace the machine-learning fraud classification model."
        }

        return JobAnalysisResponse(
            success=True,
            extracted_information=extracted_info,
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during information extraction: {str(e)}"
        )


@router.post(
    "/analyze/intelligent",
    response_model=IntelligentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Intelligent Job Summary & Warning Signal Analysis",
    description="Generates a structured job summary, skills breakdown, categorized requirements, and rule-based warning signals.",
)
async def intelligent_job_analysis(payload: JobAnalysisRequest):
    """
    POST /analyze/intelligent
    Returns structured summary, skills breakdown, requirements categorization, and warning signals.
    """
    try:
        return perform_intelligent_analysis(payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during intelligent analysis: {str(e)}"
        )


@router.post(
    "/analyze/ask",
    response_model=JobQAResponse,
    status_code=status.HTTP_200_OK,
    summary="Local Job-Posting Q&A",
    description="Answers natural language questions about a job posting using deterministic local intent matching.",
)
async def ask_job_question(payload: JobQARequest):
    """
    POST /analyze/ask
    Answers user questions using local NLP extraction results without external cloud AI APIs.
    """
    try:
        return answer_job_question(payload.job_posting, payload.question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during question answering: {str(e)}"
        )


@router.post(
    "/analyze/full",
    response_model=FullAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Combined ML Risk Screening & Intelligent NLP Analysis",
    description="Executes ML Logistic Regression risk classification, local NLP extraction, and intelligent job analysis in a single request.",
)
async def full_job_analysis(payload: JobAnalysisRequest):
    """
    POST /analyze/full
    Combines ML prediction (/predict), local NLP extraction (/analyze), and intelligent analysis (/analyze/intelligent).
    """
    try:
        # 1. Convert payload to JobPostingRequest for ML Classifier
        title_val = payload.title or "Unknown Title"
        desc_val = payload.description or "No description provided."
        pred_req = JobPostingRequest(
            title=title_val,
            description=desc_val,
            company_profile=payload.company_profile or "",
            requirements=payload.requirements or "",
            benefits=payload.benefits or "",
            location=payload.location or "Unknown",
            department=payload.department or "Unknown",
            employment_type=payload.employment_type or "Unknown",
            required_experience=payload.required_experience or "Unknown",
            required_education=payload.required_education or "Unknown",
            industry=payload.industry or "Unknown",
            function=payload.function or "Unknown",
            salary_range=payload.salary_range
        )

        ml_res = PredictionService.predict(pred_req)

        # 2. Local NLP Extraction
        extracted_info = extract_job_information(
            title=payload.title,
            description=payload.description,
            company_profile=payload.company_profile,
            requirements=payload.requirements,
            benefits=payload.benefits,
            salary_range=payload.salary_range,
            location=payload.location,
            department=payload.department,
            employment_type=payload.employment_type,
            required_experience=payload.required_experience,
            required_education=payload.required_education,
            industry=payload.industry,
            function=payload.function
        )

        # 3. Intelligent Job Analysis
        intel_res = perform_intelligent_analysis(payload)

        # 4. Construct combined output
        feature_signals = [s.model_dump() for s in ml_res.signals] if ml_res.signals else []

        return FullAnalysisResponse(
            success=True,
            prediction=ml_res.prediction,
            risk_score=ml_res.risk_score,
            risk_percentage=ml_res.risk_percentage,
            risk_level=ml_res.risk_level,
            ml_risk_score=ml_res.ml_risk_score,
            ml_risk_percentage=ml_res.ml_risk_percentage,
            final_risk_score=ml_res.final_risk_score,
            final_risk_percentage=ml_res.final_risk_percentage,
            trusthire_assessment=ml_res.trusthire_assessment,
            nlp_risk_penalty=ml_res.nlp_risk_penalty,
            model=ml_res.model,
            feature_signals=feature_signals,
            warning_signals=intel_res.warning_signals,
            extracted_information=extracted_info,
            intelligent_analysis=intel_res,
            disclaimer=ml_res.disclaimer
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during full job analysis: {str(e)}"
        )
