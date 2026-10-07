"""
TrustHire - FastAPI Pydantic Request & Response Schemas
Defines request models, validation logic, response structures, and health response models.
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


class JobPostingRequest(BaseModel):
    """
    Request model for single job posting screening.
    Supports all standard EMSCAD text, categorical, and binary attributes.
    """

    title: str = Field(..., description="Job posting title (Required)")
    description: str = Field(..., description="Full job posting description (Required)")
    company_profile: Optional[str] = Field(default="", description="Company profile / overview")
    requirements: Optional[str] = Field(default="", description="Job requirements & qualifications")
    benefits: Optional[str] = Field(default="", description="Benefits & perks offered")
    location: Optional[str] = Field(default="Unknown", description="Job location (City, State, Country)")
    department: Optional[str] = Field(default="Unknown", description="Hiring department")
    salary_range: Optional[str] = Field(default=None, description="Salary range (e.g. '50000-70000')")
    employment_type: Optional[str] = Field(default="Unknown", description="Full-time, Part-time, Contract, etc.")
    required_experience: Optional[str] = Field(default="Unknown", description="Entry level, Mid-Senior, etc.")
    required_education: Optional[str] = Field(default="Unknown", description="Bachelor's, Master's, High School, etc.")
    industry: Optional[str] = Field(default="Unknown", description="Industry classification")
    function: Optional[str] = Field(default="Unknown", description="Job function / role family")
    telecommuting: Union[bool, str, int] = Field(default=False, description="Remote work indicator")
    has_company_logo: Union[bool, str, int] = Field(default=False, description="Company logo present indicator")
    has_questions: Union[bool, str, int] = Field(default=False, description="Screening questions present indicator")
    threshold: Optional[float] = Field(default=0.50, description="Decision threshold (Default: 0.50)")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Job title cannot be empty.")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Job description cannot be empty.")
        return v.strip()


class SignalItem(BaseModel):
    """
    Model signal item providing feature importance explanation.
    """

    feature: str = Field(..., description="Feature or signal name")
    direction: str = Field(..., description="Signal direction: 'risk_associated' or 'legitimate_associated'")
    importance: float = Field(..., description="Model coefficient or importance weight")


class PredictionResponse(BaseModel):
    """
    API Response model for POST /predict.
    """

    success: bool = Field(default=True, description="Request execution status")
    prediction: str = Field(..., description="Classification result: 'Legitimate' or 'Potentially Fraudulent'")
    risk_score: float = Field(..., description="Calibrated fused risk score (0.0 to 1.0)")
    risk_percentage: float = Field(..., description="Risk score percentage (0.0% to 100.0%)")
    risk_level: str = Field(..., description="Risk level category (Low Risk, Moderate Risk, High Risk)")
    ml_risk_score: Optional[float] = Field(default=None, description="Raw Logistic Regression probability (0.0 to 1.0)")
    ml_risk_percentage: Optional[float] = Field(default=None, description="Raw ML model percentage")
    final_risk_score: Optional[float] = Field(default=None, description="Fused final TrustHire risk score")
    final_risk_percentage: Optional[float] = Field(default=None, description="Fused final percentage")
    trusthire_assessment: Optional[str] = Field(default=None, description="TrustHire human-readable assessment")
    nlp_risk_penalty: Optional[float] = Field(default=0.0, description="Sum of active NLP warning signal weights")
    threshold: float = Field(default=0.50, description="Screening threshold applied")
    model: str = Field(default="Logistic Regression", description="Model name used for prediction")
    signals: List[SignalItem] = Field(default_factory=list, description="Top positive and negative feature signals")
    warning_signals: List[Dict[str, Any]] = Field(default_factory=list, description="Detected NLP warning signals")
    disclaimer: str = Field(
        default="This is a machine-learning screening signal and should not be treated as proof that a job or company is fraudulent.",
        description="User safety disclaimer",
    )


class HealthResponse(BaseModel):
    """
    API Response model for GET /health.
    """

    status: str = Field(default="ok")
    service: str = Field(default="TrustHire API")
    model_loaded: bool = Field(default=True)
