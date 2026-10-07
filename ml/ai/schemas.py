"""
TrustHire - AI Job Information Extraction & Intelligent Analysis Schemas
Defines structured data schemas for extracted job attributes, intelligent analysis, Q&A, and full pipeline endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SalaryInformation(BaseModel):
    raw: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    currency: Optional[str] = None
    period: Optional[str] = None


class ExtractedJobInformation(BaseModel):
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    company_profile: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None  # "Remote", "On-site", "Hybrid"
    employment_type: Optional[str] = None  # "Full-time", "Part-time", "Contract", "Temporary", "Internship", etc.
    salary: Optional[SalaryInformation] = None
    experience: Optional[str] = None
    education: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    start_date: Optional[str] = None
    industry: Optional[str] = None
    department: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    internship: bool = False
    important_missing_information: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, str] = Field(default_factory=dict)


class JobAnalysisRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    company_profile: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    salary_range: Optional[str] = None
    location: Optional[str] = None
    department: Optional[str] = None
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    function: Optional[str] = None


class JobAnalysisResponse(BaseModel):
    success: bool = True
    extracted_information: ExtractedJobInformation
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WarningSignal(BaseModel):
    signal: str
    evidence: Optional[str] = None
    explanation: str
    severity: Optional[str] = "medium"
    weight: Optional[float] = 0.10


class JobSummary(BaseModel):
    role: str
    company: str
    location: str
    work_mode: str
    employment_type: str
    salary: str
    experience: str
    education: str
    duration: str
    start_date: str
    internship_status: str
    responsibilities_summary: str
    requirements_summary: str


class StructuredRequirements(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    other_requirements: List[str] = Field(default_factory=list)


class SkillsBreakdown(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    all_skills: List[str] = Field(default_factory=list)


class IntelligentAnalysisResponse(BaseModel):
    success: bool = True
    summary: JobSummary
    skills: SkillsBreakdown
    responsibilities: List[str] = Field(default_factory=list)
    requirements: StructuredRequirements
    important_information: Dict[str, Any] = Field(default_factory=dict)
    missing_information: List[str] = Field(default_factory=list)
    warning_signals: List[WarningSignal] = Field(default_factory=list)
    nlp_metadata: Dict[str, Any] = Field(default_factory=dict)


class JobQARequest(BaseModel):
    job_posting: JobAnalysisRequest
    question: str


class JobQAResponse(BaseModel):
    success: bool = True
    question: str
    answer: str
    source: str = "Submitted job posting"
    confidence: str = "high"


class FullAnalysisResponse(BaseModel):
    success: bool = True
    prediction: str
    risk_score: float
    risk_percentage: float
    risk_level: str
    ml_risk_score: Optional[float] = None
    ml_risk_percentage: Optional[float] = None
    final_risk_score: Optional[float] = None
    final_risk_percentage: Optional[float] = None
    trusthire_assessment: Optional[str] = None
    nlp_risk_penalty: Optional[float] = None
    model: str
    feature_signals: List[Dict[str, Any]] = Field(default_factory=list)
    warning_signals: List[WarningSignal] = Field(default_factory=list)
    extracted_information: ExtractedJobInformation
    intelligent_analysis: IntelligentAnalysisResponse
    disclaimer: str
