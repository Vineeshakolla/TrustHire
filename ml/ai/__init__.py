"""
TrustHire AI Module - Local Job Information Extraction, Intelligent Analysis & Q&A
"""

from ml.ai.schemas import (
    ExtractedJobInformation, SalaryInformation, JobAnalysisRequest, JobAnalysisResponse,
    JobSummary, StructuredRequirements, SkillsBreakdown, WarningSignal,
    IntelligentAnalysisResponse, JobQARequest, JobQAResponse, FullAnalysisResponse
)
from ml.ai.information_extractor import extract_job_information
from ml.ai.job_analyzer import (
    generate_job_summary, extract_key_skills, structure_requirements,
    detect_warning_signals, perform_intelligent_analysis
)
from ml.ai.job_qa import answer_job_question

__all__ = [
    "ExtractedJobInformation",
    "SalaryInformation",
    "JobAnalysisRequest",
    "JobAnalysisResponse",
    "JobSummary",
    "StructuredRequirements",
    "SkillsBreakdown",
    "WarningSignal",
    "IntelligentAnalysisResponse",
    "JobQARequest",
    "JobQAResponse",
    "FullAnalysisResponse",
    "extract_job_information",
    "generate_job_summary",
    "extract_key_skills",
    "structure_requirements",
    "detect_warning_signals",
    "perform_intelligent_analysis",
    "answer_job_question"
]
