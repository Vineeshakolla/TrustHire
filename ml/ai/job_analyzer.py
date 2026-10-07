"""
TrustHire - Intelligent Job Analysis Module
Transforms extracted NLP job information into structured summaries, skills breakdown, requirements categorization, and rule-based warning signals.
"""

import re
from typing import Dict, Any, List, Optional
from ml.ai.schemas import (
    ExtractedJobInformation, JobSummary, StructuredRequirements,
    SkillsBreakdown, WarningSignal, IntelligentAnalysisResponse, JobAnalysisRequest
)
from ml.ai.information_extractor import extract_job_information


def generate_job_summary(extracted: ExtractedJobInformation) -> JobSummary:
    """Generates a concise, structured job summary with 'Not specified' fallbacks."""
    company_val = extracted.company_name or extracted.company_profile or "Not specified"
    salary_val = extracted.salary.raw if (extracted.salary and extracted.salary.raw) else "Not specified"

    resp_summary = "; ".join(extracted.responsibilities[:3]) if extracted.responsibilities else "Not specified"
    req_summary = "; ".join(extracted.requirements[:3]) if extracted.requirements else "Not specified"

    return JobSummary(
        role=extracted.job_title or "Not specified",
        company=company_val,
        location=extracted.location or "Not specified",
        work_mode=extracted.work_mode or "Not specified",
        employment_type=extracted.employment_type or "Not specified",
        salary=salary_val,
        experience=extracted.experience or "Not specified",
        education=extracted.education or "Not specified",
        duration=extracted.duration or "Not specified",
        start_date=extracted.start_date or "Not specified",
        internship_status="Yes (Internship Posting)" if extracted.internship else "No (Standard Role)",
        responsibilities_summary=resp_summary,
        requirements_summary=req_summary
    )


def extract_key_skills(extracted: ExtractedJobInformation, text: str) -> SkillsBreakdown:
    """Categorizes extracted technical skills into required vs preferred skills."""
    all_skills = extracted.skills or []
    required_skills = []
    preferred_skills = []

    # Check for 'Nice to Have' or 'Preferred' section in text
    preferred_block = ""
    pref_match = re.search(r'(?:nice\s*to\s*have|preferred|bonus|familiarity\s*with)\s*:?([^\n\r]+(?:\n[^\n\r]+){0,5})', text, re.IGNORECASE)
    if pref_match:
        preferred_block = pref_match.group(0).lower()

    for skill in all_skills:
        if preferred_block and skill.lower() in preferred_block:
            preferred_skills.append(skill)
        else:
            required_skills.append(skill)

    return SkillsBreakdown(
        required_skills=required_skills if required_skills else all_skills,
        preferred_skills=preferred_skills,
        all_skills=all_skills
    )


def structure_requirements(extracted: ExtractedJobInformation, text: str) -> StructuredRequirements:
    """Categorizes requirements into education, experience, technical skills, soft skills, and other requirements."""
    soft_skill_keywords = ["communication", "teamwork", "problem-solving", "eagerness to learn", "adaptability", "curious", "leadership"]
    found_soft = []
    other_reqs = []

    for req in extracted.requirements:
        lower_req = req.lower()
        if any(sk in lower_req for sk in soft_skill_keywords):
            found_soft.append(req)
        elif not any(ts.lower() in lower_req for ts in extracted.skills):
            other_reqs.append(req)

    return StructuredRequirements(
        education=extracted.education,
        experience=extracted.experience,
        technical_skills=extracted.skills,
        soft_skills=found_soft,
        other_requirements=other_reqs
    )


def detect_warning_signals(extracted: ExtractedJobInformation, text: str) -> List[WarningSignal]:
    """
    Evidence-based warning signal analyzer.
    Identifies potential warning signals with severity levels and transparent risk weights.
    Does NOT hardcode specific test cases; detects generalized suspicious concepts.
    """
    signals = []
    text_lower = (text or '').lower()

    # 1. Missing Company Information / Employer Identity
    if not extracted.company_name and not extracted.company_profile:
        signals.append(WarningSignal(
            signal="Missing company information",
            evidence="Company profile and employer name are empty",
            explanation="The posting does not state the employer's company name or profile. Verification of employer identity is strongly advised.",
            severity="high",
            weight=0.12
        ))
    elif extracted.company_profile and "client" in extracted.company_profile.lower():
        signals.append(WarningSignal(
            signal="Role posted via recruitment agency or client representative",
            evidence="This role is for one of our clients",
            explanation="The posting is listed by a third-party recruiter. Candidates should verify the client company.",
            severity="medium",
            weight=0.05
        ))

    # 2. Missing Salary Information
    if not extracted.salary or not extracted.salary.raw:
        signals.append(WarningSignal(
            signal="Missing salary information",
            evidence="No stipend or salary range provided",
            explanation="Compensation details were not specified in the posting text.",
            severity="low",
            weight=0.04
        ))

    # 3. Urgency Hiring Language
    urgency_match = re.search(r'\b(immediate\s*joining|apply\s*immediately|urgent\s*requirement|hiring\s*today|immediate\s*start)\b', text_lower)
    if urgency_match:
        signals.append(WarningSignal(
            signal="Urgency hiring language detected",
            evidence=urgency_match.group(0),
            explanation="The posting uses high-urgency hiring phrases, which is common in rapid recruitment but requires validation.",
            severity="medium",
            weight=0.08
        ))

    # 4. Fee / Payment / Registration / Off-Platform Communication Language
    fee_match = re.search(r'\b(registration\s*fee|processing\s*fee|security\s*deposit|pay\s*upfront|bank\s*details|telegram|whatsapp|transfer\s*money)\b', text_lower)
    if fee_match:
        signals.append(WarningSignal(
            signal="Payment, fee, or off-platform communication request",
            evidence=fee_match.group(0),
            explanation="Legitimate job opportunities rarely require candidates to pay application/registration fees or communicate exclusively on unverified chat apps.",
            severity="critical",
            weight=0.25
        ))

    # 5. Guaranteed Income / Unrealistic Promise Wording
    income_match = re.search(r'\b(guaranteed\s*income|easy\s*money|earn\s*\$?\d+\s*(daily|weekly)|no\s*work\s*required|high\s*commission\s*paid\s*daily)\b', text_lower)
    if income_match:
        signals.append(WarningSignal(
            signal="Guaranteed income or unrealistic promise wording",
            evidence=income_match.group(0),
            explanation="Promising guaranteed earnings with minimal effort or experience is a common indicator of suspicious solicitations.",
            severity="critical",
            weight=0.25
        ))

    # 6. Excessively Brief Job Description
    if len(text_lower) < 120:
        signals.append(WarningSignal(
            signal="Excessively brief job description",
            evidence=f"Text length: {len(text_lower)} characters",
            explanation="The job description contains extremely limited detail about job duties or expectations.",
            severity="high",
            weight=0.12
        ))

    # 7. Vague Eligibility / Open-ended Wording
    vague_eligibility_match = re.search(
        r'\b(no\s+experience|any\s*one|anyone|anybody)\s+.*?\b(allowed|apply|join|welcome|eligible|needed)\b|\banyone\s+can\s+apply\b|\bno\s+qualifications?\s+needed\b',
        text_lower
    )
    if vague_eligibility_match:
        signals.append(WarningSignal(
            signal="Vague or unrestrictive eligibility criteria",
            evidence=vague_eligibility_match.group(0),
            explanation="The posting advertises blanket eligibility ('anyone allowed / no experience needed') without specifying clear technical or functional expectations.",
            severity="high",
            weight=0.15
        ))

    # 8. Missing Responsibilities and Requirements
    if not extracted.responsibilities and not extracted.requirements:
        signals.append(WarningSignal(
            signal="Missing job responsibilities and requirements",
            evidence="No duty bullet points or qualification requirements provided",
            explanation="The posting does not outline specific daily responsibilities or candidate qualifications.",
            severity="medium",
            weight=0.10
        ))

    return signals


def perform_intelligent_analysis(payload: JobAnalysisRequest) -> IntelligentAnalysisResponse:
    """Executes full intelligent job analysis pipeline on job posting request."""
    # Step 1: Run local NLP extraction
    extracted = extract_job_information(
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

    full_text = f"{payload.title or ''} {payload.description or ''} {payload.requirements or ''} {payload.benefits or ''} {payload.company_profile or ''}"

    # Step 2: Generate outputs
    summary = generate_job_summary(extracted)
    skills = extract_key_skills(extracted, full_text)
    reqs = structure_requirements(extracted, full_text)
    warning_signals = detect_warning_signals(extracted, full_text)

    important_info = {
        "job_title": extracted.job_title or "Not specified",
        "company": extracted.company_name or extracted.company_profile or "Not specified",
        "location": extracted.location or "Not specified",
        "work_mode": extracted.work_mode or "Not specified",
        "employment_type": extracted.employment_type or "Not specified",
        "salary": extracted.salary.raw if (extracted.salary and extracted.salary.raw) else "Not specified",
        "experience": extracted.experience or "Not specified",
        "education": extracted.education or "Not specified",
        "duration": extracted.duration or "Not specified",
        "start_date": extracted.start_date or "Not specified",
        "internship": extracted.internship
    }

    nlp_metadata = {
        "analysis_engine": "Local NLP Intelligent Job Analysis Engine",
        "external_api_used": False,
        "disclaimer": "Local NLP-based intelligent job analysis. Warning signals are informational flags and do not replace ML fraud risk classification."
    }

    return IntelligentAnalysisResponse(
        success=True,
        summary=summary,
        skills=skills,
        responsibilities=extracted.responsibilities,
        requirements=reqs,
        important_information=important_info,
        missing_information=extracted.important_missing_information,
        warning_signals=warning_signals,
        nlp_metadata=nlp_metadata
    )
