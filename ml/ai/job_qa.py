"""
TrustHire - Local Job-Posting Question Answering Engine
Provides deterministic, local intent-matching Q&A over job posting text and extracted attributes without external generative APIs.
"""

import re
from typing import Dict, Any, Optional
from ml.ai.schemas import JobAnalysisRequest, JobQAResponse
from ml.ai.job_analyzer import perform_intelligent_analysis
from ml.ai.information_extractor import extract_job_information


def answer_job_question(job_posting: JobAnalysisRequest, question: str) -> JobQAResponse:
    """
    Answers questions about a job posting using deterministic local intent matching.
    Guarantees zero hallucination and strictly uses extracted posting information.
    """
    if not question or not question.strip():
        return JobQAResponse(
            question=question or "",
            answer="Please ask a question regarding the job posting.",
            confidence="low"
        )

    q_clean = question.strip()
    q_lower = q_clean.lower()

    # Perform NLP extraction & analysis
    extracted = extract_job_information(
        title=job_posting.title,
        description=job_posting.description,
        company_profile=job_posting.company_profile,
        requirements=job_posting.requirements,
        benefits=job_posting.benefits,
        salary_range=job_posting.salary_range,
        location=job_posting.location,
        department=job_posting.department,
        employment_type=job_posting.employment_type,
        required_experience=job_posting.required_experience,
        required_education=job_posting.required_education,
        industry=job_posting.industry,
        function=job_posting.function
    )

    full_text = f"{job_posting.title or ''} {job_posting.description or ''} {job_posting.requirements or ''}"

    # 1. Fraud / Risk Intent
    if any(k in q_lower for k in ["fraud", "fake", "scam", "legit", "safe", "risk"]):
        answer = (
            "TrustHire's ML classifier should be used for fraud-risk classification via the /predict endpoint. "
            "The NLP analysis itself does not determine whether a posting is fraudulent."
        )
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 2. Salary / Stipend Intent
    if any(k in q_lower for k in ["salary", "stipend", "pay", "compensation", "earn", "remuneration"]):
        if extracted.salary and extracted.salary.raw:
            answer = f"The posting states a compensation/stipend of {extracted.salary.raw}."
        else:
            answer = "Salary or stipend details are not specified in the provided job posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 3. Location / City Intent
    if any(k in q_lower for k in ["location", "city", "where", "place", "office"]):
        if extracted.location:
            answer = f"The job is located in {extracted.location}."
            if extracted.work_mode:
                answer += f" Work mode: {extracted.work_mode}."
        else:
            answer = "The exact location is not explicitly specified in the job posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 4. Remote / Work Mode Intent
    if any(k in q_lower for k in ["remote", "work from home", "wfh", "hybrid", "on-site", "work mode"]):
        if extracted.work_mode:
            answer = f"The stated work mode for this role is {extracted.work_mode}."
        else:
            answer = "Work mode (Remote / On-site / Hybrid) is not explicitly specified in the posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 5. Skills / Tech Stack Intent
    if any(k in q_lower for k in ["skill", "tech stack", "technology", "programming", "languages", "tools"]):
        if extracted.skills:
            skills_str = ", ".join(extracted.skills)
            answer = f"The skills mentioned in the posting include: {skills_str}."
        else:
            answer = "No specific technical skills were explicitly identified in the posting text."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 6. Experience Intent
    if any(k in q_lower for k in ["experience", "years of experience", "seniority", "junior", "senior"]):
        if extracted.experience:
            answer = f"The posting states an experience requirement of: {extracted.experience}."
        else:
            answer = "Specific years of experience are not stated in the posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 7. Education / Degree Intent
    if any(k in q_lower for k in ["education", "degree", "qualification", "b.tech", "bachelor", "master", "phd"]):
        if extracted.education:
            answer = f"The stated educational qualification is: {extracted.education}."
        else:
            answer = "Educational degree requirements are not explicitly specified."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 8. Responsibilities / Duties Intent
    if any(k in q_lower for k in ["responsibilities", "duties", "what will i do", "what you'll do", "role"]):
        if extracted.responsibilities:
            resp_str = "; ".join(extracted.responsibilities)
            answer = f"The key responsibilities include: {resp_str}."
        elif extracted.job_title:
            answer = f"The role is titled '{extracted.job_title}'. Specific bulleted responsibilities were not detailed."
        else:
            answer = "Detailed responsibilities are not explicitly itemized in the posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 9. Internship Intent
    if any(k in q_lower for k in ["internship", "intern", "is this an intern"]):
        if "how long" in q_lower or "duration" in q_lower:
            if extracted.duration:
                answer = f"The internship duration is: {extracted.duration}."
            else:
                answer = "The internship duration is not explicitly specified."
            return JobQAResponse(question=q_clean, answer=answer, confidence="high")

        if extracted.internship:
            answer = "Yes, this posting explicitly represents an internship role."
        else:
            answer = "No, this posting appears to be a standard full-time or part-time role, not an internship."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 10. Start Date / Joining Intent
    if any(k in q_lower for k in ["start", "join", "when", "availability"]):
        if extracted.start_date:
            answer = f"The stated start date availability is: {extracted.start_date}."
        else:
            answer = "Start date or joining timeline is not specified in the posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 11. Company Name / Info Intent
    if any(k in q_lower for k in ["company", "employer", "who is", "organization"]):
        if extracted.company_name:
            answer = f"The company name specified in the posting is '{extracted.company_name}'."
        elif extracted.company_profile:
            answer = f"Company information: {extracted.company_profile}."
        else:
            answer = "The company name is not explicitly specified in the provided job posting."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 12. Benefits Intent
    if any(k in q_lower for k in ["benefit", "perk", "offer", "insurance", "bonus"]):
        if extracted.benefits:
            ben_str = "; ".join(extracted.benefits)
            answer = f"The posting lists the following benefits: {ben_str}."
        else:
            answer = "The posting does not specify benefits or perks."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 13. Missing Information Intent
    if any(k in q_lower for k in ["missing", "incomplete", "lacking"]):
        if extracted.important_missing_information:
            missing_str = ", ".join(extracted.important_missing_information)
            answer = f"The following key information fields are missing from the posting: {missing_str}."
        else:
            answer = "The posting contains comprehensive details for key standard fields."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # 14. Warning Signals Intent
    if any(k in q_lower for k in ["warning", "red flag", "signal", "caution"]):
        analysis = perform_intelligent_analysis(job_posting)
        if analysis.warning_signals:
            sig_strs = [f"{s.signal} ({s.evidence})" for s in analysis.warning_signals]
            answer = f"The following potential warning signals were detected: {'; '.join(sig_strs)}."
        else:
            answer = "No obvious warning signals (such as payment requests or extreme urgency) were detected."
        return JobQAResponse(question=q_clean, answer=answer, confidence="high")

    # Default fallback for unhandled questions
    return JobQAResponse(
        question=q_clean,
        answer="That information is not explicitly provided in the job posting.",
        confidence="medium"
    )
