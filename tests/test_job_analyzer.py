"""
TrustHire - Intelligent Job Analyzer Test Suite
Tests summary generation, skills breakdown, requirements categorization, warning signal detection, and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml.ai.schemas import JobAnalysisRequest
from ml.ai.job_analyzer import (
    generate_job_summary, extract_key_skills, structure_requirements,
    detect_warning_signals, perform_intelligent_analysis
)

client = TestClient(app)

REAL_WORLD_JOB_POSTING = """This role is for one of our clients
Industry: Software Development
Seniority level: Junior - Entry Level
Min Experience: 0 years
Location: Bengaluru (Remote)
JobType: full-time

About The Role

We are looking for a motivated and curious Software Engineering Intern to join the team. This is a hands-on role where you'll work alongside experienced engineers, contribute to real projects, and grow your technical skills in a fast-paced environment.

Stipend: ₹30,000 – ₹50,000 per month

Duration: Minimum 2 months, extendable up to 6 months

Start Date: Immediate / As soon as possible

Requirements:
Strong problem-solving skills and eagerness to learn
Good communication and teamwork skills
Currently pursuing or recently completed a degree in Computer Science, Engineering, or a related field
Basic understanding of data structures, algorithms, and software development principles
Proficiency in at least one programming language (e.g., Python, JavaScript, Java, C++)

Nice to Have:
Familiarity with web frameworks, cloud platforms, or AI/ML tools
Prior internship experience or personal projects on GitHub
Exposure to agile development practices"""


def test_1_summary_generation():
    """Test 1: Verify summary generation extracts role, location, salary, experience, education, duration, and start date."""
    req = JobAnalysisRequest(description=REAL_WORLD_JOB_POSTING)
    res = perform_intelligent_analysis(req)

    assert res.summary.role == "Software Engineering Intern"
    assert res.summary.location == "Bengaluru"
    assert res.summary.work_mode == "Remote"
    assert res.summary.employment_type == "Full-time"
    assert "₹30,000" in res.summary.salary
    assert res.summary.experience == "0 years"
    assert "Computer Science" in res.summary.education
    assert res.summary.duration == "Minimum 2 months, extendable up to 6 months"
    assert "Immediate" in res.summary.start_date
    assert res.summary.internship_status == "Yes (Internship Posting)"


def test_2_skills_breakdown():
    """Test 2: Verify required vs preferred technical skills breakdown."""
    req = JobAnalysisRequest(description=REAL_WORLD_JOB_POSTING)
    res = perform_intelligent_analysis(req)

    assert "Python" in res.skills.all_skills
    assert "JavaScript" in res.skills.all_skills
    assert "Java" in res.skills.all_skills
    assert "C++" in res.skills.all_skills


def test_3_requirements_categorization():
    """Test 3: Verify requirements categorized into education, experience, technical, soft skills, and other."""
    req = JobAnalysisRequest(description=REAL_WORLD_JOB_POSTING)
    res = perform_intelligent_analysis(req)

    assert res.requirements.education is not None
    assert res.requirements.experience == "0 years"
    assert "Python" in res.requirements.technical_skills
    assert len(res.requirements.soft_skills) > 0


def test_4_warning_signal_detection():
    """Test 4: Verify rule-based warning signal detection (agency notice, missing company)."""
    req = JobAnalysisRequest(description=REAL_WORLD_JOB_POSTING)
    res = perform_intelligent_analysis(req)

    signal_titles = [w.signal for w in res.warning_signals]
    assert any("client" in s.lower() or "missing" in s.lower() for s in signal_titles)


def test_5_urgency_fee_warning_signals():
    """Test 5: Verify urgency and fee payment warning signals."""
    suspicious_text = "Urgent requirement! Apply immediately! Send registration fee of $50 via Telegram for guaranteed income."
    req = JobAnalysisRequest(title="Data Entry", description=suspicious_text)
    res = perform_intelligent_analysis(req)

    signals = [w.signal for w in res.warning_signals]
    assert any("urgency" in s.lower() for s in signals)
    assert any("payment" in s.lower() or "fee" in s.lower() for s in signals)
    assert any("guaranteed income" in s.lower() for s in signals)


def test_6_no_hallucinated_information():
    """Test 6: Verify empty fields fallback to 'Not specified' rather than hallucinating details."""
    req = JobAnalysisRequest(title="Minimalist Posting", description="Just basic title.")
    res = perform_intelligent_analysis(req)

    assert res.summary.salary == "Not specified"
    assert res.summary.location == "Not specified"
    assert res.summary.experience == "Not specified"


def test_7_post_analyze_intelligent_endpoint():
    """Test 7: Verify POST /analyze/intelligent API endpoint."""
    payload = {
        "title": "Software Engineering Intern",
        "description": REAL_WORLD_JOB_POSTING
    }
    response = client.post("/analyze/intelligent", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["summary"]["role"] == "Software Engineering Intern"
    assert data["summary"]["location"] == "Bengaluru"


def test_8_post_analyze_full_endpoint():
    """Test 8: Verify POST /analyze/full endpoint returning combined ML + NLP results."""
    payload = {
        "title": "Software Engineering Intern",
        "description": REAL_WORLD_JOB_POSTING,
        "company_profile": "InnovateTech Labs startup",
        "location": "Bengaluru"
    }
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    assert "risk_score" in data
    assert "extracted_information" in data
    assert "intelligent_analysis" in data
