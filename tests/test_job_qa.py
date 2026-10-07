"""
TrustHire - Local Job Q&A Test Suite
Tests deterministic local intent matching for 15+ job posting question types.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml.ai.schemas import JobAnalysisRequest
from ml.ai.job_qa import answer_job_question

client = TestClient(app)

SAMPLE_JOB_POSTING = JobAnalysisRequest(
    title="Software Engineering Intern",
    description="""This role is for one of our clients
Location: Bengaluru (Remote)
JobType: full-time
Min Experience: 0 years
Stipend: ₹30,000 – ₹50,000 per month
Duration: Minimum 2 months, extendable up to 6 months
Start Date: Immediate / As soon as possible
Requirements:
Proficiency in Python, JavaScript, Java, C++, Data Structures, Algorithms.
Degree in Computer Science or Engineering.""",
    company_profile="This role is for one of our clients",
    benefits="Health insurance, learning stipend"
)


def test_1_salary_question():
    """Test 1: What is the salary?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "What is the salary?")
    assert res.success is True
    assert "₹30,000" in res.answer


def test_2_location_question():
    """Test 2: Where is the job located?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "Where is the job located?")
    assert "Bengaluru" in res.answer


def test_3_remote_question():
    """Test 3: Is it remote?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "Is it remote?")
    assert "Remote" in res.answer


def test_4_skills_question():
    """Test 4: What skills are required?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "What skills are required?")
    assert "Python" in res.answer
    assert "Java" in res.answer


def test_5_experience_question():
    """Test 5: What experience is required?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "What experience is required?")
    assert "0 years" in res.answer


def test_6_education_question():
    """Test 6: What degree is required?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "What degree is required?")
    assert "Computer Science" in res.answer or "Engineering" in res.answer


def test_7_internship_questions():
    """Test 7: Is this an internship? How long is the internship?"""
    res1 = answer_job_question(SAMPLE_JOB_POSTING, "Is this an internship?")
    assert "Yes" in res1.answer

    res2 = answer_job_question(SAMPLE_JOB_POSTING, "How long is the internship?")
    assert "2 months" in res2.answer


def test_8_start_date_question():
    """Test 8: When can I join?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "When can I join?")
    assert "Immediate" in res.answer


def test_9_company_question():
    """Test 9: Who is the company?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "Who is the company?")
    assert "client" in res.answer.lower() or "not explicitly" in res.answer.lower()


def test_10_fraud_question():
    """Test 10: Is this fraudulent?"""
    res = answer_job_question(SAMPLE_JOB_POSTING, "Is this job fraudulent?")
    assert "predict" in res.answer.lower() or "ml classifier" in res.answer.lower()


def test_11_unknown_question_handling():
    """Test 11: Unknown / unhandled question fallback."""
    res = answer_job_question(SAMPLE_JOB_POSTING, "What is the dress code for Friday?")
    assert "not explicitly provided" in res.answer


def test_12_post_analyze_ask_api_endpoint():
    """Test 12: Verify POST /analyze/ask API endpoint."""
    payload = {
        "job_posting": {
            "title": "Software Engineering Intern",
            "description": "Location: Bengaluru (Remote). Stipend: ₹30,000 per month."
        },
        "question": "What is the salary?"
    }
    response = client.post("/analyze/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "30,000" in data["answer"]
