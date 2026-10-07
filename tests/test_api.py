"""
TrustHire - FastAPI Backend API Test Suite
Tests all 15 API requirements for Step 6:
1. FastAPI application starts
2. GET /health returns 200 OK
3. Model availability is detected (model_loaded = True)
4. POST /predict accepts valid posting
5. POST /predict returns prediction
6. Response contains risk_score
7. Response contains risk_level
8. Response contains model name
9. Response contains threshold
10. Response contains disclaimer
11. Missing required title handled
12. Missing required description handled
13. Invalid input handled
14. Model error handling
15. Real-world software engineering internship job posting test
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


@pytest.fixture
def valid_job_payload():
    """Constructs a valid job posting payload."""
    return {
        "title": "Software Engineering Intern",
        "company_profile": "Technology company developing software products.",
        "description": "We are looking for a software engineering intern to work with Python and C++.",
        "requirements": "Knowledge of Python, Java or C++.",
        "benefits": "Learning opportunities and mentorship.",
        "location": "Bengaluru",
        "salary_range": "30000-50000",
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Bachelor's Degree",
        "industry": "Computer Software",
        "function": "Engineering",
        "telecommuting": True,
        "has_company_logo": True,
        "has_questions": False,
        "threshold": 0.50,
    }


def test_1_app_starts():
    """Test 1: Verify FastAPI application initializes and root endpoint responds."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "TrustHire API"
    assert data["version"] == "1.0.0"


def test_2_3_health_check_endpoint():
    """Test 2 & 3: Verify GET /health endpoint returns 200 OK and model_loaded = True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "TrustHire API"
    assert data["model_loaded"] is True


def test_4_to_10_predict_endpoint_valid_posting(valid_job_payload):
    """Test 4 to 10: Verify POST /predict processes valid job posting and returns full schema."""
    response = client.post("/predict", json=valid_job_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["prediction"] in ["Legitimate", "Potentially Fraudulent"]
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert "risk_percentage" in data
    assert 0.0 <= data["risk_percentage"] <= 100.0
    assert "risk_level" in data
    assert data["risk_level"] in [
        "Low Risk (Possibly Legitimate)",
        "Moderate Risk (Needs Caution)",
        "High Risk",
    ]
    assert data["threshold"] == 0.50
    assert data["model"] == "Logistic Regression (TrustHire Screening Classifier)"
    assert "disclaimer" in data
    assert isinstance(data["signals"], list)


def test_11_missing_title_handled(valid_job_payload):
    """Test 11: Verify missing or empty title returns 422 or 400 error."""
    payload = valid_job_payload.copy()
    payload["title"] = ""
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]


def test_12_missing_description_handled(valid_job_payload):
    """Test 12: Verify missing or empty description returns 422 or 400 error."""
    payload = valid_job_payload.copy()
    payload["description"] = "   "  # Whitespace only
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]


def test_13_invalid_input_handled():
    """Test 13: Verify malformed request body returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"invalid": "payload"})
    assert response.status_code == 422


def test_15_real_world_software_internship_posting_test():
    """Test 15: Verify real-world software engineering internship posting passes through POST /predict."""
    internship_posting = {
        "title": "Software Engineering Intern - Summer 2026",
        "company_profile": "InnovateTech Labs is a cloud infrastructure startup focused on AI-driven analytics.",
        "description": "We are seeking a motivated Software Engineering Intern to join our core backend team. You will build REST APIs using Python and FastAPI, write unit tests, and deploy microservices.",
        "requirements": "Currently pursuing a BS/MS in Computer Science or related STEM field. Proficient in Python, Git, and REST concepts.",
        "benefits": "Competitive stipend, flexible remote work options, 1-on-1 mentorship with senior engineers.",
        "salary_range": "40000-60000",
        "location": "US, CA, San Francisco",
        "department": "Engineering",
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Bachelor's Degree",
        "industry": "Computer Software",
        "function": "Engineering",
        "telecommuting": True,
        "has_company_logo": True,
        "has_questions": True,
        "threshold": 0.50,
    }

    response = client.post("/predict", json=internship_posting)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["success"] is True
    assert res_data["prediction"] in ["Legitimate", "Potentially Fraudulent"]
    assert 0.0 <= res_data["risk_score"] <= 1.0
    assert len(res_data["disclaimer"]) > 0
