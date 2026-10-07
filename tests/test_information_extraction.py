"""
TrustHire - AI Job Information Extraction Test Suite
Tests all 15 extraction requirements and real-world test case:
1. Job title extraction
2. Location extraction
3. Remote detection
4. Employment type normalization
5. Salary extraction & parsing
6. Experience extraction
7. Education extraction
8. Skill extraction
9. Duration extraction
10. Internship detection
11. Missing information detection
12. Empty optional fields
13. Real-world long job posting test
14. Client notice company name = None
15. Deterministic non-invented values
16. POST /analyze API endpoint integration
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml.ai.information_extractor import extract_job_information

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


def test_1_job_title_extraction():
    """Test 1: Verify job title extraction from explicit title or headers."""
    info = extract_job_information(title="Software Engineering Intern", description="Python developer intern.")
    assert info.job_title == "Software Engineering Intern"


def test_2_3_location_and_remote_detection():
    """Test 2 & 3: Verify location extraction and Remote work mode detection."""
    info = extract_job_information(location="Bengaluru (Remote)", description="Remote software role.")
    assert info.location == "Bengaluru"
    assert info.work_mode == "Remote"


def test_4_employment_type_normalization():
    """Test 4: Verify employment type normalization (full-time -> Full-time)."""
    info = extract_job_information(employment_type="full-time", description="Role is full time.")
    assert info.employment_type == "Full-time"


def test_5_salary_parsing():
    """Test 5: Verify numerical salary range parsing, currency, and period."""
    info = extract_job_information(salary_range="₹30,000 – ₹50,000 per month")
    assert info.salary is not None
    assert info.salary.minimum == 30000.0
    assert info.salary.maximum == 50000.0
    assert info.salary.currency == "INR"
    assert info.salary.period == "month"


def test_6_experience_extraction():
    """Test 6: Verify experience requirement extraction."""
    info = extract_job_information(description="Min Experience: 0 years in coding.")
    assert info.experience == "0 years"


def test_7_education_extraction():
    """Test 7: Verify explicit education requirement extraction."""
    info = extract_job_information(description="Requires degree in Computer Science or Engineering.")
    assert info.education is not None
    assert "Computer Science" in info.education or "Engineering" in info.education


def test_8_skill_extraction():
    """Test 8: Verify technical skill extraction using SkillDictionary."""
    info = extract_job_information(description="Proficiency in Python, JavaScript, Java, and C++. Familiarity with React and Docker.")
    assert "Python" in info.skills
    assert "JavaScript" in info.skills
    assert "Java" in info.skills
    assert "C++" in info.skills
    assert "React" in info.skills
    assert "Docker" in info.skills


def test_9_duration_extraction():
    """Test 9: Verify internship/contract duration extraction."""
    info = extract_job_information(description="Duration: Minimum 2 months, extendable up to 6 months.")
    assert info.duration == "Minimum 2 months, extendable up to 6 months"


def test_10_internship_detection():
    """Test 10: Verify internship detection flag (internship = True)."""
    info = extract_job_information(title="Software Engineering Intern")
    assert info.internship is True


def test_11_missing_information_detection():
    """Test 11: Verify missing critical fields identification."""
    info = extract_job_information(title="Python Developer", description="Short job post.")
    assert "Salary / Stipend" in info.important_missing_information
    assert "Company Information" in info.important_missing_information


def test_12_empty_optional_fields():
    """Test 12: Verify extractor handles empty or null optional fields safely."""
    info = extract_job_information(title="Software Engineer", description="Clean description.")
    assert info.job_title == "Software Engineer"
    assert info.company_name is None


def test_13_14_real_world_job_posting_test_case():
    """Test 13 & 14: Test extractor against section 24 real-world job posting text."""
    info = extract_job_information(description=REAL_WORLD_JOB_POSTING)

    assert info.job_title == "Software Engineering Intern"
    assert info.company_name is None  # Test 14: Client notice company name remains None
    assert info.location == "Bengaluru"
    assert info.work_mode == "Remote"
    assert info.employment_type == "Full-time"
    assert info.salary is not None
    assert info.salary.minimum == 30000.0
    assert info.salary.maximum == 50000.0
    assert info.experience == "0 years"
    assert info.duration == "Minimum 2 months, extendable up to 6 months"
    assert info.internship is True

    # Check extracted skills
    assert "Python" in info.skills
    assert "JavaScript" in info.skills
    assert "Java" in info.skills
    assert "C++" in info.skills
    assert "Data Structures" in info.skills
    assert "Algorithms" in info.skills

    # Check requirements list
    assert len(info.requirements) > 0


def test_15_non_invented_values():
    """Test 15: Verify extractor does not invent values when posting provides no evidence."""
    info = extract_job_information(title="Generic Job", description="Standard text with no salary or location.")
    assert info.salary is None
    assert info.location is None
    assert info.company_name is None


def test_16_post_analyze_api_endpoint():
    """Test 16: Verify POST /analyze API endpoint returns structured JSON response."""
    payload = {
        "title": "Software Engineering Intern",
        "description": REAL_WORLD_JOB_POSTING
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    ext = data["extracted_information"]
    assert ext["job_title"] == "Software Engineering Intern"
    assert ext["location"] == "Bengaluru"
    assert ext["work_mode"] == "Remote"
    assert ext["internship"] is True
    assert data["metadata"]["external_api_used"] is False
