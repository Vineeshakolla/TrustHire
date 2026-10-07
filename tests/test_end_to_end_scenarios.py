"""
End-to-End Realistic Scenario Test Suite for TrustHire.
Tests 10 diverse real-world job posting scenarios against the full backend pipeline:
ML Risk Classification + NLP Extraction + Intelligent Job Analysis + Local Q&A.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# 10 Diverse Real-World Job Posting Test Cases
SCENARIOS = {
    "A_software_intern": {
        "title": "Software Engineering Intern",
        "company_profile": "InnovateTech Labs is a cloud infrastructure startup developing high-throughput analytics tools.",
        "description": "We are seeking a motivated Software Engineering Intern to join our backend engineering team. Responsibilities include building REST APIs using Python and FastAPI, writing unit tests, and improving performance. Requirements: Currently pursuing CS degree, proficient in Python, C++, Java, Data Structures, Algorithms. Stipend: ₹30,000 – ₹50,000 per month. Duration: 6 months. Start Date: Immediate.",
        "requirements": "Proficiency in Python, C++, Java, Data Structures, Algorithms.",
        "benefits": "Competitive stipend, flexible hybrid work environment, 1-on-1 mentorship.",
        "location": "Bengaluru (Remote)",
        "salary_range": "30000-50000",
        "employment_type": "Internship"
    },
    "B_software_fulltime": {
        "title": "Senior Backend Engineer",
        "company_profile": "Global Enterprise Systems provides scalable enterprise cloud database solutions.",
        "description": "Looking for a Senior Backend Engineer with 5+ years experience building distributed microservices in Go and Java. Experience with Kubernetes, PostgreSQL, and Kafka required. Compensation: $140,000 - $180,000 per year. Full benefits package included.",
        "requirements": "5+ years backend experience, Go, Java, Kubernetes, PostgreSQL, Kafka.",
        "benefits": "Health insurance, 401(k) matching, unlimited PTO.",
        "location": "San Francisco, CA",
        "salary_range": "140000-180000",
        "employment_type": "Full-time"
    },
    "C_datascience_intern": {
        "title": "Data Science & Machine Learning Intern",
        "company_profile": "DataMind Analytics specializes in predictive machine learning models for healthcare.",
        "description": "Join our AI research team as a Data Science Intern. You will assist in cleaning datasets, building ML pipelines using PyTorch and Scikit-Learn, and visualizing model evaluation metrics. Stipend: $35/hr. Duration: 3 months.",
        "requirements": "BS or MS in Data Science, Computer Science, or Math. PyTorch, Python, Pandas, Scikit-Learn.",
        "benefits": "Mentorship, publication opportunities, stipend.",
        "location": "Boston, MA (Hybrid)",
        "salary_range": "30-40",
        "employment_type": "Internship"
    },
    "D_marketing_intern": {
        "title": "Digital Marketing Intern",
        "company_profile": "Creative Brand Agency manages social media campaigns for global e-commerce brands.",
        "description": "We are hiring a Digital Marketing Intern to design social graphics, manage SEO content, and analyze campaign analytics. Duration: 4 months. Stipend: $20/hr. Requirements: Basic Photoshop, SEO knowledge, strong writing skills.",
        "requirements": "Photoshop, SEO, social media management, Google Analytics.",
        "benefits": "Flexible hours, agency networking, stipend.",
        "location": "New York, NY",
        "salary_range": "15-25",
        "employment_type": "Internship"
    },
    "E_finance_job": {
        "title": "Financial Analyst",
        "company_profile": "Apex Investment Group offers asset management and equity research services.",
        "description": "Responsible for financial forecasting, building DCF models, and preparing investment committee presentations. Requirements: Bachelor's in Finance or Economics, 2+ years experience, Excel financial modeling, CFA Level 1 preferred. Salary: $85,000 - $105,000.",
        "requirements": "Finance degree, Excel modeling, DCF valuation, CFA Level 1.",
        "benefits": "Performance bonus, health insurance, tuition reimbursement.",
        "location": "Chicago, IL",
        "salary_range": "85000-105000",
        "employment_type": "Full-time"
    },
    "F_missing_salary": {
        "title": "Full Stack Web Developer",
        "company_profile": "NextGen Web Solutions builds custom React and Node.js applications.",
        "description": "Seeking a Full Stack Web Developer proficient in React, Node.js, Express, and MongoDB. You will implement user interfaces and backend APIs for client applications. Location: Austin, TX. Experience: 2 years.",
        "requirements": "React, Node.js, Express, MongoDB, HTML, CSS, JavaScript.",
        "benefits": "Health insurance, remote work options.",
        "location": "Austin, TX",
        "salary_range": None,
        "employment_type": "Full-time"
    },
    "G_missing_company": {
        "title": "Java Systems Engineer",
        "company_profile": "",  # Empty company profile
        "description": "Client representative posting for a Java Systems Engineer. Duties: maintain legacy Java backend, refactor database queries, and optimize memory usage. Requirements: 3+ years Java, Spring Boot, MySQL.",
        "requirements": "Java 17, Spring Boot, Microservices, SQL.",
        "benefits": "Competitive compensation.",
        "location": "Remote",
        "salary_range": "60000-90000",
        "employment_type": "Contract"
    },
    "H_short_description": {
        "title": "Junior QA Tester",
        "company_profile": "Tech Solutions",
        "description": "Test mobile apps and report bugs.",  # Very short description (<100 chars)
        "requirements": "Manual testing, Bug reporting.",
        "benefits": "Standard benefits.",
        "location": "Remote",
        "salary_range": "20000-30000",
        "employment_type": "Full-time"
    },
    "I_suspicious_urgent": {
        "title": "Data Entry Specialist — Immediate Joining",
        "company_profile": "Global Staffing Agency",
        "description": "URGENT REQUIREMENT! Apply immediately for Data Entry Specialist role. No experience needed! Immediate joining guaranteed! Transfer small processing fee or registration fee to start immediately. Contact via Telegram or WhatsApp.",
        "requirements": "Basic computer skills, mobile phone, internet connection.",
        "benefits": "Daily payout, work from home.",
        "location": "Work From Home",
        "salary_range": "5000-10000",
        "employment_type": "Part-time"
    },
    "J_unusually_high_income": {
        "title": "Online Assistant — Easy Money Guaranteed",
        "company_profile": "Quick Earn Enterprises",
        "description": "Earn $5,000 weekly working just 1 hour a day! Guaranteed income! Easy money guaranteed with no prior experience! Simply process online payments and transfer money. High commission paid daily!",
        "requirements": "Bank account, internet connection.",
        "benefits": "Guaranteed weekly income.",
        "location": "Remote",
        "salary_range": "200000-500000",
        "employment_type": "Part-time"
    }
}


@pytest.mark.parametrize("scenario_name,payload", list(SCENARIOS.items()))
def test_end_to_end_scenarios_full_analysis(scenario_name, payload):
    """Verify that all 10 realistic job postings pass through /analyze/full cleanly."""
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200, f"Failed for scenario {scenario_name}"

    data = response.json()
    assert data["success"] is True
    assert data["prediction"] in ["Legitimate", "Potentially Fraudulent"]
    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["risk_percentage"] <= 100.0
    assert "Risk" in data["risk_level"] or "Legitimate" in data["risk_level"]
    assert isinstance(data["feature_signals"], list)
    assert "summary" in data["intelligent_analysis"]
    assert "skills" in data["intelligent_analysis"]
    assert "warning_signals" in data["intelligent_analysis"]
    assert "disclaimer" in data


def test_scenario_missing_salary_flagging():
    """Verify scenario F detects missing salary in missing_information list."""
    payload = SCENARIOS["F_missing_salary"]
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    missing_info = data["intelligent_analysis"]["missing_information"]
    assert any("salary" in item.lower() for item in missing_info)


def test_scenario_missing_company_warning():
    """Verify scenario G generates a warning signal for missing company profile."""
    payload = SCENARIOS["G_missing_company"]
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    warning_signals = data["intelligent_analysis"]["warning_signals"]
    assert len(warning_signals) > 0
    assert any("company" in ws["signal"].lower() or "client" in ws["signal"].lower() for ws in warning_signals)


def test_scenario_suspicious_urgent_warning_signals():
    """Verify scenario I detects urgency, registration fee, and off-platform contact warning signals."""
    payload = SCENARIOS["I_suspicious_urgent"]
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    warning_signals = data["intelligent_analysis"]["warning_signals"]
    assert len(warning_signals) >= 2
    signals_text = [ws["signal"].lower() for ws in warning_signals]
    assert any("urgency" in s or "urgent" in s or "immediate" in s for s in signals_text)
    assert any("fee" in s or "payment" in s or "communication" in s or "channel" in s for s in signals_text)


def test_scenario_high_income_warning_signals():
    """Verify scenario J detects guaranteed income claims warning signal."""
    payload = SCENARIOS["J_unusually_high_income"]
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    warning_signals = data["intelligent_analysis"]["warning_signals"]
    assert len(warning_signals) >= 1
    signals_text = [ws["signal"].lower() for ws in warning_signals]
    assert any("income" in s or "guarantee" in s or "earnings" in s for s in signals_text)


def test_scenario_qa_questions():
    """Verify Q&A endpoint answers questions accurately across scenarios."""
    payload = SCENARIOS["A_software_intern"]
    
    # Question 1: Salary
    res1 = client.post("/analyze/ask", json={"job_posting": payload, "question": "What is the salary?"})
    assert res1.status_code == 200
    assert "30000" in res1.json()["answer"] or "stipend" in res1.json()["answer"].lower() or "compensation" in res1.json()["answer"].lower()

    # Question 2: Skills
    res2 = client.post("/analyze/ask", json={"job_posting": payload, "question": "What skills are required?"})
    assert res2.status_code == 200
    assert "Python" in res2.json()["answer"]

    # Question 3: Unmentioned Detail
    res3 = client.post("/analyze/ask", json={"job_posting": payload, "question": "What is the visa sponsorship policy?"})
    assert res3.status_code == 200
    assert "not explicitly provided" in res3.json()["answer"].lower()
