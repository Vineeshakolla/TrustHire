"""
Adversarial Job Postings Test Suite for TrustHire.
Tests 12 realistic & suspicious edge-case job postings across the Risk Fusion Layer.
Verifies that suspicious, vague, and scam-like postings receive elevated risk scores and warning signals,
while legitimate internships and entry-level roles remain low risk.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from ml.prediction.predict import predict_job_posting

client = TestClient(app)

ADVERSARIAL_JOBS = {
    # 1. Legitimate software internship
    "1_legit_software_intern": {
        "title": "Software Engineering Intern",
        "company_profile": "InnovateTech Labs is an enterprise AI startup building cloud-native analytics services.",
        "description": "We are seeking a Software Engineering Intern to join our backend engineering team. Responsibilities include constructing RESTful APIs in Python using FastAPI, writing automated unit tests, and optimizing database queries. Requirements: Currently pursuing CS or Engineering degree, proficiency in Python, Data Structures, and SQL. Duration: 6 months. Stipend: $3,000/month.",
        "requirements": "Proficiency in Python, Data Structures, SQL. CS Degree student.",
        "benefits": "Monthly stipend, mentorship, flexible hybrid schedule.",
        "location": "San Jose, CA",
        "salary_range": "3000-4000",
        "employment_type": "Internship"
    },
    # 2. Legitimate data science internship
    "2_legit_datascience_intern": {
        "title": "Data Science Intern",
        "company_profile": "DataMind Health Analytics develops predictive ML healthcare solutions.",
        "description": "Join our AI research team for a 3-month summer internship. You will preprocess EHR datasets, train classification models in PyTorch, and evaluate ROC-AUC metrics. Requirements: BS/MS candidate in Data Science or Statistics, Python, Pandas, PyTorch.",
        "requirements": "Python, Pandas, PyTorch, Statistics background.",
        "benefits": "Competitive stipend, publication opportunities.",
        "location": "Boston, MA",
        "salary_range": "35-45",
        "employment_type": "Internship"
    },
    # 3. Legitimate finance internship
    "3_legit_finance_intern": {
        "title": "Financial Analyst Intern",
        "company_profile": "Apex Asset Management manages equity portfolios for institutional investors.",
        "description": "Looking for a detail-oriented Financial Analyst Intern to assist with financial modeling, quarterly earnings analysis, and market research reports. Requirements: Pursuing degree in Finance or Economics, Advanced Excel, Financial Statement Analysis.",
        "requirements": "Finance degree candidate, Excel financial modeling.",
        "benefits": "Hourly stipend, networking with portfolio managers.",
        "location": "Chicago, IL",
        "salary_range": "25-30",
        "employment_type": "Internship"
    },
    # 4. Extremely vague internship (USER'S FAILING EXAMPLE)
    "4_failing_example_vague_intern": {
        "title": "Software intern",
        "company_profile": "",
        "description": "No experience, any one can allowed",
        "requirements": "",
        "benefits": "",
        "location": "Hyderabad",
        "department": "EEE",
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Bachelor's Degree",
        "industry": "Software Development",
        "function": ""
    },
    # 5. Missing company information
    "5_missing_company": {
        "title": "DevOps Engineer",
        "company_profile": "",
        "description": "Contract role for maintaining CI/CD pipelines and Docker containers.",
        "requirements": "Docker, Kubernetes, AWS.",
        "benefits": "Hourly pay.",
        "location": "Remote",
        "employment_type": "Contract"
    },
    # 6. "Earn huge money from home"
    "6_earn_huge_money": {
        "title": "Online Assistant — Earn Huge Money Daily",
        "company_profile": "Global Wealth Systems",
        "description": "Earn $2,000 daily working just 30 minutes a day! Easy money guaranteed from home! No work or skills required!",
        "requirements": "Bank account.",
        "benefits": "Daily payout.",
        "location": "Work From Home",
        "employment_type": "Part-time"
    },
    # 7. Registration fee required
    "7_registration_fee": {
        "title": "Data Processing Associate",
        "company_profile": "Quick Job Staffing Agency",
        "description": "Immediate start data processing job! Must pay a small registration fee of $50 before starting work to cover training materials and security deposit.",
        "requirements": "Laptop, internet.",
        "benefits": "Flexibility.",
        "location": "Remote",
        "employment_type": "Part-time"
    },
    # 8. Telegram-only recruitment
    "8_telegram_recruitment": {
        "title": "Remote Typing Specialist",
        "company_profile": "Online Admin Hub",
        "description": "Hiring typists immediately! Do not apply via email or job portal. Contact recruiter directly on Telegram @QuickRecruitNow or WhatsApp to get hired.",
        "requirements": "Telegram app, typing speed 40 wpm.",
        "benefits": "Weekly payment.",
        "location": "Remote",
        "employment_type": "Part-time"
    },
    # 9. Urgent hiring + no experience + vague description
    "9_urgent_no_experience": {
        "title": "Urgent Administrative Assistant",
        "company_profile": "",
        "description": "URGENT REQUIREMENT! Apply immediately! Hiring today! No experience needed, anyone can join today!",
        "requirements": "",
        "benefits": "",
        "location": "Remote",
        "employment_type": "Full-time"
    },
    # 10. Guaranteed income job
    "10_guaranteed_income": {
        "title": "Account Manager — Guaranteed Payout",
        "company_profile": "Fast Cash Solutions",
        "description": "Guaranteed income of $5,000 per month with no prior experience. Simply handle money transfers and receive high commission paid daily.",
        "requirements": "Personal bank account.",
        "benefits": "Guaranteed earnings.",
        "location": "Remote",
        "employment_type": "Full-time"
    },
    # 11. Data entry scam-like posting
    "11_data_entry_scam": {
        "title": "Data Entry Representative",
        "company_profile": "",
        "description": "Work from home data entry position. Earn $500 per day! Immediate start. Send registration fee to unlock work portal on Telegram.",
        "requirements": "Computer, internet.",
        "benefits": "High payout.",
        "location": "Remote",
        "employment_type": "Part-time"
    },
    # 12. Legitimate entry-level job with short description
    "12_legit_entry_level_short": {
        "title": "Junior QA Tester",
        "company_profile": "Tech Solutions Inc. provides quality assurance testing services for mobile application developers.",
        "description": "Test iOS and Android mobile applications, execute manual test suites, and log detailed bug reports in Jira.",
        "requirements": "Manual testing experience, Jira, mobile app testing.",
        "benefits": "Health insurance, 401(k), paid vacation.",
        "location": "Austin, TX",
        "salary_range": "50000-60000",
        "employment_type": "Full-time"
    }
}


def test_failing_example_vague_internship():
    """Verify that the user's failing example receives elevated risk and is NOT marked Legitimate."""
    payload = ADVERSARIAL_JOBS["4_failing_example_vague_intern"]
    res = predict_job_posting(payload)

    # Must NOT be marked Legitimate
    assert res["prediction_label"] != "Legitimate"
    assert res["final_risk_score"] >= 0.50
    assert "Moderate" in res["trusthire_assessment"] or "High" in res["trusthire_assessment"]
    assert len(res["warning_signals"]) >= 2


def test_legitimate_internships_remain_low_risk():
    """Verify that legitimate software, data science, and finance internships remain low risk."""
    for key in ["1_legit_software_intern", "2_legit_datascience_intern", "3_legit_finance_intern"]:
        payload = ADVERSARIAL_JOBS[key]
        res = predict_job_posting(payload)
        assert res["prediction_label"] == "Legitimate"
        assert res["final_risk_score"] < 0.35
        assert "Low Risk" in res["trusthire_assessment"] or "Possibility of Legitimate" in res["trusthire_assessment"]


def test_scam_categories_receive_high_risk():
    """Verify that registration fees, Telegram recruitment, guaranteed income, and data entry scams are flagged High Risk."""
    scam_keys = ["6_earn_huge_money", "7_registration_fee", "8_telegram_recruitment", "10_guaranteed_income", "11_data_entry_scam"]
    for key in scam_keys:
        payload = ADVERSARIAL_JOBS[key]
        res = predict_job_posting(payload)
        assert res["prediction_label"] == "Potentially Fraudulent"
        assert res["final_risk_score"] >= 0.50
        assert len(res["warning_signals"]) > 0


def test_api_full_analysis_adversarial_endpoints():
    """Verify /analyze/full endpoint correctly processes adversarial inputs."""
    payload = ADVERSARIAL_JOBS["4_failing_example_vague_intern"]
    response = client.post("/analyze/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] != "Legitimate"
    assert data["risk_score"] >= 0.50
    assert data["trusthire_assessment"] is not None
