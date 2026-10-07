"""
TrustHire - Single Job Posting Real-World Demonstration Script
Runs a sample realistic job posting through the prediction module and prints the result.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.prediction.predict import predict_job_posting


def run_demo():
    sample_posting = {
        "title": "Data Entry Specialist - Earn $35/hr Work From Home",
        "company_profile": "",  # Missing profile
        "description": "We are seeking motivated individuals for immediate data entry positions. No experience required. High weekly income guaranteed.",
        "requirements": "Basic computer knowledge and internet connection.",
        "benefits": "Flexible schedule, daily payment options.",
        "salary_range": "30000-50000",
        "location": "US, NY, New York",
        "department": "Administrative",
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "Unspecified",
        "industry": "Information Technology and Services",
        "function": "Administrative",
        "telecommuting": "t",
        "has_company_logo": "f",
        "has_questions": "f",
    }

    result = predict_job_posting(sample_posting, threshold=0.50)
    print("\n==================================================")
    print("TRUSTHIRE REAL-WORLD SINGLE JOB INFERENCE DEMO")
    print("==================================================")
    print(json.dumps(result, indent=4))
    print("==================================================\n")


if __name__ == "__main__":
    run_demo()
