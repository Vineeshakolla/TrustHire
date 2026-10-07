"""
TrustHire - Single Job Prediction Module Test Suite
Tests all 10 requirements for Step 5:
1. Final model loads successfully
2. Preprocessing pipeline loads successfully
3. Model and preprocessing feature dimensions match
4. Valid job posting can be processed
5. Prediction is generated
6. Risk score is generated
7. Missing optional fields do not crash prediction
8. Invalid input is handled safely
9. No training occurs during prediction
10. All existing tests pass
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from ml.prediction.predict import predict_job_posting, load_prediction_artifacts
from backend.app.config import SAVED_MODELS_DIR


@pytest.fixture
def sample_real_job_posting():
    """Constructs a realistic full job posting for inference testing."""
    return {
        "title": "Entry Level Data Entry Clerk / Remote Assistant",
        "company_profile": "Fast growing online services company expanding customer support nationwide.",
        "description": "Earn $30-$40 per hour working from home! No experience required. Immediate start available.",
        "requirements": "Basic computer skills, high speed internet connection, willingness to learn.",
        "benefits": "Flexible hours, weekly payout, work from home opportunity.",
        "salary_range": "30000-50000",
        "location": "US, NY, New York",
        "department": "Administrative",
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "High School or equivalent",
        "industry": "Information Technology and Services",
        "function": "Administrative",
        "telecommuting": "t",
        "has_company_logo": "f",
        "has_questions": "f",
    }


def test_1_final_model_loads():
    """Test 1: Verify trusthire_model.joblib loads successfully."""
    model_path = SAVED_MODELS_DIR / "trusthire_model.joblib"
    assert model_path.exists()
    model = joblib.load(model_path)
    assert model is not None


def test_2_preprocessing_pipeline_loads():
    """Test 2: Verify trusthire_preprocessing.joblib loads successfully."""
    preproc_path = SAVED_MODELS_DIR / "trusthire_preprocessing.joblib"
    assert preproc_path.exists()
    pipeline = joblib.load(preproc_path)
    assert pipeline is not None


def test_3_feature_dimensions_match():
    """Test 3: Verify model expected features match pipeline output features."""
    model, pipeline, _, _ = load_prediction_artifacts()
    # Dummy row transformation
    dummy_df = pd.DataFrame([{
        "title": "Software Engineer",
        "company_profile": "Profile",
        "description": "Description",
        "requirements": "Reqs",
        "benefits": "Benefits",
        "salary_range": "50000-70000",
        "location": "US, NY, New York",
        "department": "Tech",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Software",
        "function": "Engineering",
        "telecommuting": 0,
        "has_company_logo": 1,
        "has_questions": 1,
        "combined_text": "TITLE: Software Engineer | COMPANY_PROFILE: Profile | DESCRIPTION: Description | REQUIREMENTS: Reqs | BENEFITS: Benefits",
        "text_length": 100,
        "title_length": 17,
        "description_length": 11,
        "requirements_length": 4,
        "word_count": 15,
        "url_count": 0,
        "missing_field_count": 0,
        "salary_available": 1,
        "salary_min": 50000.0,
        "salary_max": 70000.0,
        "salary_mid": 60000.0,
    }])
    trans = pipeline.transform(dummy_df)
    assert trans.shape[1] == model.coef_.shape[1]


def test_4_5_6_valid_job_prediction_and_risk_score(sample_real_job_posting):
    """Test 4, 5 & 6: Verify a valid job posting generates prediction, label, risk score, and signals."""
    res = predict_job_posting(sample_real_job_posting)

    assert "prediction" in res
    assert res["prediction"] in [0, 1]
    assert "prediction_label" in res
    assert res["prediction_label"] in ["Legitimate", "Potentially Fraudulent"]
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert "risk_percentage" in res
    assert 0.0 <= res["risk_percentage"] <= 100.0
    assert "risk_level" in res
    assert "disclaimer" in res


def test_7_missing_optional_fields_do_not_crash():
    """Test 7: Verify missing or partial fields do not crash prediction."""
    minimal_job = {
        "title": "Backend Python Developer",
        "description": "We are hiring a Python FastAPI developer to work on ML microservices.",
    }
    res = predict_job_posting(minimal_job)
    assert res["prediction"] in [0, 1]
    assert 0.0 <= res["risk_score"] <= 1.0


def test_8_invalid_input_handled_safely():
    """Test 8: Verify invalid non-dict input raises a clear TypeError."""
    with pytest.raises(TypeError):
        predict_job_posting("invalid_string_input")


def test_9_no_training_during_prediction(sample_real_job_posting, monkeypatch):
    """Test 9: Verify model fit() is never called during prediction execution."""
    model, _, _, _ = load_prediction_artifacts()

    def raise_if_fit_called(*args, **kwargs):
        raise RuntimeError("fit() must NEVER be called during prediction!")

    monkeypatch.setattr(model, "fit", raise_if_fit_called)
    # Should run successfully without triggering fit()
    res = predict_job_posting(sample_real_job_posting)
    assert res is not None
