"""
TrustHire - Single Job Posting Prediction Module
Provides the main inference interface predict_job_posting() for evaluating raw job postings
using the saved preprocessing pipeline, trained TrustHire model, and transparent NLP Risk Fusion.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config import SAVED_MODELS_DIR
from ml.preprocessing.cleaning import convert_binary_columns, apply_salary_features
from ml.preprocessing.text_processing import extract_text_derived_features
from ml.training.model_utils import get_feature_names_from_pipeline
from ml.ai.information_extractor import extract_job_information
from ml.ai.job_analyzer import detect_warning_signals

MODEL_PATH = SAVED_MODELS_DIR / "trusthire_model.joblib"
PREPROC_PATH = SAVED_MODELS_DIR / "trusthire_preprocessing.joblib"
METADATA_PATH = SAVED_MODELS_DIR / "model_metadata.json"

# Cached instances for performance
_LOADED_MODEL = None
_LOADED_PIPELINE = None
_LOADED_METADATA = None
_FEATURE_NAMES = None


def load_prediction_artifacts():
    """
    Loads saved model, preprocessing pipeline, and metadata into memory (cached).
    """
    global _LOADED_MODEL, _LOADED_PIPELINE, _LOADED_METADATA, _FEATURE_NAMES

    if _LOADED_MODEL is None or _LOADED_PIPELINE is None:
        if not MODEL_PATH.exists() or not PREPROC_PATH.exists():
            raise FileNotFoundError(
                f"Prediction artifacts missing in {SAVED_MODELS_DIR}. "
                "Please run model packaging first."
            )
        _LOADED_MODEL = joblib.load(MODEL_PATH)
        _LOADED_PIPELINE = joblib.load(PREPROC_PATH)
        _FEATURE_NAMES = get_feature_names_from_pipeline(_LOADED_PIPELINE)

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _LOADED_METADATA = json.load(f)
        else:
            _LOADED_METADATA = {}

    return _LOADED_MODEL, _LOADED_PIPELINE, _LOADED_METADATA, _FEATURE_NAMES


def validate_and_format_job_input(job_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates and formats raw job posting input dictionary into a single-row DataFrame.
    Fills missing optional fields safely.
    """
    if not isinstance(job_data, dict):
        raise TypeError(f"Expected job_data to be a dictionary, got {type(job_data).__name__}")

    # Standard expected fields with default values
    default_fields = {
        "title": "",
        "company_profile": "",
        "description": "",
        "requirements": "",
        "benefits": "",
        "salary_range": None,
        "location": "Unknown",
        "department": "Unknown",
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
        "industry": "Unknown",
        "function": "Unknown",
        "telecommuting": "f",
        "has_company_logo": "f",
        "has_questions": "f",
    }

    formatted = {}
    for k, default_val in default_fields.items():
        val = job_data.get(k, default_val)
        formatted[k] = "" if val is None and k in ["title", "company_profile", "description", "requirements", "benefits"] else val

    df_row = pd.DataFrame([formatted])
    return df_row


def predict_job_posting(
    job_data: Dict[str, Any],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Processes a raw job posting through the exact preprocessing pipeline, TrustHire ML model,
    and transparent Risk-Fusion Layer (ML probability + NLP warning signals).

    Returns a dictionary containing:
    - prediction: 0 (Legitimate) or 1 (Potentially Fraudulent)
    - prediction_label: "Legitimate" or "Potentially Fraudulent"
    - risk_score: Float between 0.0 and 1.0 (Fused final risk score)
    - risk_percentage: Float between 0.0% and 100.0%
    - risk_level: "Low Risk (Possibly Legitimate)", "Moderate Risk — Needs Verification", or "High Risk"
    - ml_risk_score: Raw Logistic Regression probability (0.0 to 1.0)
    - ml_risk_percentage: Raw model probability percentage (0.0% to 100.0%)
    - final_risk_score: Fused final risk score
    - trusthire_assessment: Human-readable screening assessment
    - nlp_risk_penalty: Sum of active NLP warning signal weights
    - warning_signals: List of detected warning signal items
    """
    model, pipeline, metadata, feature_names = load_prediction_artifacts()

    # 1. Format raw input into single-row DataFrame
    df_raw = validate_and_format_job_input(job_data)

    # 2. Apply cleaning and feature derivation pipeline
    df_binary = convert_binary_columns(
        df_raw, binary_cols=["telecommuting", "has_company_logo", "has_questions"]
    )
    df_salary = apply_salary_features(df_binary, salary_col="salary_range")
    df_processed = extract_text_derived_features(df_salary)

    # 3. Transform feature row using fitted ColumnTransformer pipeline
    X_trans = pipeline.transform(df_processed)

    # 4. Generate raw ML model probability score
    if hasattr(model, "predict_proba"):
        ml_risk_score = float(model.predict_proba(X_trans)[0, 1])
    elif hasattr(model, "decision_function"):
        d_val = float(model.decision_function(X_trans)[0])
        ml_risk_score = 1.0 / (1.0 + np.exp(-d_val))  # Sigmoid scaling
    else:
        ml_risk_score = float(model.predict(X_trans)[0])

    ml_risk_score = max(0.0, min(1.0, ml_risk_score))
    ml_risk_pct = round(ml_risk_score * 100.0, 2)

    # 5. Extract local NLP warning signals & calculate NLP risk penalty
    title_text = str(job_data.get("title") or "")
    desc_text = str(job_data.get("description") or "")
    company_text = str(job_data.get("company_profile") or "")
    req_text = str(job_data.get("requirements") or "")
    ben_text = str(job_data.get("benefits") or "")

    extracted_info = extract_job_information(
        title=title_text,
        description=desc_text,
        company_profile=company_text,
        requirements=req_text,
        benefits=ben_text,
        salary_range=job_data.get("salary_range"),
        location=job_data.get("location"),
        department=job_data.get("department"),
        employment_type=job_data.get("employment_type"),
        required_experience=job_data.get("required_experience"),
        required_education=job_data.get("required_education"),
        industry=job_data.get("industry"),
        function=job_data.get("function")
    )

    full_text = f"{title_text} {desc_text} {req_text} {ben_text} {company_text}"
    warning_signals = detect_warning_signals(extracted_info, full_text)

    # Calculate NLP risk penalty from signal weights
    nlp_risk_penalty = 0.0
    for s in warning_signals:
        w = getattr(s, "weight", 0.10)
        nlp_risk_penalty += w

    # 6. Fused Final Risk Score
    final_risk_score = max(0.0, min(1.0, ml_risk_score + nlp_risk_penalty))
    final_risk_pct = round(final_risk_score * 100.0, 2)

    # 7. Apply TrustHire Decision Rules
    if final_risk_score >= 0.60:
        trusthire_assessment = "High Risk (Highly Suspicious)"
        pred_class = 1
        pred_label = "Potentially Fraudulent"
    elif final_risk_score >= 0.35:
        trusthire_assessment = "Moderate Risk — Needs Verification"
        pred_class = 1
        pred_label = "Potentially Fraudulent"
    else:
        trusthire_assessment = "Low Risk (Possibly Legitimate)"
        pred_class = 0
        pred_label = "Legitimate"

    # 8. Extract model-associated feature signals (for Logistic Regression)
    positive_signals = []
    negative_signals = []

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        row_vals = X_trans.toarray()[0] if hasattr(X_trans, "toarray") else X_trans[0]

        contributions = row_vals * coefs
        active_indices = np.where(row_vals != 0)[0]

        signal_list = []
        for idx in active_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            contrib = float(contributions[idx])
            val = float(row_vals[idx])
            signal_list.append({
                "feature": feat_name,
                "contribution": round(contrib, 4),
                "feature_value": round(val, 4),
            })

        pos_sorted = sorted([s for s in signal_list if s["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)
        neg_sorted = sorted([s for s in signal_list if s["contribution"] < 0], key=lambda x: x["contribution"])

        positive_signals = pos_sorted[:5]
        negative_signals = neg_sorted[:5]

    return {
        "prediction": pred_class,
        "prediction_label": pred_label,
        "risk_score": round(final_risk_score, 4),
        "risk_percentage": final_risk_pct,
        "risk_level": trusthire_assessment,
        "ml_risk_score": round(ml_risk_score, 4),
        "ml_risk_percentage": ml_risk_pct,
        "final_risk_score": round(final_risk_score, 4),
        "final_risk_percentage": final_risk_pct,
        "trusthire_assessment": trusthire_assessment,
        "nlp_risk_penalty": round(nlp_risk_penalty, 4),
        "threshold_used": threshold,
        "model_name": metadata.get("model_name", "Logistic Regression"),
        "top_positive_signals": positive_signals,
        "top_negative_signals": negative_signals,
        "warning_signals": [ws.model_dump() if hasattr(ws, "model_dump") else ws for ws in warning_signals],
        "disclaimer": "This result is an automated machine-learning screening signal, not absolute proof of fraud.",
    }
