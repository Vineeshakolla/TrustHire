"""
TrustHire - Configuration Module
Provides portable relative paths for data, models, and database resources.
"""

from pathlib import Path

# Project root directory (TrustHire/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Backend directory
BACKEND_DIR = BASE_DIR / "backend"

# ML Data & Artifact directories
ML_DIR = BASE_DIR / "ml"
RAW_DATA_DIR = ML_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ML_DIR / "data" / "processed"
SAVED_MODELS_DIR = ML_DIR / "saved_models"
EDA_OUTPUT_DIR = ML_DIR / "data" / "eda_outputs"
DATASET_REPORT_JSON = ML_DIR / "data" / "dataset_report.json"
DATASET_REPORT_MD = ML_DIR / "data" / "dataset_report.md"


# AI directories
AI_DIR = BASE_DIR / "ai"
AI_MODELS_DIR = AI_DIR / "models"
AI_PROMPTS_DIR = AI_DIR / "prompts"

# Database directory and file path
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "trusthire.db"

# Application Metadata
PROJECT_NAME = "TrustHire"
PROJECT_VERSION = "0.1.0"
PROJECT_DESCRIPTION = (
    "Machine Learning-Based Detection of Potentially Fraudulent Job and Internship Postings"
)
