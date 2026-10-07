"""
TrustHire - Backend Configuration Module
Re-exports portable paths and project metadata from backend.app.config.
"""

from backend.app.config import (
    BASE_DIR,
    BACKEND_DIR,
    ML_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    EDA_OUTPUT_DIR,
    DB_DIR,
    DB_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_DESCRIPTION,
)

MODEL_PATH = SAVED_MODELS_DIR / "trusthire_model.joblib"
PREPROCESSING_PATH = SAVED_MODELS_DIR / "trusthire_preprocessing.joblib"
METADATA_PATH = SAVED_MODELS_DIR / "model_metadata.json"
