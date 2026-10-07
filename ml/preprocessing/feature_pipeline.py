"""
TrustHire - Leakage-Safe Scikit-Learn Preprocessing Pipeline
Constructs a ColumnTransformer pipeline that safely combines:
1. TF-IDF vectorization on combined job text
2. One-Hot Encoding on categorical metadata (handling unseen categories)
3. Scaling on derived numerical features
4. Passthrough of binary indicator features
Excludes non-predictive identifiers such as job_id.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Define feature groups
CATEGORICAL_FEATURES = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

BINARY_FEATURES = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_available",
]

NUMERICAL_FEATURES = [
    "text_length",
    "title_length",
    "description_length",
    "requirements_length",
    "word_count",
    "url_count",
    "missing_field_count",
    "salary_min",
    "salary_max",
    "salary_mid",
]

TEXT_FEATURE = "combined_text"
EXCLUDED_FEATURES = ["job_id", "in_balanced_dataset", "salary_range", "title", "company_profile", "description", "requirements", "benefits"]


def build_preprocessing_pipeline(
    tfidf_max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer feature processing pipeline.
    Ensures zero data leakage when fitted on training data only.
    """
    # 1. Text TF-IDF Sub-Pipeline
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
        ))
    ])

    # 2. Categorical Sub-Pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 3. Numerical Sub-Pipeline
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 4. Composite ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, TEXT_FEATURE),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",  # Exclude all unlisted features (e.g., job_id)
        n_jobs=-1,
    )

    return preprocessor
