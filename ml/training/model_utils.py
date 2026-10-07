"""
TrustHire - Model Training Utilities
Provides helper functions for loading processed datasets, applying fitted preprocessing,
extracting feature names, and preparing data for non-negative models (e.g. Naive Bayes).
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from backend.app.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR


def load_processed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
    target_col: str = "fraudulent",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads train.csv and test.csv from ml/data/processed/.
    Returns (X_train, X_test, y_train, y_test).
    """
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed datasets not found in {processed_dir}. "
            "Please run Step 3 preprocessing first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in processed data.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    return X_train, X_test, y_train, y_test


def load_fitted_pipeline(models_dir: Path = SAVED_MODELS_DIR):
    """
    Loads the fitted Scikit-Learn preprocessing pipeline joblib artifact.
    """
    pipeline_path = models_dir / "preprocessing_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

    return joblib.load(pipeline_path)


def transform_data_with_pipeline(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Transforms X_train and X_test using the ALREADY FITTED preprocessing pipeline.
    Does NOT fit the pipeline again.
    """
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    return X_train_trans, X_test_trans


def get_feature_names_from_pipeline(pipeline) -> List[str]:
    """
    Extracts feature names from fitted ColumnTransformer pipeline.
    Falls back to generic feature names if not directly supported.
    """
    try:
        names = list(pipeline.get_feature_names_out())
        # Clean up transformer prefixes for readability (e.g. 'text__', 'cat__', 'num__', 'bin__')
        cleaned_names = []
        for name in names:
            if "__" in name:
                cleaned_names.append(name.split("__", 1)[1])
            else:
                cleaned_names.append(name)
        return cleaned_names
    except Exception:
        # Fallback to feature indices
        n_features = pipeline.transform(pipeline.feature_names_in_ if hasattr(pipeline, "feature_names_in_") else pd.DataFrame()).shape[1]
        return [f"feature_{i}" for i in range(n_features)]


def prepare_non_negative_data(X_train_trans, X_test_trans):
    """
    Scales feature matrices to non-negative values [0, 1] specifically for Multinomial Naive Bayes.
    Fits MinMaxScaler ONLY on X_train_trans to prevent data leakage.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_nonneg = scaler.fit_transform(X_train_trans)
    X_test_nonneg = scaler.transform(X_test_trans)
    return X_train_nonneg, X_test_nonneg
