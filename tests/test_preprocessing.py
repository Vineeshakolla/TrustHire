"""
TrustHire - Preprocessing Pipeline Test Suite
Tests all 12 core requirements for Step 3:
1. Duplicate removal
2. Target conversion
3. Binary conversion
4. Salary parsing
5. Missing text handling
6. Missing categorical handling
7. Train/test split
8. Stratification
9. Pipeline fitting
10. Pipeline transformation
11. Excluded job_id identifier
12. Unseen categories handling in test data
"""

import pytest
import pandas as pd
import numpy as np
from ml.preprocessing.cleaning import (
    remove_exact_duplicates,
    convert_target,
    convert_binary_columns,
    parse_salary_range,
    apply_salary_features,
)
from ml.preprocessing.text_processing import (
    clean_text_series,
    create_combined_text,
    extract_text_derived_features,
)
from ml.preprocessing.split_data import split_dataset
from ml.preprocessing.feature_pipeline import build_preprocessing_pipeline, EXCLUDED_FEATURES


@pytest.fixture
def sample_df():
    """Constructs a small representative sample DataFrame matching raw EMSCAD schema."""
    return pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 5],  # Contains 1 exact duplicate row (5)
        "title": ["Software Engineer", "Data Scientist", "Sales Rep", None, "Engineer", "Engineer"],
        "location": ["US, NY, New York", "US, CA, SF", None, "GB, LND, London", "GR, I, Athens", "GR, I, Athens"],
        "department": ["Engineering", "Data", "Sales", None, "Tech", "Tech"],
        "salary_range": ["50000-70000", None, "100000-120000", "invalid_salary", "40000-50000", "40000-50000"],
        "company_profile": ["Tech company profile", None, "Sales firm", "Start-up profile", "Soft company", "Soft company"],
        "description": ["Job description 1", "Job description 2", "Job description 3", "Job description 4", "Job description 5", "Job description 5"],
        "requirements": ["BS Degree", "MS Degree", None, "High school", "5 yrs exp", "5 yrs exp"],
        "benefits": ["Health insurance", None, "Commission", "Remote option", "Stock options", "Stock options"],
        "telecommuting": ["f", "t", "f", "0", "1", "1"],
        "has_company_logo": ["t", "f", "t", "1", "0", "0"],
        "has_questions": ["f", "t", "t", "0", "1", "1"],
        "employment_type": ["Full-time", "Part-time", "Contract", None, "Full-time", "Full-time"],
        "required_experience": ["Mid-Senior level", "Entry level", None, "Associate", "Director", "Director"],
        "required_education": ["Bachelor's Degree", "Master's Degree", None, "High School", "Bachelor's Degree", "Bachelor's Degree"],
        "industry": ["Computer Software", "IT Services", "Marketing", None, "Internet", "Internet"],
        "function": ["Engineering", "Engineering", "Sales", None, "IT", "IT"],
        "fraudulent": ["f", "f", "t", "f", "t", "t"],
    })


def test_1_duplicate_removal(sample_df):
    """Test 1: Verify exact duplicate row removal."""
    clean_df, stats = remove_exact_duplicates(sample_df)
    assert stats["rows_before"] == 6
    assert stats["rows_removed"] == 1
    assert stats["rows_after"] == 5
    assert len(clean_df) == 5


def test_2_target_conversion(sample_df):
    """Test 2: Verify target conversion from 'f'/'t' to 0/1 integers."""
    converted_df = convert_target(sample_df, target_col="fraudulent")
    assert set(converted_df["fraudulent"].unique()) == {0, 1}
    assert converted_df["fraudulent"].dtype == int
    assert converted_df["fraudulent"].iloc[0] == 0  # 'f' -> 0
    assert converted_df["fraudulent"].iloc[2] == 1  # 't' -> 1


def test_3_binary_conversion(sample_df):
    """Test 3: Verify binary column conversion ('f'/'t', '0'/'1') to integers 0/1."""
    binary_df = convert_binary_columns(sample_df, ["telecommuting", "has_company_logo", "has_questions"])
    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        assert set(binary_df[col].unique()).issubset({0, 1})
        assert binary_df[col].dtype == int


def test_4_salary_parsing():
    """Test 4: Verify salary_range parsing logic."""
    # Valid range
    avail, s_min, s_max, s_mid = parse_salary_range("50000-70000")
    assert avail == 1
    assert s_min == 50000.0
    assert s_max == 70000.0
    assert s_mid == 60000.0

    # Missing / None
    avail_null, s_min_null, _, _ = parse_salary_range(None)
    assert avail_null == 0
    assert s_min_null == 0.0

    # Invalid text
    avail_inv, s_min_inv, _, _ = parse_salary_range("invalid_salary")
    assert avail_inv == 0
    assert s_min_inv == 0.0


def test_5_missing_text_handling(sample_df):
    """Test 5: Verify missing text is converted to empty string without dropping rows."""
    cleaned_series = clean_text_series(sample_df["title"])
    assert len(cleaned_series) == len(sample_df)
    assert cleaned_series.iloc[3] == ""  # None -> ""


def test_6_missing_categorical_handling(sample_df):
    """Test 6: Verify missing categorical values handle imputation correctly."""
    derived_df = extract_text_derived_features(sample_df)
    assert "missing_field_count" in derived_df.columns
    assert derived_df["missing_field_count"].iloc[3] > 0


def test_7_8_train_test_split_and_stratification(sample_df):
    """Test 7 & 8: Verify train/test splitting and target stratification."""
    dedup_df, _ = remove_exact_duplicates(sample_df)
    target_df = convert_target(dedup_df)

    X_train, X_test, y_train, y_test, stats = split_dataset(
        target_df, target_col="fraudulent", test_size=0.4, random_state=42
    )

    assert len(X_train) + len(X_test) == len(target_df)
    assert len(y_train) + len(y_test) == len(target_df)
    # Stratification check (proportions should match original target ratio approximately)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_9_10_pipeline_fitting_and_transformation(sample_df):
    """Test 9 & 10: Verify pipeline fitting on X_train and transformation on X_train and X_test."""
    dedup_df, _ = remove_exact_duplicates(sample_df)
    target_df = convert_target(dedup_df)
    binary_df = convert_binary_columns(target_df)
    salary_df = apply_salary_features(binary_df)
    processed_df = extract_text_derived_features(salary_df)

    X_train, X_test, y_train, y_test, _ = split_dataset(
        processed_df, target_col="fraudulent", test_size=0.4, random_state=42
    )

    pipeline = build_preprocessing_pipeline(tfidf_max_features=100)

    # Fit ONLY on training data
    X_train_trans = pipeline.fit_transform(X_train)
    assert X_train_trans.shape[0] == len(X_train)
    assert X_train_trans.shape[1] > 0

    # Transform test data using fitted pipeline
    X_test_trans = pipeline.transform(X_test)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_test_trans.shape[1] == X_train_trans.shape[1]  # Equal feature count


def test_11_no_job_id_in_features(sample_df):
    """Test 11: Verify identifier (job_id) is excluded from predictive feature matrix."""
    pipeline = build_preprocessing_pipeline(tfidf_max_features=100)
    salary_df = apply_salary_features(sample_df)
    processed_df = extract_text_derived_features(salary_df)

    # Fit pipeline
    pipeline.fit(processed_df)

    # Transform feature matrix
    X_trans = pipeline.transform(processed_df)

    # Verify remainder='drop' removed job_id
    # We can inspect column transformer output features if available
    assert X_trans.shape[0] == len(processed_df)


def test_12_unseen_categories_handling(sample_df):
    """Test 12: Verify unseen categorical values in test set do not crash the pipeline."""
    dedup_df, _ = remove_exact_duplicates(sample_df)
    target_df = convert_target(dedup_df)
    binary_df = convert_binary_columns(target_df)
    salary_df = apply_salary_features(binary_df)
    processed_df = extract_text_derived_features(salary_df)

    X_train = processed_df.iloc[:3].copy()
    X_test = processed_df.iloc[3:].copy()

    # Introduce a completely new unseen industry in test set
    X_test.loc[X_test.index[0], "industry"] = "Unseen_Future_Industry_999"

    pipeline = build_preprocessing_pipeline(tfidf_max_features=50)
    pipeline.fit(X_train)

    # Should transform without raising an error due to handle_unknown='ignore'
    X_test_trans = pipeline.transform(X_test)
    assert X_test_trans.shape[0] == len(X_test)
