"""
TrustHire - Model Training & Evaluation Test Suite
Tests all 8 requirements for Step 4:
1. Candidate models can be trained
2. Predictions have correct length
3. Metrics are calculated
4. Confusion matrices are generated
5. Model comparison table is generated
6. Saved model files can be loaded
7. Logistic Regression feature coefficients can be extracted
8. No NaN metrics are produced unexpectedly
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from ml.training.model_utils import (
    load_processed_data,
    load_fitted_pipeline,
    transform_data_with_pipeline,
    prepare_non_negative_data,
    get_feature_names_from_pipeline,
)
from ml.training.train_models import (
    train_logistic_regression,
    train_linear_svm,
    train_multinomial_nb,
    train_random_forest,
    train_xgboost,
    XGBOOST_AVAILABLE,
)
from ml.evaluation.metrics import (
    evaluate_single_model,
    extract_logistic_regression_features,
    generate_comparison_reports,
)
from ml.evaluation.visualizations import generate_all_evaluation_plots
from backend.app.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR, ML_DIR


@pytest.fixture(scope="module")
def processed_data_and_pipeline():
    """Loads dataset and pipeline once for modeling tests."""
    X_train_df, X_test_df, y_train, y_test = load_processed_data(PROCESSED_DATA_DIR)
    pipeline = load_fitted_pipeline(SAVED_MODELS_DIR)
    X_train_trans, X_test_trans = transform_data_with_pipeline(pipeline, X_train_df, X_test_df)
    X_train_nonneg, X_test_nonneg = prepare_non_negative_data(X_train_trans, X_test_trans)

    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "X_train_nonneg": X_train_nonneg,
        "X_test_nonneg": X_test_nonneg,
    }


def test_1_models_can_be_trained(processed_data_and_pipeline):
    """Test 1: Verify all candidate models can be initialized and trained."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    assert lr is not None

    svm = train_linear_svm(data["X_train_trans"], data["y_train"])
    assert svm is not None

    nb = train_multinomial_nb(data["X_train_nonneg"], data["y_train"])
    assert nb is not None

    rf = train_random_forest(data["X_train_trans"], data["y_train"])
    assert rf is not None

    if XGBOOST_AVAILABLE:
        xgb = train_xgboost(data["X_train_trans"], data["y_train"])
        assert xgb is not None


def test_2_predictions_length(processed_data_and_pipeline):
    """Test 2: Verify model predictions match the test set length."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    preds = lr.predict(data["X_test_trans"])
    assert len(preds) == len(data["y_test"])


def test_3_8_metrics_calculation_no_nans(processed_data_and_pipeline):
    """Test 3 & 8: Verify evaluation metrics are calculated without NaN values."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    eval_res = evaluate_single_model(lr, data["X_test_trans"], data["y_test"], "Logistic Regression")

    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        val = eval_res[metric_name]
        assert not np.isnan(val)
        assert 0.0 <= val <= 1.0


def test_4_confusion_matrices_generated(processed_data_and_pipeline, tmp_path):
    """Test 4: Verify confusion matrix plot generation."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    eval_res = evaluate_single_model(lr, data["X_test_trans"], data["y_test"], "Logistic Regression")

    plots = generate_all_evaluation_plots([eval_res], output_dir=tmp_path)
    assert len(plots) >= 3
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0


def test_5_model_comparison_table(processed_data_and_pipeline, tmp_path):
    """Test 5: Verify model comparison CSV and Markdown table generation."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    eval_res = evaluate_single_model(lr, data["X_test_trans"], data["y_test"], "Logistic Regression")

    csv_path, md_path = generate_comparison_reports([eval_res], output_dir=tmp_path)
    assert csv_path.exists()
    assert md_path.exists()
    assert csv_path.stat().st_size > 0
    assert md_path.stat().st_size > 0


def test_6_saved_model_loading(processed_data_and_pipeline, tmp_path):
    """Test 6: Verify saved model joblib artifacts can be saved and loaded back."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    model_file = tmp_path / "test_lr.joblib"
    joblib.dump(lr, model_file)

    loaded_lr = joblib.load(model_file)
    preds = loaded_lr.predict(data["X_test_trans"])
    assert len(preds) == len(data["y_test"])


def test_7_lr_coefficients_extraction(processed_data_and_pipeline, tmp_path):
    """Test 7: Verify extraction of Logistic Regression feature coefficients."""
    data = processed_data_and_pipeline
    lr = train_logistic_regression(data["X_train_trans"], data["y_train"])
    feature_names = get_feature_names_from_pipeline(data["pipeline"])

    out_csv = tmp_path / "lr_features.csv"
    df_feats = extract_logistic_regression_features(lr, feature_names, output_csv=out_csv)

    assert not df_feats.empty
    assert "feature" in df_feats.columns
    assert "coefficient" in df_feats.columns
    assert "direction" in df_feats.columns
    assert out_csv.exists()
