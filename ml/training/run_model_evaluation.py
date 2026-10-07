"""
TrustHire - Complete Model Training, Comparison & Evaluation Module
Orchestrates loading processed data & pipeline, training candidate models,
evaluating performance metrics, generating plots & comparison reports,
and extracting Logistic Regression model signals.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config import (
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    ML_DIR,
)

from ml.training.model_utils import (
    load_processed_data,
    load_fitted_pipeline,
    transform_data_with_pipeline,
    get_feature_names_from_pipeline,
    prepare_non_negative_data,
)
from ml.training.train_models import train_all_models, save_models, XGBOOST_AVAILABLE
from ml.evaluation.metrics import (
    evaluate_single_model,
    extract_logistic_regression_features,
    generate_comparison_reports,
)
from ml.evaluation.visualizations import generate_all_evaluation_plots

EVALUATION_REPORT_MD = ML_DIR / "evaluation" / "model_evaluation_report.md"
LOGISTIC_FEATS_CSV = ML_DIR / "evaluation" / "logistic_regression_features.csv"


def run_full_model_evaluation() -> Dict[str, Any]:
    """
    Executes the complete Step 4 model training, comparison, and evaluation pipeline.
    """
    print("==================================================")
    print("TRUSTHIRE — STEP 4 MODEL TRAINING & EVALUATION")
    print("==================================================")

    # 1. Load processed train/test data & fitted pipeline
    print("Loading processed datasets and fitted preprocessing pipeline...")
    X_train_df, X_test_df, y_train, y_test = load_processed_data(PROCESSED_DATA_DIR)
    pipeline = load_fitted_pipeline(SAVED_MODELS_DIR)

    # 2. Transform feature matrices without refitting pipeline
    print("Transforming feature matrices using fitted pipeline...")
    X_train_trans, X_test_trans = transform_data_with_pipeline(pipeline, X_train_df, X_test_df)

    # Verify matrix shapes and check for NaNs
    print(f"X_train transformed shape: {X_train_trans.shape}")
    print(f"X_test transformed shape:  {X_test_trans.shape}")

    # 3. Prepare non-negative feature matrices specifically for Multinomial Naive Bayes
    X_train_nonneg, X_test_nonneg = prepare_non_negative_data(X_train_trans, X_test_trans)

    # 4. Train candidate models
    print("\nTraining candidate machine learning models...")
    models_dict = train_all_models(X_train_trans, y_train, X_train_nonneg, random_state=42)

    # 5. Save trained model joblib artifacts
    saved_paths = save_models(models_dict, SAVED_MODELS_DIR)
    print(f"\nSaved {len(saved_paths)} model artifacts to {SAVED_MODELS_DIR}")

    # 6. Evaluate all models on test set
    print("\nEvaluating model performance on test set (3,529 samples)...")
    eval_results = []

    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "naive_bayes": "Multinomial Naive Bayes",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    for name, model in models_dict.items():
        if model is not None:
            disp_name = model_display_names.get(name, name)
            # Use non-negative test matrix for Naive Bayes, standard matrix for others
            test_matrix = X_test_nonneg if name == "naive_bayes" else X_test_trans
            res = evaluate_single_model(model, test_matrix, y_test, disp_name)
            eval_results.append(res)

    # 7. Generate evaluation plots (Confusion Matrices, ROC, PR curves)
    print("Generating evaluation plots...")
    generated_plots = generate_all_evaluation_plots(eval_results)

    # 8. Extract Logistic Regression feature coefficients
    feature_names = get_feature_names_from_pipeline(pipeline)
    lr_features_df = None
    if "logistic_regression" in models_dict and models_dict["logistic_regression"] is not None:
        print("Extracting Logistic Regression feature coefficients...")
        lr_features_df = extract_logistic_regression_features(
            models_dict["logistic_regression"],
            feature_names,
            output_csv=LOGISTIC_FEATS_CSV,
            top_n=25,
        )

    # 9. Generate comparison table CSV and Markdown
    csv_comp, md_comp = generate_comparison_reports(eval_results)

    # 10. Generate full Markdown report
    generate_markdown_report(eval_results, len(y_train), len(y_test), EVALUATION_REPORT_MD)

    # 11. Print concise terminal summary
    print_final_terminal_summary(
        eval_results=eval_results,
        n_train=len(y_train),
        n_test=len(y_test),
        saved_paths=saved_paths,
        report_path=EVALUATION_REPORT_MD,
    )

    return {
        "models": models_dict,
        "eval_results": eval_results,
        "saved_paths": saved_paths,
        "comparison_csv": csv_comp,
        "comparison_md": md_comp,
        "report_md": EVALUATION_REPORT_MD,
        "lr_features": lr_features_df,
    }


def generate_markdown_report(
    eval_results: List[Dict[str, Any]],
    n_train: int,
    n_test: int,
    report_path: Path,
):
    """
    Generates human-readable Markdown evaluation report (ml/evaluation/model_evaluation_report.md).
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# TrustHire — Step 4 Model Training & Evaluation Report",
        "",
        "## 1. Executive Summary",
        "- **Dataset Used**: EMSCAD Raw Job Postings (`fake_job_postings.csv`)",
        f"- **Training Dataset Size**: {n_train:,} samples (80% Stratified Split)",
        f"- **Testing Dataset Size**: {n_test:,} samples (20% Stratified Split)",
        "- **Positive Target Class**: `1` = **Potentially Fraudulent** (~4.87% of test set)",
        "- **Negative Target Class**: `0` = **Legitimate** (~95.13% of test set)",
        "",
        "## 2. Models Evaluated",
        "1. **Logistic Regression** (Primary candidate, `class_weight='balanced'`)",
        "2. **Linear SVM** (`LinearSVC`, `class_weight='balanced'`)",
        "3. **Multinomial Naive Bayes** (Trained on non-negative scaled feature matrix)",
        "4. **Random Forest** (`RandomForestClassifier`, `class_weight='balanced'`)",
        "5. **XGBoost** (" + ("Evaluated" if XGBOOST_AVAILABLE else "Unavailable - Package not installed") + ")",
        "",
        "## 3. Comparative Performance Metrics Table",
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    # Sort results by PR-AUC descending
    sorted_res = sorted(eval_results, key=lambda x: x["pr_auc"], reverse=True)
    for r in sorted_res:
        lines.append(
            f"| **{r['model_name']}** | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1_score']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Confusion Matrix Summary",
        "| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |",
        "| --- | --- | --- | --- | --- |",
    ])

    for r in sorted_res:
        cm = r["confusion_matrix"]
        lines.append(
            f"| **{r['model_name']}** | {cm['TP']} | {cm['FP']} | {cm['FN']} | {cm['TN']} |"
        )

    lines.extend([
        "",
        "## 5. Model Signals & Interpretability (Logistic Regression)",
        "- **Positive Coefficient Signals (Model-associated Fraud Indicators)**: Features whose presence increases the log-odds of a posting being flagged as potentially fraudulent (e.g. missing company profiles, specific keywords/phrases, unverified contact methods).",
        "- **Negative Coefficient Signals (Model-associated Legitimate Indicators)**: Features whose presence increases the log-odds of a posting being flagged as legitimate (e.g. verified company logo, detailed requirement sections, established industry metadata).",
        "- **Important Precaution**: Feature coefficients represent statistical correlations in historical data, NOT absolute proof of fraud.",
        "",
        "## 6. Observations & Trade-Offs",
        "- **Imbalance Impact**: Standard accuracy (95%+) is misleading because a naive dummy model predicting all legitimate achieves ~95.1% accuracy with 0% recall.",
        "- **Precision vs. Recall Trade-Off**:",
        "  - High Recall models capture nearly all fraudulent postings (low False Negatives) but produce more False Positives (Needs Caution).",
        "  - High Precision models produce very few false alarms but risk missing subtle fraudulent postings.",
        "- **PR-AUC Importance**: Precision-Recall AUC is the most informative metric for severe class imbalance.",
        "",
        "## 7. Next Recommended Development Step",
        "- Proceed to Step 5: System integration, hyperparameter selection, FastAPI API endpoint, and simple frontend display.",
        "",
        "---",
        "*Report generated programmatically during Step 4 Model Evaluation.*",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_final_terminal_summary(
    eval_results: List[Dict[str, Any]],
    n_train: int,
    n_test: int,
    saved_paths: Dict[str, Path],
    report_path: Path,
):
    """
    Displays the formatted concise summary required by Section 23 of prompt.
    """
    models_str = ", ".join([r["model_name"] for r in eval_results])

    comp_lines = ["Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC", "--- | --- | --- | --- | --- | --- | ---"]
    sorted_res = sorted(eval_results, key=lambda x: x["pr_auc"], reverse=True)
    for r in sorted_res:
        comp_lines.append(
            f"{r['model_name']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f}"
        )
    comp_table_str = "\n".join(comp_lines)

    top_model = sorted_res[0] if sorted_res else None
    top_obs = f"Highest PR-AUC achieved by {top_model['model_name']} ({top_model['pr_auc']:.4f}) with Recall {top_model['recall']:.4f} and Precision {top_model['precision']:.4f}." if top_model else "N/A"

    saved_str = "\n".join([f"- ml/saved_models/{p.name}" for p in saved_paths.values()])

    summary = f"""
==================================================
TRUSTHIRE — MODEL EVALUATION
==================================================

Dataset:
EMSCAD (fake_job_postings.csv)

Training samples:
{n_train:,}

Testing samples:
{n_test:,}

Models successfully trained:
{models_str}

Model comparison:

{comp_table_str}

Important observation:
{top_obs}

Candidate Metrics Breakdown:
"""
    for r in eval_results:
        summary += f"\n{r['model_name']}:\n  Accuracy: {r['accuracy']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1_score']:.4f} | ROC-AUC: {r['roc_auc']:.4f} | PR-AUC: {r['pr_auc']:.4f}\n"

    summary += f"""
Saved models:
{saved_str}

Evaluation plots:
- ml/evaluation/plots/*_confusion_matrix.png
- ml/evaluation/plots/roc_curve_comparison.png
- ml/evaluation/plots/pr_curve_comparison.png

Evaluation report:
{report_path}

==================================================
"""
    print(summary)


if __name__ == "__main__":
    run_full_model_evaluation()
