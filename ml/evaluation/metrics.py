"""
TrustHire - Evaluation Metrics & Interpretability Module
Calculates Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix,
and extracts Logistic Regression feature signals.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from backend.app.config import ML_DIR

EVALUATION_DIR = ML_DIR / "evaluation"


def evaluate_single_model(
    model: Any,
    X_test,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluates a single model on the test dataset.
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)

    # Get scores for ROC-AUC & PR-AUC
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred

    # Calculate metrics for positive class (1 = Fraudulent)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_test, y_scores))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate ROC and PR curves data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_scores)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "cm_raw": cm,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision_curve, recall_curve),
    }


def extract_logistic_regression_features(
    model: Any,
    feature_names: List[str],
    output_csv: Optional[Path] = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Extracts feature coefficients from Logistic Regression.
    Identifies top positive (fraudulent signal) and negative (legitimate signal) features.
    """
    if not hasattr(model, "coef_"):
        raise AttributeError("Provided model does not contain coef_ attribute.")

    coefs = model.coef_[0]
    if len(coefs) != len(feature_names):
        # Fallback if length mismatch
        feature_names = [f"feature_{i}" for i in range(len(coefs))]

    df_feats = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    })

    df_feats["direction"] = df_feats["coefficient"].apply(
        lambda c: "fraudulent_signal" if c > 0 else "legitimate_signal"
    )

    df_feats = df_feats.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        export_df = df_feats[["feature", "coefficient", "direction"]].head(top_n * 2)
        export_df.to_csv(output_csv, index=False)

    return df_feats


def generate_comparison_reports(
    eval_results: List[Dict[str, Any]],
    output_dir: Path = EVALUATION_DIR,
) -> Tuple[Path, Path]:
    """
    Generates model_comparison.csv and model_comparison.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison.csv"
    md_path = output_dir / "model_comparison.md"

    table_data = []
    for r in eval_results:
        table_data.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"],
            "PR-AUC": r["pr_auc"],
            "True Positives (TP)": r["confusion_matrix"]["TP"],
            "False Positives (FP)": r["confusion_matrix"]["FP"],
            "False Negatives (FN)": r["confusion_matrix"]["FN"],
            "True Negatives (TN)": r["confusion_matrix"]["TN"],
        })

    df_comp = pd.DataFrame(table_data).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    df_comp.to_csv(csv_path, index=False)

    # Generate Markdown table
    md_lines = [
        "# TrustHire — Candidate Model Comparison",
        "",
        "## Performance Metrics Table (Sorted by PR-AUC)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in df_comp.to_dict(orient="records"):
        md_lines.append(
            f"| **{row['Model']}** | {row['Accuracy']:.4f} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} | {row['PR-AUC']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Detailed Confusion Matrix Breakdown",
        "",
        "| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |",
        "| --- | --- | --- | --- | --- |",
    ])

    for row in df_comp.to_dict(orient="records"):
        md_lines.append(
            f"| **{row['Model']}** | {row['True Positives (TP)']} | {row['False Positives (FP)']} | "
            f"{row['False Negatives (FN)']} | {row['True Negatives (TN)']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return csv_path, md_path
