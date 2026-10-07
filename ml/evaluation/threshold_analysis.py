"""
TrustHire - Threshold Analysis Module
Evaluates candidate model performance across decision thresholds (0.10 to 0.90),
calculating Precision, Recall, F1-score, False Positives, and False Negatives.
Generates threshold_analysis.csv and threshold_analysis.png plot.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from backend.app.config import ML_DIR, SAVED_MODELS_DIR, PROCESSED_DATA_DIR
from ml.training.model_utils import load_processed_data, load_fitted_pipeline, transform_data_with_pipeline
import joblib

EVALUATION_DIR = ML_DIR / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"


def run_threshold_analysis(
    thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Evaluates candidate models across decision thresholds.
    Saves ml/evaluation/threshold_analysis.csv and ml/evaluation/plots/threshold_analysis.png.
    """
    X_train_df, X_test_df, y_train, y_test = load_processed_data(PROCESSED_DATA_DIR)
    pipeline = load_fitted_pipeline(SAVED_MODELS_DIR)
    X_train_trans, X_test_trans = transform_data_with_pipeline(pipeline, X_train_df, X_test_df)

    # Load trained models
    lr_model = joblib.load(SAVED_MODELS_DIR / "logistic_regression.joblib")
    svm_model = joblib.load(SAVED_MODELS_DIR / "linear_svm.joblib")
    xgb_path = SAVED_MODELS_DIR / "xgboost.joblib"
    xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None

    # Models with probabilities or continuous decision functions
    eval_models = [
        ("Logistic Regression", lr_model, "proba"),
        ("Linear SVM", svm_model, "decision"),
    ]
    if xgb_model is not None:
        eval_models.append(("XGBoost", xgb_model, "proba"))

    records = []

    for model_name, model, score_type in eval_models:
        if score_type == "proba":
            scores = model.predict_proba(X_test_trans)[:, 1]
        else:
            # Linear SVM decision function: MinMax scale to [0, 1] for threshold evaluation
            d_scores = model.decision_function(X_test_trans)
            scores = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-9)

        for th in thresholds:
            y_pred_th = (scores >= th).astype(int)

            prec = precision_score(y_test, y_pred_th, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred_th, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred_th, pos_label=1, zero_division=0)

            cm = confusion_matrix(y_test, y_pred_th, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            records.append({
                "model_name": model_name,
                "threshold": round(th, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            })

    df_th = pd.DataFrame(records)

    # Save CSV
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVALUATION_DIR / "threshold_analysis.csv"
    df_th.to_csv(csv_path, index=False)

    # Plot visualization
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "threshold_analysis.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="darkgrid")

    for model_name in df_th["model_name"].unique():
        sub_df = df_th[df_th["model_name"] == model_name]
        ax.plot(sub_df["threshold"], sub_df["precision"], marker="o", linestyle="-", label=f"{model_name} Precision")
        ax.plot(sub_df["threshold"], sub_df["recall"], marker="s", linestyle="--", label=f"{model_name} Recall")

    ax.set_title("TrustHire: Precision vs Recall Across Decision Thresholds", fontsize=13, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xticks(thresholds)
    ax.set_ylim([0.0, 1.05])
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    return df_th, csv_path, plot_path


if __name__ == "__main__":
    df_th, csv_p, plot_p = run_threshold_analysis()
    print("Threshold analysis complete.")
    print(df_th.to_string())
