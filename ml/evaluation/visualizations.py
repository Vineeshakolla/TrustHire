"""
TrustHire - Evaluation Visualizations Module
Generates plots for Confusion Matrices, combined ROC-AUC curves,
and combined Precision-Recall curves.
"""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from backend.app.config import ML_DIR

PLOTS_DIR = ML_DIR / "evaluation" / "plots"


def plot_confusion_matrix(eval_result: Dict[str, Any], output_dir: Path = PLOTS_DIR) -> Path:
    """
    Plots and saves a confusion matrix for a trained candidate model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = eval_result["model_name"]
    file_stub = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"{file_stub}_confusion_matrix.png"

    cm = eval_result["cm_raw"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_theme(style="dark")

    labels = ["Legitimate", "Potentially\nFraudulent"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)

    # Add text annotations for TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, 0.25, f"TN: {tn}", ha="center", va="center", color="black", fontsize=10)
    ax.text(1.5, 0.25, f"FP: {fp}", ha="center", va="center", color="red", fontsize=10)
    ax.text(0.5, 1.25, f"FN: {fn}", ha="center", va="center", color="red", fontsize=10)
    ax.text(1.5, 1.25, f"TP: {tp}", ha="center", va="center", color="green", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_roc_curves_comparison(eval_results: List[Dict[str, Any]], output_dir: Path = PLOTS_DIR) -> Path:
    """
    Plots combined ROC curves for all candidate models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roc_curve_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="darkgrid")

    for r in eval_results:
        fpr, tpr = r["roc_curve"]
        auc = r["roc_auc"]
        ax.plot(fpr, tpr, label=f"{r['model_name']} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", linewidth=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("TrustHire: Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_pr_curves_comparison(eval_results: List[Dict[str, Any]], output_dir: Path = PLOTS_DIR) -> Path:
    """
    Plots combined Precision-Recall curves for all candidate models.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pr_curve_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="darkgrid")

    for r in eval_results:
        prec, rec = r["pr_curve"]
        pr_auc = r["pr_auc"]
        ax.plot(rec, prec, label=f"{r['model_name']} (PR-AUC = {pr_auc:.4f})", linewidth=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("TrustHire: Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def generate_all_evaluation_plots(eval_results: List[Dict[str, Any]], output_dir: Path = PLOTS_DIR) -> List[Path]:
    """
    Generates all confusion matrix plots, combined ROC curves plot, and combined PR curves plot.
    """
    generated = []

    for r in eval_results:
        cm_path = plot_confusion_matrix(r, output_dir)
        generated.append(cm_path)

    roc_path = plot_roc_curves_comparison(eval_results, output_dir)
    generated.append(roc_path)

    pr_path = plot_pr_curves_comparison(eval_results, output_dir)
    generated.append(pr_path)

    return generated
