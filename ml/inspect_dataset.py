"""
TrustHire - EMSCAD Dataset Inspection & Exploratory Data Analysis (EDA) Module
Provides modular, reusable functions for loading, inspecting, analyzing, and generating reports for the raw EMSCAD dataset.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config import (
    RAW_DATA_DIR,
    EDA_OUTPUT_DIR,
    DATASET_REPORT_JSON,
    DATASET_REPORT_MD,
)


# Expected EMSCAD Attributes
EXPECTED_EMSCAD_COLUMNS = [
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
]

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
SALARY_COLUMN = "salary_range"
ID_COLUMN = "job_id"
TARGET_COLUMN = "fraudulent"


def find_raw_dataset(raw_dir: Optional[Path] = None) -> Path:
    """
    Locates the raw CSV dataset inside ml/data/raw/.
    Raises FileNotFoundError if no CSV file is present.
    """
    target_dir = raw_dir or RAW_DATA_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {target_dir}")

    csv_files = list(target_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset file found in '{target_dir}'. "
            "Please place the EMSCAD dataset CSV file inside ml/data/raw/."
        )

    # Return the first found CSV file
    return csv_files[0]


def load_dataset(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, Path]:
    """
    Loads the CSV dataset into a Pandas DataFrame without modifying the raw file.
    """
    actual_path = filepath or find_raw_dataset()
    df = pd.read_csv(actual_path)
    return df, actual_path


def inspect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects column schema, data types, and compares against expected EMSCAD attributes.
    """
    actual_cols = list(df.columns)
    expected_found = [col for col in EXPECTED_EMSCAD_COLUMNS if col in actual_cols]
    expected_missing = [col for col in EXPECTED_EMSCAD_COLUMNS if col not in actual_cols]
    unexpected_present = [col for col in actual_cols if col not in EXPECTED_EMSCAD_COLUMNS]

    schema_status = "PASS" if not expected_missing and not unexpected_present else "WARN"

    return {
        "status": schema_status,
        "total_columns": len(actual_cols),
        "actual_columns": actual_cols,
        "column_dtypes": {col: str(df[col].dtype) for col in actual_cols},
        "expected_columns_found": expected_found,
        "expected_columns_missing": expected_missing,
        "unexpected_columns_present": unexpected_present,
    }


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing count and missing percentage for every column.
    Returns a DataFrame sorted by missing percentage descending.
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_rows) * 100

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.round(2).values,
    }).sort_values(by="Missing %", ascending=False).reset_index(drop=True)

    return missing_df


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks exact duplicate rows and duplicate job_id values.
    """
    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    exact_duplicate_pct = round((exact_duplicates / total_rows) * 100, 2) if total_rows > 0 else 0.0

    duplicate_job_ids = 0
    duplicate_job_id_pct = 0.0

    if ID_COLUMN in df.columns:
        duplicate_job_ids = int(df[ID_COLUMN].duplicated().sum())
        duplicate_job_id_pct = round((duplicate_job_ids / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "total_exact_duplicates": exact_duplicates,
        "exact_duplicate_percentage": exact_duplicate_pct,
        "total_duplicate_job_ids": duplicate_job_ids,
        "duplicate_job_id_percentage": duplicate_job_id_pct,
    }


def analyze_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
    """
    Analyzes target column (fraudulent).
    """
    if target_col not in df.columns:
        return {"exists": False}

    total_rows = len(df)
    counts = df[target_col].value_counts(dropna=False).to_dict()
    percentages = {
        k: round((v / total_rows) * 100, 2) for k, v in counts.items()
    }

    # EMSCAD standard: 0 -> Legitimate, 1 -> Potentially Fraudulent / Fraudulent
    mapping = {
        0: "Legitimate (Possibly Legitimate)",
        1: "Potentially Fraudulent (Higher Risk)",
    }

    return {
        "exists": True,
        "column_name": target_col,
        "dtype": str(df[target_col].dtype),
        "unique_values": [int(x) if isinstance(x, (int, np.integer)) else x for x in df[target_col].unique()],
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "percentages": {str(k): float(v) for k, v in percentages.items()},
        "label_mapping": mapping,
        "is_imbalanced": (percentages.get("1", 0) < 20 or percentages.get(1, 0) < 20),
    }


def analyze_text_features(df: pd.DataFrame, text_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculates statistics (character lengths & word counts) for text features.
    """
    cols = text_cols or [c for c in TEXT_COLUMNS if c in df.columns]
    results = {}

    for col in cols:
        series = df[col].astype(str).fillna("")
        # Empty if NaN or empty string or whitespace
        is_empty = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        empty_count = int(is_empty.sum())
        non_empty_count = len(df) - empty_count

        char_lengths = series.apply(lambda x: len(x) if x else 0)
        word_counts = series.apply(lambda x: len(x.split()) if x else 0)

        # For non-empty strings specifically:
        non_empty_char_lengths = char_lengths[~is_empty]
        non_empty_word_counts = word_counts[~is_empty]

        results[col] = {
            "non_empty_records": non_empty_count,
            "empty_records": empty_count,
            "char_len_min": int(non_empty_char_lengths.min()) if not non_empty_char_lengths.empty else 0,
            "char_len_max": int(non_empty_char_lengths.max()) if not non_empty_char_lengths.empty else 0,
            "char_len_mean": round(float(non_empty_char_lengths.mean()), 2) if not non_empty_char_lengths.empty else 0.0,
            "char_len_median": float(non_empty_char_lengths.median()) if not non_empty_char_lengths.empty else 0.0,
            "word_count_min": int(non_empty_word_counts.min()) if not non_empty_word_counts.empty else 0,
            "word_count_max": int(non_empty_word_counts.max()) if not non_empty_word_counts.empty else 0,
            "word_count_mean": round(float(non_empty_word_counts.mean()), 2) if not non_empty_word_counts.empty else 0.0,
            "word_count_median": float(non_empty_word_counts.median()) if not non_empty_word_counts.empty else 0.0,
        }

    return results


def analyze_categorical_features(df: pd.DataFrame, cat_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyzes categorical column unique counts and top frequent values.
    """
    cols = cat_cols or [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    results = {}

    for col in cols:
        unique_count = int(df[col].nunique(dropna=True))
        missing_count = int(df[col].isnull().sum())
        top_values = df[col].value_counts(dropna=True).head(5).to_dict()

        results[col] = {
            "unique_values_count": unique_count,
            "missing_count": missing_count,
            "top_5_frequent_values": {str(k): int(v) for k, v in top_values.items()},
        }

    return results


def analyze_binary_features(df: pd.DataFrame, bin_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Inspects binary indicators.
    """
    cols = bin_cols or [c for c in BINARY_COLUMNS if c in df.columns]
    results = {}
    total_rows = len(df)

    for col in cols:
        counts = df[col].value_counts(dropna=False).to_dict()
        percentages = {str(k): round((v / total_rows) * 100, 2) for k, v in counts.items()}

        results[col] = {
            "unique_values": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in df[col].unique()],
            "value_counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": percentages,
            "is_strictly_binary": set(df[col].dropna().unique()).issubset({0, 1}),
        }

    return results


def analyze_salary(df: pd.DataFrame, salary_col: str = SALARY_COLUMN) -> Dict[str, Any]:
    """
    Inspects the salary_range column.
    """
    if salary_col not in df.columns:
        return {"exists": False}

    total_rows = len(df)
    missing_count = int(df[salary_col].isnull().sum())
    non_empty_count = total_rows - missing_count
    unique_count = int(df[salary_col].nunique(dropna=True))
    sample_values = list(df[salary_col].dropna().unique()[:5])

    return {
        "exists": True,
        "dtype": str(df[salary_col].dtype),
        "missing_count": missing_count,
        "missing_percentage": round((missing_count / total_rows) * 100, 2),
        "non_empty_count": non_empty_count,
        "unique_values_count": unique_count,
        "sample_real_examples": [str(v) for v in sample_values],
    }


def generate_visualizations(df: pd.DataFrame, output_dir: Optional[Path] = None) -> List[Path]:
    """
    Generates EDA plots and saves them inside ml/data/eda_outputs/.
    """
    out_dir = output_dir or EDA_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Styling settings
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("mako")

    # 1. Target Distribution Plot
    if TARGET_COLUMN in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        target_counts = df[TARGET_COLUMN].value_counts().rename({0: "Legitimate (0)", 1: "Fraudulent (1)"})
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"], hue=target_counts.index, legend=False)
        ax.set_title("TrustHire: Target Variable Distribution (fraudulent)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Posting Category")
        ax.set_ylabel("Count")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{int(height)} ({height/len(df):.1%})",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
        plt.tight_layout()
        p1 = out_dir / "target_distribution.png"
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        generated_files.append(p1)

    # 2. Missing Values Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if not missing.empty:
        sns.barplot(x=missing.values / len(df) * 100, y=missing.index, ax=ax, palette=palette, hue=missing.index, legend=False)

        ax.set_title("TrustHire: Missing Values Percentage by Column", fontsize=13, fontweight='bold')
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_ylabel("Column Name")
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.1f}%", (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9, xytext=(5, 0), textcoords='offset points')
    else:
        ax.text(0.5, 0.5, "No missing values found", ha='center', va='center', fontsize=12)
    plt.tight_layout()
    p2 = out_dir / "missing_values.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    generated_files.append(p2)

    # 3. Text Length Distribution Plot
    text_cols_present = [c for c in TEXT_COLUMNS if c in df.columns]
    if text_cols_present:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in text_cols_present:
            lengths = df[c].fillna("").astype(str).apply(len)
            sns.kdeplot(lengths, ax=ax, label=c, log_scale=True)
        ax.set_title("TrustHire: Text Fields Character Length Distribution (Log Scale)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Character Length (Log Scale)")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        p3 = out_dir / "text_length_distribution.png"
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        generated_files.append(p3)

    # 4. Employment Type Distribution
    if "employment_type" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        emp_counts = df["employment_type"].fillna("Unspecified").value_counts()
        sns.barplot(x=emp_counts.values, y=emp_counts.index, ax=ax, palette="Blues_r", hue=emp_counts.index, legend=False)
        ax.set_title("TrustHire: Employment Type Distribution", fontsize=13, fontweight='bold')
        ax.set_xlabel("Count")
        ax.set_ylabel("Employment Type")
        plt.tight_layout()
        p4 = out_dir / "employment_type_distribution.png"
        fig.savefig(p4, dpi=300)
        plt.close(fig)
        generated_files.append(p4)

    # 5. Required Experience Distribution
    if "required_experience" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        exp_counts = df["required_experience"].fillna("Unspecified").value_counts()
        sns.barplot(x=exp_counts.values, y=exp_counts.index, ax=ax, palette="viridis", hue=exp_counts.index, legend=False)
        ax.set_title("TrustHire: Required Experience Distribution", fontsize=13, fontweight='bold')
        ax.set_xlabel("Count")
        ax.set_ylabel("Experience Level")
        plt.tight_layout()
        p5 = out_dir / "required_experience_distribution.png"
        fig.savefig(p5, dpi=300)
        plt.close(fig)
        generated_files.append(p5)

    # 6. Top Industries Distribution
    if "industry" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        ind_counts = df["industry"].fillna("Unspecified").value_counts().head(10)
        sns.barplot(x=ind_counts.values, y=ind_counts.index, ax=ax, palette="Purples_r", hue=ind_counts.index, legend=False)
        ax.set_title("TrustHire: Top 10 Job Posting Industries", fontsize=13, fontweight='bold')
        ax.set_xlabel("Count")
        ax.set_ylabel("Industry")
        plt.tight_layout()
        p6 = out_dir / "top_industries.png"
        fig.savefig(p6, dpi=300)
        plt.close(fig)
        generated_files.append(p6)

    # 7. Company Logo Distribution
    if "has_company_logo" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        logo_counts = df["has_company_logo"].value_counts().rename({1: "Logo Present", 0: "No Logo"})
        sns.barplot(x=logo_counts.index, y=logo_counts.values, ax=ax, palette="Set2", hue=logo_counts.index, legend=False)
        ax.set_title("TrustHire: Company Logo Availability", fontsize=12, fontweight='bold')
        ax.set_xlabel("Status")
        ax.set_ylabel("Count")
        plt.tight_layout()
        p7 = out_dir / "company_logo_distribution.png"
        fig.savefig(p7, dpi=300)
        plt.close(fig)
        generated_files.append(p7)

    # 8. Telecommuting Distribution
    if "telecommuting" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        tele_counts = df["telecommuting"].value_counts().rename({1: "Telecommuting (Remote)", 0: "On-Site"})
        sns.barplot(x=tele_counts.index, y=tele_counts.values, ax=ax, palette="Set3", hue=tele_counts.index, legend=False)
        ax.set_title("TrustHire: Telecommuting Job Postings", fontsize=12, fontweight='bold')
        ax.set_xlabel("Work Type")
        ax.set_ylabel("Count")
        plt.tight_layout()
        p8 = out_dir / "telecommuting_distribution.png"
        fig.savefig(p8, dpi=300)
        plt.close(fig)
        generated_files.append(p8)


    return generated_files


def generate_dataset_report(
    df: pd.DataFrame,
    filepath: Path,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Generates both machine-readable JSON and human-readable Markdown dataset reports.
    """
    j_path = json_path or DATASET_REPORT_JSON
    m_path = md_path or DATASET_REPORT_MD

    j_path.parent.mkdir(parents=True, exist_ok=True)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    schema_info = inspect_schema(df)
    missing_df = analyze_missing_values(df)
    dup_info = analyze_duplicates(df)
    target_info = analyze_target(df)
    text_info = analyze_text_features(df)
    cat_info = analyze_categorical_features(df)
    bin_info = analyze_binary_features(df)
    salary_info = analyze_salary(df)

    # Compile JSON report
    report_data = {
        "project": "TrustHire",
        "step": "Step 2 - EMSCAD Dataset Inspection & EDA",
        "dataset_filename": filepath.name,
        "dataset_relative_path": str(filepath.relative_to(filepath.parents[2])),
        "rows": len(df),
        "columns": len(df.columns),
        "schema_status": schema_info["status"],
        "schema": schema_info,
        "missing_values": missing_df.to_dict(orient="records"),
        "duplicates": dup_info,
        "target_analysis": target_info,
        "salary_analysis": salary_info,
        "text_feature_analysis": text_info,
        "categorical_feature_analysis": cat_info,
        "binary_feature_analysis": bin_info,
    }

    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    # Compile Markdown report
    md_lines = [
        "# TrustHire — EMSCAD Dataset Inspection & EDA Report",
        "",
        "## 1. Overview",
        f"- **Dataset Filename**: `{filepath.name}`",
        f"- **Dataset Relative Path**: `{filepath.relative_to(filepath.parents[2])}`",
        f"- **Total Rows**: {len(df):,}",
        f"- **Total Columns**: {len(df.columns)}",
        f"- **Schema Validation Status**: **{schema_info['status']}**",
        "",
        "## 2. Target Variable Analysis (`fraudulent`)",
    ]

    if target_info.get("exists"):
        md_lines.extend([
            f"- **Data Type**: `{target_info['dtype']}`",
            f"- **Unique Values**: `{target_info['unique_values']}`",
            "- **Value Counts & Distribution**:",
            f"  - `0` (Legitimate): {target_info['value_counts'].get('0', 0):,} ({target_info['percentages'].get('0', 0)}%)",
            f"  - `1` (Potentially Fraudulent): {target_info['value_counts'].get('1', 0):,} ({target_info['percentages'].get('1', 0)}%)",
            f"- **Class Imbalance Identified**: {'Yes (Imbalanced Dataset)' if target_info['is_imbalanced'] else 'No'}",
        ])
    else:
        md_lines.append("- **Target Column (`fraudulent`) NOT FOUND in dataset!**")

    md_lines.extend([
        "",
        "## 3. Duplicate Analysis",
        f"- **Exact Duplicate Rows**: {dup_info['total_exact_duplicates']} ({dup_info['exact_duplicate_percentage']}%)",
        f"- **Duplicate Job IDs**: {dup_info['total_duplicate_job_ids']} ({dup_info['duplicate_job_id_percentage']}%)",
        "",
        "## 4. Missing Values Analysis",
        "| Column | Data Type | Missing Count | Missing % |",
        "| --- | --- | --- | --- |",
    ])

    for row in missing_df.to_dict(orient="records"):
        md_lines.append(f"| `{row['Column']}` | `{row['Data Type']}` | {row['Missing Count']:,} | {row['Missing %']}% |")

    md_lines.extend([
        "",
        "## 5. Text Features Summary",
        "| Column | Non-Empty Count | Empty Count | Char Length (Mean) | Word Count (Mean) |",
        "| --- | --- | --- | --- | --- |",
    ])

    for col, stat in text_info.items():
        md_lines.append(
            f"| `{col}` | {stat['non_empty_records']:,} | {stat['empty_records']:,} | {stat['char_len_mean']} | {stat['word_count_mean']} |"
        )

    md_lines.extend([
        "",
        "## 6. Categorical Features Summary",
        "| Column | Unique Count | Missing Count | Top Frequent Value |",
        "| --- | --- | --- | --- |",
    ])

    for col, stat in cat_info.items():
        top_val = list(stat['top_5_frequent_values'].keys())[0] if stat['top_5_frequent_values'] else "N/A"
        top_count = list(stat['top_5_frequent_values'].values())[0] if stat['top_5_frequent_values'] else 0
        md_lines.append(f"| `{col}` | {stat['unique_values_count']:,} | {stat['missing_count']:,} | `{top_val}` ({top_count:,}) |")

    md_lines.extend([
        "",
        "## 7. Salary Analysis (`salary_range`)",
    ])
    if salary_info.get("exists"):
        md_lines.extend([
            f"- **Missing Count**: {salary_info['missing_count']:,} ({salary_info['missing_percentage']}%)",
            f"- **Unique Values Count**: {salary_info['unique_values_count']:,}",
            f"- **Sample Values**: `{salary_info['sample_real_examples']}`",
        ])

    md_lines.extend([
        "",
        "---",
        "*Report generated programmatically during Step 2 EMSCAD Dataset Inspection.*",
    ])

    with open(m_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return j_path, m_path


def print_terminal_summary(
    df: pd.DataFrame,
    filepath: Path,
    target_info: Dict[str, Any],
    missing_df: pd.DataFrame,
    dup_info: Dict[str, Any],
    schema_info: Dict[str, Any],
    eda_path: Path,
    report_path: Path,
):
    """
    Displays the formatted concise summary required by Step 17 of prompt.
    """
    rel_path = filepath.relative_to(filepath.parents[2]) if len(filepath.parents) >= 3 else filepath

    top_missing_str = ", ".join(
        [f"{r['Column']} ({r['Missing %']}%)" for r in missing_df.head(5).to_dict(orient="records")]
    )

    t_0 = target_info.get("value_counts", {}).get("0", 0)
    t_1 = target_info.get("value_counts", {}).get("1", 0)
    pct_0 = target_info.get("percentages", {}).get("0", 0.0)
    pct_1 = target_info.get("percentages", {}).get("1", 0.0)

    summary = f"""
==================================================
TRUSTHIRE — EMSCAD DATASET REPORT
==================================================

Dataset:
{filepath.name} ({rel_path})

Rows:
{len(df):,}

Columns:
{len(df.columns)}

Target:
{TARGET_COLUMN}

Class distribution:
0 (Legitimate): {t_0:,} ({pct_0}%)
1 (Potentially Fraudulent): {t_1:,} ({pct_1}%)

Missing values (Top 5):
{top_missing_str}

Exact duplicates:
{dup_info['total_exact_duplicates']:,} ({dup_info['exact_duplicate_percentage']}%)

Duplicate IDs:
{dup_info['total_duplicate_job_ids']:,} ({dup_info['duplicate_job_id_percentage']}%)

Schema status:
{schema_info['status']}

EDA outputs:
{eda_path}

Dataset report:
{report_path}

==================================================
"""
    print(summary)


def run_inspection() -> Dict[str, Any]:
    """
    Main orchestration function for dataset inspection and EDA.
    """
    df, csv_path = load_dataset()

    schema_info = inspect_schema(df)
    missing_df = analyze_missing_values(df)
    dup_info = analyze_duplicates(df)
    target_info = analyze_target(df)

    eda_files = generate_visualizations(df)
    j_path, m_path = generate_dataset_report(df, csv_path)

    print_terminal_summary(
        df=df,
        filepath=csv_path,
        target_info=target_info,
        missing_df=missing_df,
        dup_info=dup_info,
        schema_info=schema_info,
        eda_path=EDA_OUTPUT_DIR,
        report_path=m_path,
    )

    return {
        "df": df,
        "filepath": csv_path,
        "schema": schema_info,
        "missing": missing_df,
        "duplicates": dup_info,
        "target": target_info,
        "eda_files": eda_files,
        "json_report": j_path,
        "md_report": m_path,
    }


if __name__ == "__main__":
    run_inspection()
