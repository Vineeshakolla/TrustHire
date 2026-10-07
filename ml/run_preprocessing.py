"""
TrustHire - Complete Data Preprocessing & Pipeline Execution Module
Orchestrates raw dataset cleaning, target transformation, feature derivation,
stratified train/test data splitting, and fitting the leakage-safe scikit-learn pipeline.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    ML_DIR,
)

from ml.preprocessing.cleaning import (
    remove_exact_duplicates,
    convert_target,
    convert_binary_columns,
    apply_salary_features,
)
from ml.preprocessing.text_processing import extract_text_derived_features
from ml.preprocessing.split_data import split_dataset, save_processed_splits
from ml.preprocessing.feature_pipeline import build_preprocessing_pipeline


PREPROCESSING_REPORT_MD = ML_DIR / "data" / "preprocessing_report.md"


def run_full_preprocessing() -> Dict[str, Any]:
    """
    Executes the complete Step 3 preprocessing pipeline.
    """
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV raw dataset found in {RAW_DATA_DIR}")

    raw_path = csv_files[0]
    df_raw = pd.read_csv(raw_path)

    # 1. Remove exact duplicates
    df_dedup, dup_stats = remove_exact_duplicates(df_raw)

    # 2. Convert target column
    df_target = convert_target(df_dedup, target_col="fraudulent")

    # 3. Convert binary columns
    df_binary = convert_binary_columns(
        df_target, binary_cols=["telecommuting", "has_company_logo", "has_questions"]
    )

    # 4. Parse salary range
    df_salary = apply_salary_features(df_binary, salary_col="salary_range")

    # 5. Extract text derived features & clean text fields
    df_processed = extract_text_derived_features(df_salary)

    # 6. Stratified train/test split (split BEFORE fitting any encoder or vectorizer)
    X_train, X_test, y_train, y_test, split_stats = split_dataset(
        df_processed, target_col="fraudulent", test_size=0.2, random_state=42
    )

    # 7. Save processed splits to ml/data/processed/
    train_csv, test_csv = save_processed_splits(X_train, X_test, y_train, y_test)

    # 8. Build & Fit Scikit-Learn Preprocessing Pipeline ONLY on X_train
    pipeline = build_preprocessing_pipeline(tfidf_max_features=5000)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # 9. Save fitted pipeline for future model training & prediction
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = SAVED_MODELS_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    # 10. Generate Preprocessing Report
    generate_preprocessing_report(
        raw_filename=raw_path.name,
        dup_stats=dup_stats,
        split_stats=split_stats,
        X_train_shape=X_train_transformed.shape,
        X_test_shape=X_test_transformed.shape,
        report_path=PREPROCESSING_REPORT_MD,
    )

    # Print concise final summary
    print_final_summary(
        raw_filename=raw_path.name,
        dup_stats=dup_stats,
        split_stats=split_stats,
        X_train_shape=X_train_transformed.shape,
        X_test_shape=X_test_transformed.shape,
        pipeline_path=pipeline_path,
        report_path=PREPROCESSING_REPORT_MD,
    )

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "pipeline": pipeline,
        "dup_stats": dup_stats,
        "split_stats": split_stats,
    }


def generate_preprocessing_report(
    raw_filename: str,
    dup_stats: Dict[str, int],
    split_stats: Dict[str, Any],
    X_train_shape: tuple,
    X_test_shape: tuple,
    report_path: Path,
):
    """
    Generates human-readable Markdown preprocessing report (ml/data/preprocessing_report.md).
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tr = split_stats["train_set"]
    te = split_stats["test_set"]

    lines = [
        "# TrustHire — Step 3 Data Preprocessing & Leakage-Safe Pipeline Report",
        "",
        "## 1. Executive Summary",
        f"- **Raw Dataset File**: `{raw_filename}`",
        f"- **Original Row Count**: {dup_stats['rows_before']:,}",
        f"- **Exact Duplicates Removed**: {dup_stats['rows_removed']:,}",
        f"- **Rows After Preprocessing**: {dup_stats['rows_after']:,}",
        "",
        "## 2. Target Variable Encoding (`fraudulent`)",
        "- **Target Representation**: Binary Integer",
        "  - `0` = **Legitimate**",
        "  - `1` = **Potentially Fraudulent**",
        "",
        "## 3. Stratified Train / Test Split",
        "- **Split Ratio**: 80% Train / 20% Test (Stratified, `random_state=42`)",
        f"- **Training Set Size**: {tr['total']:,} samples",
        f"  - Legitimate (0): {tr['counts'].get(0, 0):,} ({tr['percentages'].get(0, 0)}%)",
        f"  - Fraudulent (1): {tr['counts'].get(1, 0):,} ({tr['percentages'].get(1, 0)}%)",
        f"- **Test Set Size**: {te['total']:,} samples",
        f"  - Legitimate (0): {te['counts'].get(0, 0):,} ({te['percentages'].get(0, 0)}%)",
        f"  - Fraudulent (1): {te['counts'].get(1, 0):,} ({te['percentages'].get(1, 0)}%)",
        "",
        "## 4. Feature Groups & Derived Features",
        "- **Text Features**: `title`, `company_profile`, `description`, `requirements`, `benefits`",
        "  - Cleaned HTML noise and combined into structured `combined_text` field with field headers.",
        "- **Categorical Features**: `location`, `department`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`",
        "  - Imputed missing values with `'Unknown'`, encoded using `OneHotEncoder(handle_unknown='ignore')` fitted on train data only.",
        "- **Binary Features**: `telecommuting`, `has_company_logo`, `has_questions`, `salary_available`",
        "  - Converted string indicators (`'f'`/`'t'`) into integers `0`/`1`.",
        "- **Salary Range Processing**: Extracted `salary_available`, `salary_min`, `salary_max`, `salary_mid` without zero-filling missing entries.",
        "- **Derived Numerical Metadata**: `text_length`, `title_length`, `description_length`, `requirements_length`, `word_count`, `url_count`, `missing_field_count`.",
        "- **Excluded Identifier**: `job_id` excluded from predictive feature matrix.",
        "",
        "## 5. Leakage Prevention Strategy",
        "1. **Pre-Split Separation**: Stratified train/test splitting performed BEFORE fitting vectorizers or scalers.",
        "2. **Strict Fit Scope**: Scikit-Learn `ColumnTransformer` (TF-IDF + OneHotEncoder + StandardScaler) fitted **ONLY** on `X_train`.",
        "3. **Safe Test Transformation**: `X_test` transformed using the already-fitted pipeline.",
        "4. **Out-of-Vocabulary Category Handling**: `OneHotEncoder(handle_unknown='ignore')` ensures unseen test categories do not crash or alter encoding.",
        "",
        "## 6. Transformed Feature Matrix Shapes",
        f"- **X_train Transformed Shape**: `{X_train_shape}` ({X_train_shape[0]:,} samples × {X_train_shape[1]:,} features)",
        f"- **X_test Transformed Shape**: `{X_test_shape}` ({X_test_shape[0]:,} samples × {X_test_shape[1]:,} features)",
        "",
        "---",
        "*Report generated programmatically during Step 3 Preprocessing Pipeline execution.*",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_final_summary(
    raw_filename: str,
    dup_stats: Dict[str, int],
    split_stats: Dict[str, Any],
    X_train_shape: tuple,
    X_test_shape: tuple,
    pipeline_path: Path,
    report_path: Path,
):
    """
    Prints concise final output summary to terminal.
    """
    tr = split_stats["train_set"]
    te = split_stats["test_set"]

    summary = f"""
==================================================
TRUSTHIRE — STEP 3 PREPROCESSING REPORT
==================================================

Raw Dataset:
{raw_filename}

Original rows: {dup_stats['rows_before']:,}
Duplicates removed: {dup_stats['rows_removed']:,}
Rows after preprocessing: {dup_stats['rows_after']:,}

Target Transformation:
0 = Legitimate
1 = Fraudulent

Class Distribution (Stratified 80/20 Split):

Training Set ({tr['total']:,} rows):
- Legitimate (0): {tr['counts'].get(0, 0):,} ({tr['percentages'].get(0, 0)}%)
- Fraudulent (1): {tr['counts'].get(1, 0):,} ({tr['percentages'].get(1, 0)}%)

Test Set ({te['total']:,} rows):
- Legitimate (0): {te['counts'].get(0, 0):,} ({te['percentages'].get(0, 0)}%)
- Fraudulent (1): {te['counts'].get(1, 0):,} ({te['percentages'].get(1, 0)}%)

Transformed Feature Matrix Shapes:
- X_train: {X_train_shape}
- X_test:  {X_test_shape}

Feature Groups:
- Text features: title, company_profile, description, requirements, benefits -> combined_text (TF-IDF max_features=5000)
- Categorical features: location, department, employment_type, required_experience, required_education, industry, function (OneHotEncoder)
- Binary features: telecommuting, has_company_logo, has_questions, salary_available
- Numerical derived: text_length, title_length, description_length, requirements_length, word_count, url_count, missing_field_count, salary_min, salary_max, salary_mid (StandardScaler)

Excluded Identifiers:
job_id

Data Leakage Prevention:
Split before fit. ColumnTransformer fitted ONLY on X_train; X_test transformed using fitted pipeline.

Artifacts Created:
- ml/data/processed/train.csv
- ml/data/processed/test.csv
- {pipeline_path}
- {report_path}

==================================================
"""
    print(summary)


if __name__ == "__main__":
    run_full_preprocessing()
