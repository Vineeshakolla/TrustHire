# TrustHire — Step 3 Data Preprocessing & Leakage-Safe Pipeline Report

## 1. Executive Summary
- **Raw Dataset File**: `fake_job_postings.csv`
- **Original Row Count**: 17,880
- **Exact Duplicates Removed**: 235
- **Rows After Preprocessing**: 17,645

## 2. Target Variable Encoding (`fraudulent`)
- **Target Representation**: Binary Integer
  - `0` = **Legitimate**
  - `1` = **Potentially Fraudulent**

## 3. Stratified Train / Test Split
- **Split Ratio**: 80% Train / 20% Test (Stratified, `random_state=42`)
- **Training Set Size**: 14,116 samples
  - Legitimate (0): 13,430 (95.14%)
  - Fraudulent (1): 686 (4.86%)
- **Test Set Size**: 3,529 samples
  - Legitimate (0): 3,357 (95.13%)
  - Fraudulent (1): 172 (4.87%)

## 4. Feature Groups & Derived Features
- **Text Features**: `title`, `company_profile`, `description`, `requirements`, `benefits`
  - Cleaned HTML noise and combined into structured `combined_text` field with field headers.
- **Categorical Features**: `location`, `department`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`
  - Imputed missing values with `'Unknown'`, encoded using `OneHotEncoder(handle_unknown='ignore')` fitted on train data only.
- **Binary Features**: `telecommuting`, `has_company_logo`, `has_questions`, `salary_available`
  - Converted string indicators (`'f'`/`'t'`) into integers `0`/`1`.
- **Salary Range Processing**: Extracted `salary_available`, `salary_min`, `salary_max`, `salary_mid` without zero-filling missing entries.
- **Derived Numerical Metadata**: `text_length`, `title_length`, `description_length`, `requirements_length`, `word_count`, `url_count`, `missing_field_count`.
- **Excluded Identifier**: `job_id` excluded from predictive feature matrix.

## 5. Leakage Prevention Strategy
1. **Pre-Split Separation**: Stratified train/test splitting performed BEFORE fitting vectorizers or scalers.
2. **Strict Fit Scope**: Scikit-Learn `ColumnTransformer` (TF-IDF + OneHotEncoder + StandardScaler) fitted **ONLY** on `X_train`.
3. **Safe Test Transformation**: `X_test` transformed using the already-fitted pipeline.
4. **Out-of-Vocabulary Category Handling**: `OneHotEncoder(handle_unknown='ignore')` ensures unseen test categories do not crash or alter encoding.

## 6. Transformed Feature Matrix Shapes
- **X_train Transformed Shape**: `(14116, 9150)` (14,116 samples × 9,150 features)
- **X_test Transformed Shape**: `(3529, 9150)` (3,529 samples × 9,150 features)

---
*Report generated programmatically during Step 3 Preprocessing Pipeline execution.*