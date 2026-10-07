# TrustHire — Step 4 Model Training & Evaluation Report

## 1. Executive Summary
- **Dataset Used**: EMSCAD Raw Job Postings (`fake_job_postings.csv`)
- **Training Dataset Size**: 14,116 samples (80% Stratified Split)
- **Testing Dataset Size**: 3,529 samples (20% Stratified Split)
- **Positive Target Class**: `1` = **Potentially Fraudulent** (~4.87% of test set)
- **Negative Target Class**: `0` = **Legitimate** (~95.13% of test set)

## 2. Models Evaluated
1. **Logistic Regression** (Primary candidate, `class_weight='balanced'`)
2. **Linear SVM** (`LinearSVC`, `class_weight='balanced'`)
3. **Multinomial Naive Bayes** (Trained on non-negative scaled feature matrix)
4. **Random Forest** (`RandomForestClassifier`, `class_weight='balanced'`)
5. **XGBoost** (Evaluated)

## 3. Comparative Performance Metrics Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | 0.9711 | 0.6535 | 0.8663 | 0.7450 | 0.9899 | 0.8963 |
| **Linear SVM** | 0.9838 | 0.8324 | 0.8372 | 0.8348 | 0.9834 | 0.8880 |
| **Random Forest** | 0.9782 | 0.9798 | 0.5640 | 0.7159 | 0.9890 | 0.8872 |
| **Logistic Regression** | 0.9694 | 0.6301 | 0.9012 | 0.7416 | 0.9857 | 0.8511 |
| **Multinomial Naive Bayes** | 0.9192 | 0.3664 | 0.9012 | 0.5210 | 0.9720 | 0.7909 |

## 4. Confusion Matrix Summary
| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| --- | --- | --- | --- | --- |
| **XGBoost** | 149 | 79 | 23 | 3278 |
| **Linear SVM** | 144 | 29 | 28 | 3328 |
| **Random Forest** | 97 | 2 | 75 | 3355 |
| **Logistic Regression** | 155 | 91 | 17 | 3266 |
| **Multinomial Naive Bayes** | 155 | 268 | 17 | 3089 |

## 5. Model Signals & Interpretability (Logistic Regression)
- **Positive Coefficient Signals (Model-associated Fraud Indicators)**: Features whose presence increases the log-odds of a posting being flagged as potentially fraudulent (e.g. missing company profiles, specific keywords/phrases, unverified contact methods).
- **Negative Coefficient Signals (Model-associated Legitimate Indicators)**: Features whose presence increases the log-odds of a posting being flagged as legitimate (e.g. verified company logo, detailed requirement sections, established industry metadata).
- **Important Precaution**: Feature coefficients represent statistical correlations in historical data, NOT absolute proof of fraud.

## 6. Observations & Trade-Offs
- **Imbalance Impact**: Standard accuracy (95%+) is misleading because a naive dummy model predicting all legitimate achieves ~95.1% accuracy with 0% recall.
- **Precision vs. Recall Trade-Off**:
  - High Recall models capture nearly all fraudulent postings (low False Negatives) but produce more False Positives (Needs Caution).
  - High Precision models produce very few false alarms but risk missing subtle fraudulent postings.
- **PR-AUC Importance**: Precision-Recall AUC is the most informative metric for severe class imbalance.

## 7. Next Recommended Development Step
- Proceed to Step 5: System integration, hyperparameter selection, FastAPI API endpoint, and simple frontend display.

---
*Report generated programmatically during Step 4 Model Evaluation.*