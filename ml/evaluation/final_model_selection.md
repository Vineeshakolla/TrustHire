# TrustHire — Final Model Selection & Packaging Report

## 1. Selected Model Overview
- **Selected Model**: **Logistic Regression** (with `class_weight='balanced'`)
- **Model Artifact**: `trusthire_model.joblib`
- **Preprocessing Artifact**: `trusthire_preprocessing.joblib`
- **Dataset**: EMSCAD (fake_job_postings.csv) (14,116 train / 3,529 test)
- **Feature Dimension**: 9,150 input features

## 2. Evidence-Based Selection Rationale
Logistic Regression was selected for the TrustHire screening system based on five core criteria:

1. **High Fraud Detection Sensitivity (Recall = 90.12%)**: At the default threshold (0.50), Logistic Regression successfully captures 155 out of 172 fraudulent postings in the test set, missing only 17 positive cases (False Negatives). In a fraud detection context, minimizing missed fraudulent postings is critical.
2. **Direct Linear Feature Interpretability**: Unlike tree ensembles or kernel SVMs, Logistic Regression provides direct access to feature log-odds coefficients (`model.coef_`). This allows TrustHire to explain specific text and attribute signals to the user.
3. **Native Calibrated Probabilities (`predict_proba`)**: Natively outputs smooth, calibrated probability scores ranging from 0.0 to 1.0 (0% to 100%), ideal for driving the TrustHire continuous Risk Indicator UI.
4. **Threshold Flexibility**: Threshold analysis demonstrates that adjusting the threshold to 0.60/0.70 boosts precision to 70.6%–77.1% while retaining 84.3%–86.6% recall.
5. **Real-Time Efficiency**: Instant inference speed with minimal memory footprint, making it ideal for FastAPI microservice integration.

## 3. Measured Test Evaluation Metrics
- **Accuracy**: `0.9694`
- **Precision**: `0.6301`
- **Recall**: `0.9012`
- **F1-Score**: `0.7416`
- **ROC-AUC**: `0.9857`
- **PR-AUC**: `0.8511`

### Confusion Matrix Breakdown (Test Set = 3,529 postings)
- **True Positives (TP)**: `155` (Correctly identified fraudulent postings)
- **False Positives (FP)**: `91` (Legitimate postings flagged for caution)
- **False Negatives (FN)**: `17` (Fraudulent postings missed)
- **True Negatives (TN)**: `3266` (Correctly identified legitimate postings)

## 4. Risk Indicator Categories
- **Low Risk (0.00 – 0.29)**: High confidence legitimate posting.
- **Moderate Risk / Needs Caution (0.30 – 0.59)**: Posting exhibits minor risk signals; user advised to proceed with caution.
- **High Risk / Potentially Fraudulent (0.60 – 1.00)**: Strong model risk signals present; detailed review recommended.

## 5. Model Limitations & Safety Guidelines
- Predictions represent statistical screening signals, not legally binding proof of fraud.
- Model depends on features extracted from historical EMSCAD job posting data.

---
*Report generated programmatically during Step 5 Model Packaging.*