"""
# TrustHire — Backend Architecture & Prediction Flow Documentation

## 1. System Overview

TrustHire is a machine-learning-based application designed to detect potentially fraudulent job and internship postings. The system architecture strictly separates **offline model training** from **runtime prediction execution**.

---

## 2. Component Architecture Flow

```
+-------------------------------------------------------------+
|                      Frontend Interface                     |
|                 (React / HTML / CSS / JS)                   |
+-------------------------------------------------------------+
                              |
                              | HTTP POST /predict (JSON)
                              v
+-------------------------------------------------------------+
|                    FastAPI Backend Service                  |
|                 (backend/main.py & routes)                  |
+-------------------------------------------------------------+
                              |
                              | Delegates job data & threshold
                              v
+-------------------------------------------------------------+
|                     Prediction Service                      |
|           (backend/services/prediction_service.py)          |
+-------------------------------------------------------------+
                              |
                              | Calls core ML prediction logic
                              v
+-------------------------------------------------------------+
|                ML Prediction Engine Module                  |
|                 (ml/prediction/predict.py)                  |
+-------------------------------------------------------------+
                              |
                              | Transforms raw input fields
                              v
+-------------------------------------------------------------+
|             Saved Preprocessing Pipeline Artifact           |
|        (ml/saved_models/trusthire_preprocessing.joblib)     |
+-------------------------------------------------------------+
                              |
                              | Produces 9,150-dimension feature vector
                              v
+-------------------------------------------------------------+
|              Trained Logistic Regression Model              |
|           (ml/saved_models/trusthire_model.joblib)          |
+-------------------------------------------------------------+
                              |
                              | Returns calibrated probability & log-odds
                              v
+-------------------------------------------------------------+
|            JSON Screening Response to Frontend              |
| (Prediction, Risk Score %, Risk Level, Feature Signals)     |
+-------------------------------------------------------------+
```

---

## 3. Key Design Principles

1. **Offline Training vs. Runtime Inference**:
   - Model training, feature selection, TF-IDF vocabulary building, and hyperparameter calibration occur strictly offline during development (Steps 3–5).
   - Runtime API prediction requests **NEVER retrain the model** or fit vectorizers. Inference is fast, deterministic, and lightweight.

2. **No Data Leakage**:
   - The preprocessing pipeline (`trusthire_preprocessing.joblib`) was fitted exclusively on training data and remains frozen during production inference.

3. **100% Local Machine Learning**:
   - The system operates entirely on local trained Scikit-Learn model artifacts. No external AI APIs (e.g. OpenAI, Gemini, Claude) are invoked.

4. **Calibrated Risk Scores & Terminology**:
   - Standard dataset target `1` maps to `"Potentially Fraudulent"` and `0` maps to `"Legitimate"`.
   - The continuous model risk score drives clear risk level indicators (`Low Risk`, `Moderate Risk`, `High Risk`).

5. **Model Interpretability Signals**:
   - For Logistic Regression, the system evaluates individual feature contributions (`feature_value * coefficient`) to provide transparent signals explaining why a posting received its risk rating.
