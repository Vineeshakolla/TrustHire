# TrustHire — Final Validation & Verification Report

**Project Title:** TrustHire — Machine Learning-Based Detection of Fraudulent Job and Internship Postings  
**Date:** October 6, 2026  
**Status:** Step 10 Completed — Final Validation & System Polish  

---

## 1. Dataset Verification

| Metric / Attribute | Value | Verification Status |
| :--- | :--- | :--- |
| **Dataset Source** | Benchmark EMSCAD (`fake_job_postings.csv`) | Verified |
| **Original Rows** | 17,880 | Verified |
| **Original Columns** | 18 attributes | Verified |
| **Duplicate Rows Removed** | 235 duplicate postings | Verified |
| **Processed Modeling Dataset** | 17,645 postings | Verified |
| **Class Distribution (Raw)** | Legitimate: 17,014 \| Fraudulent: 866 | Verified |
| **Class Distribution (Processed)** | Legitimate: 16,787 \| Fraudulent: 858 | Verified |
| **Split Strategy** | 80/20 Stratified Train/Test Split | Verified |
| **Training Set** | 14,116 samples (13,430 Legitimate, 686 Fraudulent) | Verified |
| **Test Set** | 3,529 samples (3,357 Legitimate, 172 Fraudulent) | Verified |
| **Data Integrity** | Raw EMSCAD dataset files remain 100% untouched. | Verified |

---

## 2. ML Model Verification

Five candidate classification models were evaluated on the benchmark test set (3,529 postings):

| Model Candidate | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (`class_weight='balanced'`)** | **0.6301** | **0.9012** | **0.7416** | **0.9857** | **0.8511** | **Selected Model** |
| Linear SVM | 0.8324 | 0.8372 | 0.8348 | 0.9834 | 0.8880 | Candidate |
| Multinomial Naive Bayes | 0.3664 | 0.9012 | 0.5210 | 0.9720 | 0.7909 | Candidate |
| Random Forest | 0.9798 | 0.5640 | 0.7159 | 0.9890 | 0.8872 | Candidate |
| XGBoost | 0.6535 | 0.8663 | 0.7450 | 0.9899 | 0.8963 | Candidate |

### Selection Rationale
Logistic Regression with balanced class weighting was selected because:
1. **High Fraud Recall (90.12%):** Essential for catching potentially fraudulent job postings.
2. **Sparse TF-IDF Suitability:** Performs optimally on 9,150 high-dimensional sparse feature matrices.
3. **Interpretable Coefficients:** Enables transparent feature signal explanations ("Why did TrustHire flag this posting?").
4. **Calibrated Probabilities:** Yields smooth, continuous probability estimates (0.0 to 1.0).
5. **Fast Inference:** Fast runtime prediction without cloud GPU dependencies.

---

## 3. Preprocessing Verification

- **Pipeline Artifact:** Saved at `ml/saved_models/trusthire_preprocessing.joblib`
- **Model Artifact:** Saved at `ml/saved_models/trusthire_model.joblib`
- **Transformed Feature Matrix:** 9,150 total dimensions.
  - TF-IDF N-grams (Title, Company Profile, Description, Requirements, Benefits): 5,000 features (`ngram_range=(1,2)`).
  - Derived Numeric Features: Character & word lengths, salary bounds (`salary_min`, `salary_max`, `salary_median`, `has_salary`).
  - Encoded Categorical Attributes: One-Hot encoded `location`, `department`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`.
  - Binary Indicators: `telecommuting`, `has_company_logo`, `has_questions`.
- **Leakage Prevention:** Transformers are loaded from frozen serialized artifacts during runtime inference; no re-fitting occurs on incoming test or user postings.

---

## 4. API & Endpoint Verification

| Endpoint | HTTP Method | Input Payload | Output / Response | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | GET | None | `{"online": True, "model_loaded": True}` | Passed |
| `/predict` | POST | `JobPostingSchema` | `PredictionResponse` (prediction, risk_score, risk_percentage, signals) | Passed |
| `/analyze` | POST | `JobPostingSchema` | `ExtractionResponse` (18 structured fields, confidence) | Passed |
| `/analyze/intelligent` | POST | `JobPostingSchema` | `IntelligentAnalysisResponse` (summary, skills, warning signals) | Passed |
| `/analyze/ask` | POST | `QARequestSchema` | `QAResponse` (intent, answer, confidence) | Passed |
| `/analyze/full` | POST | `JobPostingSchema` | `FullAnalysisResponse` (combined ML, NLP, summary, signals, Q&A) | Passed |

---

## 5. NLP & AI Features Verification

### Information Extraction Layer (Layer 2)
- **Extracted Fields:** Title, Company, Location, Work Mode (Remote/Hybrid/On-site), Employment Type, Salary Range, Experience, Education, Skills, Duration, Start Date, Internship Flag, Responsibilities, Requirements, Benefits, Missing Metadata.
- **Confidence Scoring:** High, Medium, Low confidence heuristics per field based on pattern strength.

### Intelligent Analysis Layer (Layer 3)
- **Job Summary:** Formats role, company profile, location, salary/stipend, experience, education, duration, start date.
- **Skill Categorization:** Groups technical and soft skills into Required vs Preferred skill chips.
- **Categorized Requirements:** Segregates requirements into Education, Experience, Technical Skills, Soft Skills, and Other Items.
- **Warning Signals:** Detects 8 rule-based caution patterns (Urgency, Fees, Off-platform messaging, Salary anomaly, Short description, Missing logo, Missing company info, High income claims) using non-accusatory, cautious phrasing.

### Local Deterministic Q&A Engine
- **Supported Intents:** Salary, Location, Remote Work, Required Skills, Experience, Education, Responsibilities, Internship Duration, Start Date, Benefits, Missing Information, Warning Signals, TrustHire Risk Rating.
- **Non-Hallucination Safeguard:** If information is absent from the job text, returns `"That information is not explicitly provided in the job posting."`

---

## 6. Risk Meter Verification

- **Meter Design:** Continuous horizontal bar with dynamic indicator dot positioning.
- **Risk Percentage Calculation:** $Risk\ Percentage = Risk\ Score \times 100\%$ ($0.0\% \le Risk \le 100.0\%$).
- **Visual Scale & Ranges:**
  - `0–20%`: **High Possibility of Legitimate** (Emerald Accent)
  - `21–40%`: **Possibility of Legitimate** (Cyan Accent)
  - `41–60%`: **Moderate Risk** (Amber Accent)
  - `61–80%`: **High Risk** (Orange/Rose Accent)
  - `81–100%`: **Very High Risk** (Rose/Red Accent)

---

## 7. Realistic Test Scenarios Evaluation

Ten diverse real-world job posting test scenarios were created in `tests/test_end_to_end_scenarios.py`:

| Scenario ID | Test Case Title | Risk % | Risk Level Category | Missing Info / Warning Signals | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Software Engineering Intern | 27.2% | Possibility of Legitimate | None | Passed |
| **B** | Senior Backend Engineer | 18.5% | High Possibility of Legitimate | None | Passed |
| **C** | Data Science & ML Intern | 22.4% | Possibility of Legitimate | None | Passed |
| **D** | Digital Marketing Intern | 29.1% | Possibility of Legitimate | None | Passed |
| **E** | Financial Analyst | 16.8% | High Possibility of Legitimate | None | Passed |
| **F** | Full Stack Web Developer (Missing Salary) | 34.5% | Possibility of Legitimate | Missing Salary | Passed |
| **G** | Java Systems Engineer (Missing Company) | 48.2% | Moderate Risk (Needs Caution) | Missing Company Profile Warning | Passed |
| **H** | Junior QA Tester (Short Description) | 52.6% | Moderate Risk (Needs Caution) | Short Description Warning | Passed |
| **I** | Data Entry Specialist (Urgent + Fee + Telegram) | 88.4% | Very High Risk | Urgency, Fee, Telegram Signals | Passed |
| **J** | Online Assistant (High Income Claim) | 91.2% | Very High Risk | Unusually High Income Claim | Passed |

---

## 8. Test Suite Summary

- **Total Backend Tests:** **86 / 86 PASSED**
- **Execution Time:** ~3.5 minutes (full model & API integration suite)
- **Coverage Areas:**
  - `test_api.py`: 12 tests (FastAPI route schemas, validation, health)
  - `test_data_pipeline.py`: 16 tests (Preprocessing, text cleaning, TF-IDF, salary parsing)
  - `test_modeling.py`: 15 tests (Logistic Regression, evaluation metrics, thresholds)
  - `test_prediction.py`: 14 tests (Single posting inference, signal log-odds)
  - `test_nlp_extraction.py`: 14 tests (Regex extraction, skill dictionaries, Q&A intents)
  - `test_end_to_end_scenarios.py`: 15 tests (10 real-world job scenarios, Q&A assertions)

---

## 9. Final System Architecture Diagram

```
                       EMSCAD Dataset (fake_job_postings.csv)
                                        │
                                        ▼
                             Leakage-Safe Preprocessing
                                        │
                                        ▼
                           80/20 Stratified Train/Test
                                        │
                                        ▼
                             9,150 Feature Matrix
                                        │
                                        ▼
                         Logistic Regression Classifier
                                        │
                                        ▼
                        Saved Model & Pipeline Artifacts
                                        │
                                        ▼
                               FastAPI REST Server
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  │                                           │
                  ▼                                           ▼
          ML Risk Screening                        Local AI & NLP Engine
         (POST /predict)                           (POST /analyze/*)
                  │                                           │
                  ├─────────────────────┬─────────────────────┤
                  │                     │                     │
                  ▼                     ▼                     ▼
          Risk Percentage        Info Extraction       Intelligent Summary
          Legitimate/Fraud       Skill Breakdown       Warning Signals
          Log-Odds Signals       Requirements          Local Intent Q&A
                  │                     │                     │
                  └─────────────────────┼─────────────────────┘
                                        │
                                        ▼
                             React + Vite Web UI
                                (TrustHire SPA)
```

---

## 10. Security, Data Integrity & Constraints Audit

1. **Model & Preprocessing Integrity:** Saved model (`ml/saved_models/trusthire_model.joblib`) and preprocessing (`ml/saved_models/trusthire_preprocessing.joblib`) were not modified, replaced, or retrained.
2. **Raw Dataset Integrity:** `fake_job_postings.csv` remains 100% unchanged.
3. **Threshold Stability:** Production screening threshold remains at `0.50`.
4. **Zero External AI Dependencies:** Uses zero external AI APIs (No OpenAI, Gemini, or Claude). 100% local deterministic execution.
5. **Cautious Language:** Uses non-accusatory screening language ("Potentially Fraudulent", "Potential warning signal", "Information requiring verification").

---

## 11. Known Limitations

1. **Statistical Screening Indicator:** Risk scores represent probabilistic pattern matching on benchmark EMSCAD data, not legal proof of fraud.
2. **Domain Specificity:** Pre-trained primarily on English tech, corporate, and office job postings. Non-English postings are not supported.
