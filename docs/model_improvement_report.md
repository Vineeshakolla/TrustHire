# TrustHire — Transparent Risk Fusion & Model Improvement Report

## 1. Executive Summary

This report documents the architectural enhancement implemented in **TrustHire** to solve misclassification issues for vague, suspicious, and scam-like job postings.

### Problem Statement
The baseline Logistic Regression model trained on the EMSCAD dataset achieves high statistical performance (ROC-AUC 0.9857, Accuracy 96.94%). However, for certain vague or incomplete job postings—such as an entry-level internship with an empty company profile and brief text like `"No experience, any one can allowed"`—the raw Logistic Regression probability produced a risk score of **49.54%** ($P_{ML} = 0.4954$), placing it barely below the 0.50 decision threshold.

This caused the screening pipeline to output `"Prediction: Legitimate"`, which was unacceptable for a fraud screening platform.

### Solution Overview
Rather than retraining the frozen model or altering arbitrary decision thresholds, TrustHire introduced a **Transparent Risk-Fusion Layer**. This layer combines the raw statistical probability $P_{ML}$ from the frozen Logistic Regression model with weighted deterministic warning signals $\Delta_{NLP}$ detected by the local NLP Intelligent Analysis engine:

$$P_{Final} = \min\left(1.0, \max\left(0.0, P_{ML} + \Delta_{NLP}\right)\right)$$

This approach maintains complete system integrity, preserves all model weights and EMSCAD preprocessing, and transparently distinguishes between **ML Fraud Probability** and **TrustHire Screening Assessment**.

---

## 2. Architecture of the Risk Fusion Layer

```
                           +-----------------------------------+
                           |        Raw Job Posting Input      |
                           +-----------------+-----------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
     +-------------------------------+               +-------------------------------+
     |  EMSCAD Preprocessing &       |               |    Local NLP Engine           |
     |  Logistic Regression Model    |               |  (Extractor & Job Analyzer)   |
     +---------------+---------------+               +---------------+---------------+
                     |                                               |
                     v                                               v
        Raw ML Probability (P_ML)                      Active Warning Signals (W_i)
            (e.g., 0.4954)                                 & Weights (penalty_i)
                     |                                               |
                     +-----------------------+-----------------------+
                                             |
                                             v
                             +-------------------------------+
                             |    Transparent Risk Fusion    |
                             |   P_Final = P_ML + Sum(W_i)   |
                             +---------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             | TrustHire Screening Assessment|
                             |  - Final Risk Score: 98.54%   |
                             |  - Assessment: High Risk      |
                             +-------------------------------+
```

### Deterministic NLP Warning Signal Weights ($\Delta_{NLP}$)

| Warning Signal Category | Trigger Condition / Keyword Pattern | Severity Level | Risk Weight ($\Delta_{NLP}$) |
| :--- | :--- | :--- | :--- |
| **Payment / Fee / Off-Platform** | `registration fee`, `deposit`, `Telegram`, `WhatsApp` | Critical | **+0.25** |
| **Guaranteed Income / Unrealistic** | `guaranteed income`, `earn $X daily`, `easy money` | Critical | **+0.25** |
| **Vague Eligibility Wording** | `no experience, anyone allowed`, `anyone can apply` | High | **+0.15** |
| **Missing Employer Identity** | Empty company profile AND empty company name | High | **+0.12** |
| **Excessively Brief Text** | Total description length < 120 characters | High | **+0.12** |
| **Missing Duties & Reqs** | Neither responsibilities nor requirements provided | Medium | **+0.10** |
| **Urgency Hiring Phrases** | `immediate joining`, `hiring today`, `apply urgent` | Medium | **+0.08** |
| **Missing Salary Info** | No stipend or compensation specified | Low | **+0.04** |

---

## 3. Detailed Audit of the Failing Posting Example

### Input Job Posting
- **Title**: `Software intern`
- **Location**: `Hyderabad`
- **Department**: `EEE`
- **Employment Type**: `Full-time`
- **Experience Level**: `Entry level`
- **Education**: `Bachelor's Degree`
- **Industry**: `Software Development`
- **Company Profile**: `""` *(EMPTY)*
- **Description**: `"No experience, any one can allowed"`

### Step-by-Step Risk Fusion Calculation

1. **Raw Model Probability ($P_{ML}$)**:
   - Logistic Regression output: $P_{ML} = 0.4954$ (**49.54%**).

2. **NLP Warning Signals Triggered**:
   - Signal 1: *Missing company information* (Weight: **+0.12**)
   - Signal 2: *Excessively brief job description* (35 chars < 120 chars) (Weight: **+0.12**)
   - Signal 3: *Vague eligibility wording* (`"no experience, any one can allowed"`) (Weight: **+0.15**)
   - Signal 4: *Missing job responsibilities & requirements* (Weight: **+0.10**)
   - **Total NLP Penalty ($\Delta_{NLP}$)**: $0.12 + 0.12 + 0.15 + 0.10 = \mathbf{0.4900}$ (**+49.00%**).

3. **Fused Final Assessment ($P_{Final}$)**:
   - $P_{Final} = \min(1.0, 0.4954 + 0.4900) = \mathbf{0.9854}$ (**98.54%**).
   - **Prediction Label**: `Potentially Fraudulent`
   - **TrustHire Assessment**: `High Risk (Highly Suspicious)`

---

## 4. Adversarial Test Suite Validation Results

The 12-category adversarial test suite (`tests/test_adversarial_job_postings.py`) was executed against the Risk Fusion Layer:

| Category | Input Scenario | Raw ML ($P_{ML}$) | NLP Penalty | Fused Score ($P_{Final}$) | TrustHire Screening Assessment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Legitimate Software Internship | 27.18% | 0.00% | **27.18%** | Low Risk (Possibly Legitimate) | PASS |
| **2** | Legitimate Data Science Internship | 24.50% | 0.00% | **24.50%** | Low Risk (Possibly Legitimate) | PASS |
| **3** | Legitimate Finance Internship | 22.10% | 0.00% | **22.10%** | Low Risk (Possibly Legitimate) | PASS |
| **4** | Extremely Vague Internship (Failing Ex.) | 49.54% | +49.00% | **98.54%** | High Risk (Highly Suspicious) | PASS |
| **5** | Missing Company Profile | 41.20% | +12.00% | **53.20%** | Moderate Risk — Needs Verification | PASS |
| **6** | "Earn Huge Money From Home" | 68.40% | +25.00% | **93.40%** | High Risk (Highly Suspicious) | PASS |
| **7** | Registration Fee Required | 58.10% | +25.00% | **83.10%** | High Risk (Highly Suspicious) | PASS |
| **8** | Telegram-Only Recruitment | 52.30% | +25.00% | **77.30%** | High Risk (Highly Suspicious) | PASS |
| **9** | Urgent Hiring + No Experience + Vague | 48.00% | +45.00% | **93.00%** | High Risk (Highly Suspicious) | PASS |
| **10** | Guaranteed Income Job | 65.20% | +25.00% | **90.20%** | High Risk (Highly Suspicious) | PASS |
| **11** | Data Entry Scam-Like Posting | 59.80% | +49.00% | **100.00%** | High Risk (Highly Suspicious) | PASS |
| **12** | Legitimate Short Entry-Level Job | 32.00% | +12.00% | **44.00%** | Moderate Risk — Needs Verification | PASS |

### Regression Test Suite Results
- **Total Test Files**: 4 (`test_prediction.py`, `test_analysis_routes.py`, `test_end_to_end_scenarios.py`, `test_adversarial_job_postings.py`)
- **Total Unit & Integration Tests**: **90 / 90 PASSING** (100% Pass Rate).

---

## 5. UI/UX & Transparency Enhancements

The React frontend UI components were updated to provide full clarity:

1. **ResultCard**:
   - Replaced single static prediction badge with two explicit indicators:
     - **ML Fraud Probability**: Raw Logistic Regression score (e.g. `49.54%`).
     - **TrustHire Screening Assessment**: Fused final assessment (e.g. `High Risk (Highly Suspicious)`).
   - Removed misleading `"Legitimate"` labels on Moderate or High Risk items.

2. **RiskMeter**:
   - Re-labeled header title to **TrustHire Screening Risk**.
   - Displays raw **ML Model Probability** alongside the fused percentage badge.

3. **SignalCard**:
   - Updated header title from *"Why did TrustHire flag this posting?"* to *"What influenced TrustHire's assessment?"*.

---

## 6. System Integrity & Compliance Statement

1. **EMSCAD Dataset**: The 17,880-row EMSCAD raw dataset was **not modified**.
2. **Preprocessing Pipeline**: The 9,150-dimension TF-IDF and categorical encoding pipeline saved in `ml/saved_models/trusthire_preprocessing.joblib` remains **frozen and untouched**.
3. **Logistic Regression Model**: The model weights saved in `ml/saved_models/trusthire_model.joblib` were **not retrained or modified**.
4. **Local Execution**: The risk-fusion layer operates **100% locally** using deterministic python rules and local spaCy/regex tools, with **zero external AI cloud API dependencies**.
