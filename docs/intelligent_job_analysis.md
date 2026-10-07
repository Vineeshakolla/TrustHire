# TrustHire — Intelligent Job Analysis, Summary & Local Q&A Documentation

## Overview & Core Philosophy

The **TrustHire Intelligent Job Analysis Layer** (`ml/ai/job_analyzer.py` & `ml/ai/job_qa.py`) transforms raw job posting text and extracted attributes into clean, structured summaries, categorized requirements, rule-based warning signals, and a deterministic local question-answering system.

> **Mandatory Architecture Statement:**
> *"TrustHire currently uses a local NLP-based intelligent analysis layer. No external generative AI API is used."*

---

## Separation of Responsibilities

| System Component | Endpoint | Primary Responsibility | Output Wording |
| :--- | :--- | :--- | :--- |
| **ML Risk Classifier** | `POST /predict` | Binary statistical fraud screening via Scikit-Learn Logistic Regression model (`class_weight='balanced'`) | `Legitimate` vs `Potentially Fraudulent` (Calibrated Risk Score %) |
| **AI Information Extractor** | `POST /analyze` | 4-layer NLP extraction of 18 structured fields and missing information | JSON representation of job posting fields |
| **Intelligent Job Analyzer** | `POST /analyze/intelligent` | Structured summary, skills breakdown, requirements categorization, rule-based warning signals | `JobSummary`, `SkillsBreakdown`, `WarningSignal` |
| **Local Q&A Engine** | `POST /analyze/ask` | Local intent matching over posting details without hallucination | Answer string + Source + Confidence |
| **Combined Full Pipeline** | `POST /analyze/full` | End-to-end processing executing ML screening, NLP extraction, and intelligent analysis | `FullAnalysisResponse` |

---

## 1. Summary Generation (`generate_job_summary`)

Extracts clean posting highlights while falling back to `"Not specified"` for unstated fields to prevent hallucinating information.

- **Role**: Job title or `"Not specified"`
- **Company**: Stated employer or agency notice or `"Not specified"`
- **Location**: Extracted city/country or `"Not specified"`
- **Work Mode**: `"Remote"`, `"On-site"`, or `"Hybrid"`
- **Employment Type**: `"Full-time"`, `"Internship"`, etc.
- **Salary**: Numerical range / stipend string or `"Not specified"`
- **Experience**: Explicit prerequisite (e.g. `"0 years"`)
- **Education**: Stated degree requirements or `"Not specified"`
- **Duration / Start Date**: Internship contract duration & start timeline

---

## 2. Rule-Based Warning Signal Detection (`detect_warning_signals`)

Warning signals serve as informational caution flags for applicants.

> **Cautionary Guidelines:**
> Warning signals use cautious phrasing (e.g., *"Information requiring verification"*, *"Potential warning signal"*). They **do not** automatically label a posting as fraudulent.

### Signal Triggers
1. **Missing Company Information**: Missing employer name / agency listing.
2. **Missing Salary Information**: Unstated stipend or salary range.
3. **Urgency Hiring Language**: Text containing `"immediate joining"`, `"apply immediately"`, or `"urgent requirement"`.
4. **Off-Platform Communication / Fee Requests**: Wording mentioning `"registration fee"`, `"processing fee"`, `"telegram"`, or `"whatsapp"`.
5. **Guaranteed Income Claims**: Unrealistic earnings claims e.g., `"guaranteed income"`, `"easy money"`.
6. **Vague Description**: Posting text length under 100 characters.

---

## 3. Local Deterministic Q&A Engine (`answer_job_question`)

Operates 100% locally without calling external LLM services.

### Intent Matching Table

| Question Intent | Example Question | Resolution Source |
| :--- | :--- | :--- |
| **Salary** | *"What is the salary?"* | `salary.raw` from `/analyze` |
| **Location** | *"Where is the job located?"* | `location` & `work_mode` |
| **Remote** | *"Is it remote?"* | `work_mode` attribute |
| **Skills** | *"What skills are required?"* | Technical skills array |
| **Experience** | *"What experience is required?"* | `experience` attribute |
| **Education** | *"What degree is required?"* | `education` attribute |
| **Responsibilities**| *"What are the responsibilities?"* | Responsibilities section list |
| **Internship** | *"Is this an internship?"* | `internship` boolean flag |
| **Duration** | *"How long is the internship?"* | `duration` attribute |
| **Start Date** | *"When can I join?"* | `start_date` attribute |
| **Company** | *"Who is the company?"* | `company_name` / `company_profile` |
| **Benefits** | *"What benefits are provided?"* | Benefits section list |
| **Missing Info** | *"What information is missing?"* | `important_missing_information` |
| **Warning Flags**| *"What warning signals were detected?"*| Calculated `warning_signals` |
| **Fraud Risk** | *"Is this job fraudulent?"* | Directs user to ML `/predict` model |
| **Unhandled** | *"What is the dress code?"* | `"That information is not explicitly provided in the job posting."` |

---

## API Endpoint Specifications

### 1. `POST /analyze/intelligent`
**Request:** `JobAnalysisRequest`
**Response (`IntelligentAnalysisResponse`):**
```json
{
  "success": true,
  "summary": {
    "role": "Software Engineering Intern",
    "company": "This role is for one of our clients",
    "location": "Bengaluru",
    "work_mode": "Remote",
    "employment_type": "Full-time",
    "salary": "₹30,000 – ₹50,000 per month",
    "experience": "0 years",
    "education": "Computer Science",
    "duration": "Minimum 2 months, extendable up to 6 months",
    "start_date": "Immediate / As soon as possible",
    "internship_status": "Yes (Internship Posting)"
  },
  "skills": {
    "required_skills": ["Python", "JavaScript", "Java", "C++", "Data Structures", "Algorithms"],
    "preferred_skills": [],
    "all_skills": ["Algorithms", "C++", "Data Structures", "Java", "JavaScript", "Python"]
  },
  "warning_signals": [
    {
      "signal": "Role posted via recruitment agency or client representative",
      "evidence": "This role is for one of our clients",
      "explanation": "The posting is listed by a third-party recruiter or agency. Verify the end client company during interviews."
    }
  ]
}
```

### 2. `POST /analyze/ask`
**Request:**
```json
{
  "job_posting": {
    "title": "Software Engineering Intern",
    "description": "Location: Bengaluru (Remote). Stipend: ₹30,000 per month."
  },
  "question": "What is the salary?"
}
```
**Response (`JobQAResponse`):**
```json
{
  "success": true,
  "question": "What is the salary?",
  "answer": "The posting states a compensation/stipend of ₹30,000 per month.",
  "source": "Submitted job posting",
  "confidence": "high"
}
```

### 3. `POST /analyze/full`
Combines ML statistical screening (`/predict`), NLP extraction (`/analyze`), and intelligent analysis (`/analyze/intelligent`) into a single endpoint returning `FullAnalysisResponse`.
