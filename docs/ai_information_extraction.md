# TrustHire — AI Job Information Extraction Documentation

## Overview & Purpose

The **AI Job Information Extraction Layer** (`ml/ai/`) provides deterministic, local natural language processing (NLP) capabilities to extract, structure, and analyze job posting content. 

> **Important Architecture Distinction:**
> *"This module provides NLP-based information extraction and does not replace the machine-learning fraud classification model."*

While the machine-learning classifier (`ml/prediction/` & `POST /predict`) handles binary risk classification (`Legitimate` vs `Potentially Fraudulent`), the AI extraction module (`ml/ai/` & `POST /analyze`) structures unorganized job postings into standardized JSON attributes and identifies missing critical metadata.

---

## 4-Layer Extraction Architecture

The extraction engine operates locally without relying on external generative AI APIs, using a 4-layer resolution strategy:

1. **Layer 1: Structured Input Resolution**
   - Directly parses explicitly provided form inputs (e.g., job title, location, employment type).
2. **Layer 2: Regular Expression & Pattern Extraction (`patterns.py`)**
   - Applies compiled regular expressions to extract numerical salary bounds, currency codes (`INR`, `USD`), payment periods (`month`, `year`), experience prerequisites (`0 years`, `1–2 years`), internship/contract durations, and start dates.
3. **Layer 3: Structural Section Parsing (`information_extractor.py`)**
   - Analyzes header boundaries (e.g., `Responsibilities`, `Requirements`, `Benefits`) and extracts bulleted lists into structured Python lists.
4. **Layer 4: Contextual Keyword & Skill Dictionary Matching (`skill_dictionary.py`)**
   - Matches technical skills against a configurable dictionary (`Python`, `Java`, `C++`, `FastAPI`, `Docker`, `Machine Learning`, etc.) with boundary-aware regex rules.

---

## Extracted Information Schema (`ExtractedJobInformation`)

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `job_title` | `str` \| `null` | Extracted or normalized job title |
| `company_name` | `str` \| `null` | Explicit company name (`null` if client agency notice) |
| `company_profile` | `str` \| `null` | Overview or agency client note |
| `location` | `str` \| `null` | Cleaned city/region (e.g., `"Bengaluru"`) |
| `work_mode` | `str` \| `null` | `"Remote"`, `"On-site"`, or `"Hybrid"` |
| `employment_type` | `str` \| `null` | Normalized type (e.g., `"Full-time"`, `"Internship"`) |
| `salary` | `SalaryInformation` \| `null` | Structured salary object (`minimum`, `maximum`, `currency`, `period`) |
| `experience` | `str` \| `null` | Explicit experience prerequisite (e.g., `"0 years"`) |
| `education` | `str` \| `null` | Explicit education requirements (e.g., `"Computer Science"`) |
| `skills` | `List[str]` | Unique technical skills matched |
| `duration` | `str` \| `null` | Contract or internship duration |
| `start_date` | `str` \| `null` | Stated availability / start date |
| `responsibilities` | `List[str]` | Parsed duty bullet points |
| `requirements` | `List[str]` | Parsed requirement bullet points |
| `benefits` | `List[str]` | Parsed perk bullet points |
| `internship` | `bool` | `True` if intern/internship keywords are detected |
| `important_missing_information` | `List[str]` | List of unstated critical posting fields |
| `confidence_scores` | `Dict[str, str]` | Attribute extraction confidence (`"high"`, `"medium"`, `"low"`) |

---

## API Endpoint Specification: `POST /analyze`

### Request Payload (`JobAnalysisRequest`)
```json
{
  "title": "Software Engineering Intern",
  "description": "We are looking for a Software Engineering Intern...\nStipend: ₹30,000 – ₹50,000 per month\nLocation: Bengaluru (Remote)\nRequirements:\n- Proficiency in Python, JavaScript, and Java.",
  "company_profile": "This role is for one of our clients"
}
```

### Response Payload (`JobAnalysisResponse`)
```json
{
  "success": true,
  "extracted_information": {
    "job_title": "Software Engineering Intern",
    "company_name": null,
    "company_profile": "This role is for one of our clients",
    "location": "Bengaluru",
    "work_mode": "Remote",
    "employment_type": "Full-time",
    "salary": {
      "raw": "₹30,000 – ₹50,000 per month",
      "minimum": 30000.0,
      "maximum": 50000.0,
      "currency": "INR",
      "period": "month"
    },
    "experience": "0 years",
    "education": "Computer Science",
    "skills": ["C++", "Data Structures", "Java", "JavaScript", "Python"],
    "duration": "Minimum 2 months, extendable up to 6 months",
    "start_date": "Immediate / As soon as possible",
    "internship": true,
    "important_missing_information": ["Company Information", "Benefits Information"],
    "confidence_scores": {
      "job_title": "high",
      "company_name": "high",
      "location": "high",
      "work_mode": "high",
      "salary": "high"
    }
  },
  "metadata": {
    "extractor_version": "1.0.0",
    "extraction_layer": "Local NLP Layer (Regex + Pattern + Context Skill Dictionary)",
    "external_api_used": false,
    "disclaimer": "This module provides NLP-based information extraction and does not replace the machine-learning fraud classification model."
  }
}
```

---

## System Limitations & Safety Rules

1. **Non-Invention Rule**: Missing values are returned as `null` / `None` or `[]` rather than hallucinating facts.
2. **Missing Information Completeness**: Identified missing fields (e.g. missing salary) serve as completeness signals and do not imply fraudulent intent.
3. **No External Dependencies**: 100% deterministic local execution without network calls or cloud AI APIs.
