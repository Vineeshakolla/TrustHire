# TrustHire

**Machine Learning-Based Detection of Potentially Fraudulent Job and Internship Postings**

## Overview
TrustHire is a screening and decision-support system designed to analyze online job and internship postings to identify potential indicators of fraud. Rather than making absolute, definitive claims, TrustHire provides nuanced risk classifications (e.g., *Legitimate*, *Potentially Fraudulent*, *Higher Risk*, *Needs Caution*, *Possibly Legitimate*) to assist job seekers and placement officers in evaluating opportunity authenticity.

## Technology Stack
- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning & NLP**: Pandas, NumPy, Scikit-learn, TF-IDF, XGBoost, Joblib, Matplotlib, Seaborn
- **AI Explanation Layer**: Hugging Face Transformers (Local Model)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Database**: SQLite
- **Development & Version Control**: Git, GitHub, VS Code

## Development Status
- **Current Phase**: Step 1 — Project Setup & Foundation
- **Implementation Note**: The project is being developed incrementally. At this stage, the project directory structure, dependencies, configuration system, minimal FastAPI health endpoint (`GET /health`), basic frontend placeholder, and unit testing suite are initialized. Machine Learning models, dataset processing, TF-IDF feature extraction, and AI explanation layers will be implemented in subsequent development phases.

## Project Structure
```text
TrustHire/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application & /health endpoint
│   │   ├── config.py        # Centralized path & project configurations
│   │   ├── routes/          # API route definitions (future phases)
│   │   ├── services/        # Business logic & ML service integrations (future phases)
│   │   ├── models/          # Data schemas (future phases)
│   │   └── utils/           # Helper utility functions
│   └── requirements.txt
├── ml/
│   ├── data/
│   │   ├── raw/             # Raw EMSCAD dataset storage
│   │   └── processed/       # Cleaned & transformed dataset storage
│   ├── notebooks/           # Exploratory data analysis & model experiments
│   ├── preprocessing/       # Data cleaning & normalization modules
│   ├── features/            # Feature engineering & TF-IDF extraction
│   ├── training/            # ML model training scripts
│   ├── evaluation/          # Model metric evaluation & comparison
│   └── saved_models/        # Serialized model artifacts (.joblib)
├── ai/
│   ├── models/              # Local AI model storage
│   ├── services/            # Local transformer inference services
│   └── prompts/             # Prompt templates for risk explanation
├── frontend/
│   ├── index.html           # Project landing page foundation
│   ├── css/
│   │   └── style.css        # Core design styles
│   ├── js/
│   │   └── app.js           # Client-side JavaScript
│   └── assets/              # Web assets
├── database/                # SQLite database storage
├── tests/
│   ├── __init__.py
│   └── test_main.py         # Backend health endpoint unit test
├── README.md
├── .gitignore
└── requirements.txt
```

## Running the Project

### 1. Prerequisites
Ensure Python 3.9+ is installed.

### 2. Environment Setup & Dependency Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run FastAPI Backend Server
```bash
# From project root directory:
python -m uvicorn backend.app.main:app --reload
```
The health check endpoint will be accessible at: `http://127.0.0.1:8000/health`

### 4. Run Unit Tests
```bash
pytest
```
