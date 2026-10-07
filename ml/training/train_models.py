"""
TrustHire - Model Training Module
Implements model initialization, baseline training, and artifact serialization
for candidate algorithms:
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes
4. Random Forest
5. XGBoost
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from backend.app.config import SAVED_MODELS_DIR

# Check XGBoost availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """
    Trains Logistic Regression as the primary candidate model.
    Uses class_weight='balanced' to handle positive class imbalance (4.86%).
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svm(X_train, y_train, random_state: int = 42) -> LinearSVC:
    """
    Trains Linear SVM (LinearSVC).
    Uses class_weight='balanced' for class imbalance.
    Note: LinearSVC decision_function will be used for ranking/AUC scores.
    """
    model = LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
        C=1.0,
        dual="auto",
    )
    model.fit(X_train, y_train)
    return model


def train_multinomial_nb(X_train_nonneg, y_train, alpha: float = 0.1) -> MultinomialNB:
    """
    Trains Multinomial Naive Bayes on non-negative scaled feature matrix.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_nonneg, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """
    Trains Random Forest classifier with class_weight='balanced'.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42) -> Any:
    """
    Trains XGBoost classifier if installed.
    Calculates scale_pos_weight = (count(0) / count(1)) for class imbalance.
    """
    if not XGBOOST_AVAILABLE:
        print("Warning: XGBoost package is not currently installed.")
        return None

    # Calculate scale_pos_weight ratio
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train, y_train, X_train_nonneg, random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains all candidate models and returns a dictionary of trained model instances.
    """
    models = {}

    print("Training Logistic Regression...")
    models["logistic_regression"] = train_logistic_regression(X_train, y_train, random_state)

    print("Training Linear SVM...")
    models["linear_svm"] = train_linear_svm(X_train, y_train, random_state)

    print("Training Multinomial Naive Bayes...")
    models["naive_bayes"] = train_multinomial_nb(X_train_nonneg, y_train)

    print("Training Random Forest...")
    models["random_forest"] = train_random_forest(X_train, y_train, random_state)

    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        models["xgboost"] = train_xgboost(X_train, y_train, random_state)
    else:
        print("XGBoost package is not currently installed. Skipping XGBoost.")

    return models


def save_models(models_dict: Dict[str, Any], output_dir: Path = SAVED_MODELS_DIR) -> Dict[str, Path]:
    """
    Saves each trained model instance as a .joblib file in ml/saved_models/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for name, model in models_dict.items():
        if model is not None:
            file_path = output_dir / f"{name}.joblib"
            joblib.dump(model, file_path)
            saved_paths[name] = file_path

    return saved_paths
