"""
TrustHire - Stratified Train/Test Data Splitting Module
Handles splitting raw/preprocessed data safely into training and testing sets,
preventing data leakage.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from backend.app.config import PROCESSED_DATA_DIR


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "fraudulent",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Performs a stratified train/test split on the DataFrame.
    Returns:
        X_train, X_test, y_train, y_test, split_stats
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Calculate distribution statistics
    def calc_dist(series: pd.Series) -> Dict[str, Any]:
        counts = series.value_counts().to_dict()
        total = len(series)
        pcts = {k: round((v / total) * 100, 2) for k, v in counts.items()}
        return {
            "total": total,
            "counts": {int(k): int(v) for k, v in counts.items()},
            "percentages": {int(k): float(v) for k, v in pcts.items()},
        }

    stats = {
        "full_dataset": calc_dist(y),
        "train_set": calc_dist(y_train),
        "test_set": calc_dist(y_test),
        "test_size": test_size,
        "random_state": random_state,
    }

    return X_train, X_test, y_train, y_test, stats


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str = "fraudulent",
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[Path, Path]:
    """
    Saves the split train and test datasets into ml/data/processed/ as train.csv and test.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path
