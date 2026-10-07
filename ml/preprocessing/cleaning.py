"""
TrustHire - Data Cleaning & Basic Transformation Module
Provides functions for duplicate removal, target encoding, binary feature conversion,
and salary range parsing.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Removes exact duplicate rows from the DataFrame.
    Returns the cleaned DataFrame and a dictionary with row counts.
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates().copy()
    final_rows = len(df_clean)
    removed_count = initial_rows - final_rows

    stats = {
        "rows_before": initial_rows,
        "rows_removed": removed_count,
        "rows_after": final_rows,
    }
    return df_clean, stats


def convert_target(df: pd.DataFrame, target_col: str = "fraudulent") -> pd.DataFrame:
    """
    Converts target column ('f'/'t' or 0/1) to binary integer 0/1 representation:
    - 0 = Legitimate
    - 1 = Fraudulent
    """
    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Standardize string representations
    target_series = df[target_col].astype(str).str.strip().str.lower()
    mapping = {
        "f": 0,
        "0": 0,
        "false": 0,
        "t": 1,
        "1": 1,
        "true": 1,
    }

    unmapped = set(target_series.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unexpected values found in target column: {unmapped}")

    df[target_col] = target_series.map(mapping).astype(int)
    return df


def clean_html_text(text: str) -> str:
    """
    Removes HTML tags, normalizes whitespace, and strips outer spaces.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Remove HTML tags using regex
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Replace multiple whitespace/newlines with single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def convert_binary_columns(
    df: pd.DataFrame, binary_cols: list = ["telecommuting", "has_company_logo", "has_questions"]
) -> pd.DataFrame:
    """
    Converts binary indicator columns ('f'/'t' or 0/1 or True/False) to integers 0/1.
    Raises ValueError if unexpected values are found.
    """
    df = df.copy()
    mapping = {
        "f": 0,
        "0": 0,
        "false": 0,
        0: 0,
        "t": 1,
        "1": 1,
        "true": 1,
        1: 1,
    }

    for col in binary_cols:
        if col in df.columns:
            series_str = df[col].astype(str).str.strip().str.lower()
            unmapped = set(series_str.unique()) - {"f", "0", "false", "t", "1", "true"}
            if unmapped:
                raise ValueError(f"Unexpected values in binary column '{col}': {unmapped}")

            df[col] = series_str.map(mapping).fillna(0).astype(int)

    return df


def parse_salary_range(salary_str: str) -> Tuple[int, float, float, float]:
    """
    Parses salary_range strings like '20000-28000' or '10000-10000'.
    Returns a tuple: (salary_available, salary_min, salary_max, salary_mid).
    If missing, empty, or invalid, returns (0, 0.0, 0.0, 0.0).
    """
    if not isinstance(salary_str, str) or pd.isna(salary_str) or not salary_str.strip():
        return (0, 0.0, 0.0, 0.0)

    clean_str = salary_str.strip()
    match = re.match(r"^(\d+)\s*-\s*(\d+)$", clean_str)
    if match:
        try:
            s_min = float(match.group(1))
            s_max = float(match.group(2))
            s_mid = (s_min + s_max) / 2.0
            return (1, s_min, s_max, s_mid)
        except ValueError:
            return (0, 0.0, 0.0, 0.0)

    # If single number or non-standard format
    numbers = re.findall(r"\d+", clean_str)
    if len(numbers) == 1:
        val = float(numbers[0])
        return (1, val, val, val)
    elif len(numbers) >= 2:
        val1, val2 = float(numbers[0]), float(numbers[1])
        s_min, s_max = min(val1, val2), max(val1, val2)
        return (1, s_min, s_max, (s_min + s_max) / 2.0)

    return (0, 0.0, 0.0, 0.0)


def apply_salary_features(df: pd.DataFrame, salary_col: str = "salary_range") -> pd.DataFrame:
    """
    Adds parsed salary features (salary_available, salary_min, salary_max, salary_mid) to DataFrame.
    Does NOT modify the original salary_range column.
    """
    df = df.copy()
    if salary_col not in df.columns:
        df["salary_available"] = 0
        df["salary_min"] = 0.0
        df["salary_max"] = 0.0
        df["salary_mid"] = 0.0
        return df

    parsed = df[salary_col].apply(parse_salary_range)
    df["salary_available"] = [p[0] for p in parsed]
    df["salary_min"] = [p[1] for p in parsed]
    df["salary_max"] = [p[2] for p in parsed]
    df["salary_mid"] = [p[3] for p in parsed]

    return df
