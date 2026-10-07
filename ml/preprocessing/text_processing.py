"""
TrustHire - Text Preprocessing & Feature Extraction Module
Provides functions to clean text fields, concatenate text columns safely,
and extract text-based metadata features.
"""

import re
import pandas as pd
from typing import List
from ml.preprocessing.cleaning import clean_html_text

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas text Series by filling NaNs with empty string
    and removing HTML tags & excess whitespace.
    """
    return series.fillna("").astype(str).apply(clean_html_text)


def create_combined_text(df: pd.DataFrame, text_cols: List[str] = TEXT_COLUMNS) -> pd.Series:
    """
    Combines text columns into a single structured string per posting.
    Format:
    TITLE: {title} | COMPANY_PROFILE: {company_profile} | DESCRIPTION: {description} | REQUIREMENTS: {requirements} | BENEFITS: {benefits}
    """
    combined_parts = []

    for col in text_cols:
        if col in df.columns:
            cleaned_col = clean_text_series(df[col])
            header = col.upper()
            part = header + ": " + cleaned_col
            combined_parts.append(part)

    if not combined_parts:
        return pd.Series([""] * len(df), index=df.index)

    # Join text parts with ' | ' separator to prevent token merging across fields
    combined = combined_parts[0]
    for part in combined_parts[1:]:
        combined = combined + " | " + part

    return combined


def count_urls(text: str) -> int:
    """
    Counts URL patterns (http, https, www) in a text string.
    """
    if not isinstance(text, str) or not text:
        return 0
    pattern = r"https?://\S+|www\.\S+"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def extract_text_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numerical metadata features derived from text fields.
    Features:
    - text_length: character length of combined text
    - title_length: character length of title
    - description_length: character length of description
    - requirements_length: character length of requirements
    - word_count: total word count of combined text
    - url_count: count of URLs in combined text
    - missing_field_count: number of empty/NaN fields in key attributes
    """
    df = df.copy()

    # Ensure clean text fields exist
    title_clean = clean_text_series(df["title"]) if "title" in df.columns else pd.Series([""] * len(df))
    desc_clean = clean_text_series(df["description"]) if "description" in df.columns else pd.Series([""] * len(df))
    req_clean = clean_text_series(df["requirements"]) if "requirements" in df.columns else pd.Series([""] * len(df))

    combined = create_combined_text(df)

    df["combined_text"] = combined
    df["text_length"] = combined.apply(len)
    df["title_length"] = title_clean.apply(len)
    df["description_length"] = desc_clean.apply(len)
    df["requirements_length"] = req_clean.apply(len)
    df["word_count"] = combined.apply(lambda x: len(x.split()) if x else 0)
    df["url_count"] = combined.apply(count_urls)

    # Missing field count (count of NaNs or whitespace-only across main text & cat columns)
    all_check_cols = [c for c in TEXT_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns]
    missing_mask = df[all_check_cols].isnull() | (df[all_check_cols].astype(str).apply(lambda s: s.str.strip() == ""))
    df["missing_field_count"] = missing_mask.sum(axis=1)

    return df
