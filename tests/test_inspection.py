"""
TrustHire - EMSCAD Dataset Inspection Tests
"""

import pytest
from pathlib import Path
import pandas as pd

from ml.inspect_dataset import (
    find_raw_dataset,
    load_dataset,
    inspect_schema,
    analyze_missing_values,
    analyze_duplicates,
    analyze_target,
    generate_dataset_report,
    TARGET_COLUMN,
)


def test_locate_csv():
    """Verify that the raw CSV dataset file can be located."""
    csv_path = find_raw_dataset()
    assert csv_path.exists()
    assert csv_path.suffix.lower() == ".csv"


def test_load_dataset():
    """Verify that the CSV loads successfully into a non-empty DataFrame."""
    df, csv_path = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) > 0


def test_target_column_detected():
    """Verify that the target column ('fraudulent') exists in the dataset."""
    df, _ = load_dataset()
    assert TARGET_COLUMN in df.columns
    target_info = analyze_target(df)
    assert target_info["exists"] is True


def test_schema_inspection():
    """Verify that schema inspection runs and identifies expected EMSCAD columns."""
    df, _ = load_dataset()
    schema_info = inspect_schema(df)
    assert "status" in schema_info
    assert len(schema_info["expected_columns_found"]) > 0


def test_report_generation(tmp_path):
    """Verify that dataset report generation produces JSON and Markdown outputs."""
    df, csv_path = load_dataset()
    json_path = tmp_path / "test_report.json"
    md_path = tmp_path / "test_report.md"

    j_out, m_out = generate_dataset_report(df, csv_path, json_path=json_path, md_path=md_path)
    assert j_out.exists()
    assert m_out.exists()
    assert j_out.stat().st_size > 0
    assert m_out.stat().st_size > 0
