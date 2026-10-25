"""
Unit tests for data loader and audit functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.config import TARGET_COL, ID_COL
from src.data_loader import load_raw_data, audit_dataset


def test_load_raw_data_shape_and_columns():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 7043
    assert df.shape[1] == 21
    assert ID_COL in df.columns
    assert TARGET_COL in df.columns


def test_no_duplicate_customer_ids():
    df = load_raw_data()
    assert df[ID_COL].duplicated().sum() == 0


def test_audit_dataset_output():
    df = load_raw_data()
    audit = audit_dataset(df)
    assert audit["total_records"] == 7043
    assert audit["total_features"] == 21
    assert audit["duplicate_records"] == 0
    assert audit["duplicate_customer_ids"] == 0
    assert audit["total_charges_blank_count"] == 11
