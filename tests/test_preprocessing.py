"""
Unit tests for data cleaning, preprocessing, and stratified splitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import clean_raw_data, prepare_data_splits
from src.data_loader import load_raw_data
from src.config import ID_COL, TARGET_COL


def test_clean_raw_data_total_charges():
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)

    # Check TotalCharges has no NaN values after cleaning
    assert clean_df["TotalCharges"].isna().sum() == 0
    # Check that Target is integer 0 or 1
    assert set(clean_df[TARGET_COL].unique()).issubset({0, 1})


def test_leakage_free_stratified_split():
    splits = prepare_data_splits()

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    # Verify split sizes (80:20 of 7043 = 5634:1409)
    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert len(y_train) == 5634
    assert len(y_test) == 1409

    # Verify customerID and Churn are NOT in training features
    assert ID_COL not in X_train.columns
    assert TARGET_COL not in X_train.columns
    assert ID_COL not in X_test.columns
    assert TARGET_COL not in X_test.columns

    # Verify stratified balance in train and test splits
    train_rate = np.mean(y_train)
    test_rate = np.mean(y_test)
    assert abs(train_rate - test_rate) < 0.01
