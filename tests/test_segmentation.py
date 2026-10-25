"""
Unit tests for unsupervised clustering and PCA segmentation.
"""

import pytest
import pandas as pd
from src.config import RESULTS_DIR, TARGET_COL, ID_COL


def test_segmentation_artifacts_exist():
    profiles_path = RESULTS_DIR / "cluster_profiles.csv"
    assignments_path = RESULTS_DIR / "cluster_assignments.csv"

    assert profiles_path.exists(), "cluster_profiles.csv must exist."
    assert assignments_path.exists(), "cluster_assignments.csv must exist."

    prof_df = pd.read_csv(profiles_path)
    assign_df = pd.read_csv(assignments_path)

    # Check total customers matches 7043
    assert assign_df.shape[0] == 7043
    assert prof_df["Size"].sum() == 7043

    # Check cluster counts
    assert len(prof_df) >= 2


def test_churn_not_in_clustering_features():
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_raw_data

    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)
    features_df = clean_df.drop(columns=[ID_COL, TARGET_COL])

    assert TARGET_COL not in features_df.columns
    assert ID_COL not in features_df.columns
