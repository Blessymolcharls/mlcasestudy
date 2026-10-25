"""
Unit tests for evaluation metrics and model output assertions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RESULTS_DIR, CONFUSION_DIR


def test_model_comparison_file_exists_and_valid():
    comp_file = RESULTS_DIR / "model_comparison.csv"
    assert comp_file.exists(), "results/model_comparison.csv must exist."

    df = pd.read_csv(comp_file)
    assert len(df) == 4, "Must compare exactly four models."

    expected_models = {
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Multi-Layer Perceptron"
    }
    assert set(df["Model"].unique()) == expected_models

    for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Specificity"]:
        assert metric in df.columns
        assert (df[metric] >= 0.0).all() and (df[metric] <= 1.0).all()


def test_confusion_matrices_shape():
    for name in ["logistic_regression", "support_vector_machine", "random_forest", "multi-layer_perceptron"]:
        cm_path = CONFUSION_DIR / f"cm_{name}.csv"
        assert cm_path.exists(), f"Confusion matrix {cm_path} must exist."
        cm_df = pd.read_csv(cm_path, index_col=0)
        assert cm_df.shape == (2, 2)
        assert cm_df.values.sum() == 1409  # Total test set size
