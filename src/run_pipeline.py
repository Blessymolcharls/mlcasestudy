"""
Master Execution Pipeline for Customer Churn Case Study.
Runs data audit, preprocessing, EDA, model training, evaluation, feature analysis,
customer segmentation, and generates all report artifacts.
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import (
    RESULTS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    CV_FOLDS
)
from src.data_loader import load_raw_data, audit_dataset
from src.preprocessing import prepare_data_splits
from src.eda import run_eda
from src.generate_architecture_diagram import draw_system_architecture
from src.train_models import train_and_cross_validate
from src.evaluate_models import evaluate_all_models
from src.feature_analysis import run_feature_analysis
from src.segmentation import run_customer_segmentation
from src.report_data import generate_report_macros


def run_full_pipeline():
    """Execute complete reproducible machine learning workflow."""
    start_time = time.perf_counter()
    print("=" * 70)
    print("STARTING END-TO-END TELECOM CHURN ML PIPELINE")
    print("=" * 70)

    # 1. Dataset Acquisition & Audit
    print("\n[Step 1/8] Auditing raw dataset...")
    raw_df = load_raw_data()
    audit_report = audit_dataset(raw_df)
    print(f"  Rows: {audit_report['total_records']}, Columns: {audit_report['total_features']}")

    # 2. Data Preprocessing & Leakage-Free Splitting
    print("\n[Step 2/8] Executing leakage-free preprocessing and stratified split...")
    splits = prepare_data_splits()
    print(f"  Training set: {splits['X_train'].shape}, Test set: {splits['X_test'].shape}")

    # 3. Exploratory Data Analysis Visualizations
    print("\n[Step 3/8] Generating publication-quality EDA figures...")
    run_eda()

    # 4. System Architecture Diagram
    print("\n[Step 4/8] Generating system architecture diagram...")
    draw_system_architecture()

    # 5. Model Training & Cross Validation (4 Classifiers)
    print("\n[Step 5/8] Training 4 primary classification models...")
    trained_artifacts = train_and_cross_validate(splits["X_train"], splits["y_train"])

    # 6. Model Evaluation on Held-Out Test Set
    print("\n[Step 6/8] Evaluating models on held-out test data (N=1,409)...")
    comparison_df, predictions_df = evaluate_all_models(
        trained_artifacts,
        splits["X_test"],
        splits["y_test"],
        splits["ids_test"]
    )
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_artifacts[best_model_name]["model"]
    print(f"  Best model selected by F1-Score: {best_model_name}")
    print(comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])

    # 7. Feature Importance & Interpretation
    print("\n[Step 7/8] Analyzing feature importance on held-out test set...")
    importance_df = run_feature_analysis(
        best_model_name,
        best_model,
        trained_artifacts,
        splits["X_test"],
        splits["y_test"]
    )

    # 8. Unsupervised Customer Segmentation (K-Means & PCA)
    print("\n[Step 8/8] Executing K-Means customer segmentation & PCA...")
    seg_results = run_customer_segmentation()

    # 9. LaTeX Macro Generation & Experiment Summary
    print("\nGenerating report macros and experiment summary JSON...")
    generate_report_macros()

    total_pipeline_time = time.perf_counter() - start_time
    summary = {
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(total_pipeline_time, 2),
        "dataset": {
            "name": "IBM Telco Customer Churn",
            "total_records": audit_report["total_records"],
            "total_features": audit_report["total_features"],
            "train_size": len(splits["X_train"]),
            "test_size": len(splits["X_test"]),
            "random_seed": RANDOM_SEED,
            "test_split_ratio": TEST_SIZE,
            "cv_folds": CV_FOLDS
        },
        "best_classification_model": {
            "name": best_model_name,
            "f1_score": float(comparison_df.iloc[0]["F1-score"]),
            "roc_auc": float(comparison_df.iloc[0]["ROC-AUC"]),
            "accuracy": float(comparison_df.iloc[0]["Accuracy"]),
            "precision": float(comparison_df.iloc[0]["Precision"]),
            "recall": float(comparison_df.iloc[0]["Recall"]),
            "pr_auc": float(comparison_df.iloc[0]["PR-AUC"])
        },
        "all_models_summary": comparison_df.to_dict(orient="records"),
        "unsupervised_segmentation": {
            "selected_k": seg_results["chosen_k"],
            "pca_explained_variance_ratio": seg_results["pca_variance"],
            "cluster_profiles": seg_results["profile_df"].to_dict(orient="records")
        }
    }

    with open(RESULTS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_pipeline_time:.2f} SECONDS")
    print(f"Experiment summary saved to: {RESULTS_DIR / 'experiment_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
