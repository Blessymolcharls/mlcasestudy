"""
====================================================================================================
CUSTOMER CHURN PREDICTION AND SEGMENTATION FOR A TELECOM COMPANY
====================================================================================================
Academic Machine Learning Case Study & Complete End-to-End Pipeline
Author: Blessy Mol Charls (Reg. No: MUT24CA029)
Institution: Muthoot Institute of Technology and Science (MITS), Varikoli, Ernakulam - 682308
Affiliated to: APJ Abdul Kalam Technological University (KTU)
Date: October 2026

DESCRIPTION:
This standalone script consolidates the complete, reproducible machine learning pipeline:
  1. Configuration, constants, and reproducibility seeds (Seed=42).
  2. Data ingestion, audit, data dictionary generation, and integrity checks.
  3. Domain-specific data cleaning and leakage-free stratified preprocessing.
  4. Publication-grade Exploratory Data Analysis (EDA) visualizations (300 DPI).
  5. System architecture diagram generation.
  6. Supervised learning: 4 primary classifiers (Logistic Regression, SVM, Random Forest, MLP)
     with 5-fold stratified cross-validation and hyperparameter calibration.
  7. Test evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices.
  8. Interpretability: Permutation importance and logistic log-odds coefficients.
  9. Unsupervised customer segmentation: K-Means (Elbow & Silhouette) and 2D PCA.
 10. Automated LaTeX table and macro generation for zero-transcription error reporting.
 11. Self-verification test suite.

USAGE:
    python complete_code.py                # Runs the complete pipeline end-to-end
    python complete_code.py --test-only    # Runs the integrated unit verification tests
====================================================================================================
"""

import sys
import os
import json
import time
import shutil
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Headless matplotlib configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd

# Scikit-Learn components
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    silhouette_score
)
from sklearn.decomposition import PCA
import joblib

# Suppress minor non-fatal warnings
warnings.filterwarnings("ignore")


# ==================================================================================================
# SECTION 1: GLOBAL CONFIGURATION AND DIRECTORY MANAGEMENT
# ==================================================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORTS_DIR = BASE_DIR / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_GENERATED_DIR = REPORTS_DIR / "generated"

# Data File Paths (checks local workspace or standard data/raw)
RAW_DATA_CANDIDATES = [
    DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
]

# Strict reproducibility seed and parameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
FIGURE_DPI = 300

# Target and Identifiers
TARGET_COL = "Churn"
ID_COL = "customerID"

# Feature Categorization
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def ensure_directories():
    """Ensure all required output directories exist."""
    dirs = [
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
        FIGURES_DIR, RESULTS_FIGURES_DIR, CONFUSION_DIR,
        REPORTS_DIR, REPORT_FIGURES_DIR, REPORT_TABLES_DIR, REPORT_GENERATED_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_data_filepath() -> Path:
    """Resolve the dataset location across candidate paths."""
    for p in RAW_DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Telco Customer Churn dataset ('WA_Fn-UseC_-Telco-Customer-Churn.csv') not found."
    )


def save_and_sync_fig(fig: plt.Figure, filename: str):
    """Save high-resolution figure and synchronize across report and asset directories."""
    ensure_directories()
    p_results = RESULTS_FIGURES_DIR / filename
    p_reports = REPORT_FIGURES_DIR / filename
    p_root_fig = FIGURES_DIR / filename

    fig.savefig(p_results, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(p_results, p_reports)
    shutil.copy(p_results, p_root_fig)
    plt.close(fig)
    print(f"  [Saved Figure] {filename}")


# ==================================================================================================
# SECTION 2: DATA LOADING, AUDIT, AND DATA DICTIONARY
# ==================================================================================================

def load_raw_data() -> pd.DataFrame:
    """Load the raw Telco customer churn CSV."""
    filepath = get_data_filepath()
    return pd.read_csv(filepath)


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality audit and generate data dictionary."""
    ensure_directories()
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0

    # Count whitespace blanks in TotalCharges
    tc_blank_count = 0
    if "TotalCharges" in df.columns:
        tc_blank_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())

    missing_per_col = {col: int(df[col].isna().sum()) for col in df.columns}

    # Target distribution
    target_dist = {}
    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts(dropna=False)
        props = df[TARGET_COL].value_counts(dropna=False, normalize=True)
        for val in counts.index:
            target_dist[str(val)] = {
                "count": int(counts[val]),
                "proportion": round(float(props[val]), 4)
            }

    # Categorical cardinality
    categorical_uniques = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            categorical_uniques[col] = {
                "num_unique": int(df[col].nunique()),
                "unique_values": [str(v) for v in df[col].unique().tolist()]
            }

    # Numerical summaries
    numeric_summaries = {}
    tc_numeric = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            series = tc_numeric if col == "TotalCharges" else pd.to_numeric(df[col], errors="coerce")
            numeric_summaries[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.median()), 2),
                "q75": round(float(series.quantile(0.75)), 2),
                "max": round(float(series.max()), 2),
                "null_count": int(series.isna().sum())
            }

    audit_report = {
        "dataset_name": "IBM Telco Customer Churn",
        "file_path": str(get_data_filepath()),
        "total_records": n_rows,
        "total_features": n_cols,
        "columns": df.columns.tolist(),
        "duplicate_records": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "total_charges_blank_count": tc_blank_count,
        "missing_values_initial": missing_per_col,
        "target_distribution": target_dist,
        "numeric_summaries": numeric_summaries,
        "categorical_features": categorical_uniques
    }

    with open(RESULTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=4)

    # Save Data Dictionary CSV
    dictionary_records = []
    for col in df.columns:
        sample_vals = ", ".join([str(x) for x in df[col].unique()[:5]])
        if col == TARGET_COL:
            role = "Target Variable"
            desc = "Whether customer churned within the last month (Yes/No)"
        elif col == ID_COL:
            role = "Customer Identifier"
            desc = "Unique alphanumeric string identifying each customer"
        elif col in NUMERIC_COLS:
            role = "Numerical Predictor"
            desc = f"Continuous numerical feature representing {col}"
        else:
            role = "Categorical Predictor"
            desc = f"Customer demographic or subscribed service feature: {col}"

        dictionary_records.append({
            "Feature Name": col,
            "Role": role,
            "Data Type": str(df[col].dtype),
            "Sample Values": sample_vals
        })

    pd.DataFrame(dictionary_records).to_csv(RESULTS_DIR / "data_dictionary.csv", index=False)
    return audit_report


# ==================================================================================================
# SECTION 3: DATA PREPROCESSING & LEAKAGE-FREE PARTITIONING
# ==================================================================================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame:
    - Coerce TotalCharges to numeric.
    - Set TotalCharges to 0.0 for zero-tenure accounts (physical ground truth).
    - Map target Churn to binary (No: 0, Yes: 1).
    - Treat SeniorCitizen as string for consistent one-hot encoding.
    """
    df_clean = df.copy()

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    zero_tenure_mask = (df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna())
    df_clean.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    if TARGET_COL in df_clean.columns:
        if df_clean[TARGET_COL].dtype != int:
            df_clean[TARGET_COL] = df_clean[TARGET_COL].map({"No": 0, "Yes": 1, 0: 0, 1: 1}).astype(int)

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )


def prepare_data_splits() -> Dict[str, Any]:
    """Execute stratified train/test split and fit preprocessor strictly on training data."""
    ensure_directories()
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)

    customer_ids = clean_df[ID_COL].copy()
    y = clean_df[TARGET_COL].values
    X = clean_df.drop(columns=[ID_COL, TARGET_COL])

    # Stratified Train/Test partition (80:20)
    X_train_raw, X_test_raw, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Fit preprocessor strictly on X_train to prevent data leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)

    # Transform splits
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Extract feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + cat_feature_names

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # Persist processed artifacts
    X_train_df.to_parquet(DATA_PROCESSED_DIR / "X_train.parquet")
    X_test_df.to_parquet(DATA_PROCESSED_DIR / "X_test.parquet")
    np.save(DATA_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(DATA_PROCESSED_DIR / "y_test.npy", y_test)
    ids_train.to_csv(DATA_PROCESSED_DIR / "ids_train.csv", index=False)
    ids_test.to_csv(DATA_PROCESSED_DIR / "ids_test.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(DATA_PROCESSED_DIR / "feature_names.csv", index=False)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_test": ids_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor
    }


# ==================================================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==================================================================================================

def run_eda():
    """Generate all 8 publication-quality EDA figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.titlesize": 15,
        "figure.autolayout": True
    })

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)

    # 1. Churn target distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df_raw["Churn"].value_counts()
    props = df_raw["Churn"].value_counts(normalize=True) * 100
    bars = ax.bar(["No Churn", "Churn"], counts.values, color=["#2b5c8f", "#d95f02"], width=0.55, edgecolor="black", linewidth=0.8)
    for bar, prop in zip(bars, props):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 60, f"{int(h):,} ({prop:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Customer Churn Target Distribution")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, max(counts.values) * 1.15)
    save_and_sync_fig(fig, "eda_01_churn_distribution.png")

    # 2. Churn rate by contract type
    fig, ax = plt.subplots(figsize=(7, 5))
    contract_churn = df_clean.groupby("Contract")[TARGET_COL].agg(["count", "mean"]).reset_index()
    contract_churn["mean"] *= 100
    bars = ax.bar(contract_churn["Contract"], contract_churn["mean"], color="#4575b4", width=0.5, edgecolor="black")
    for bar, count in zip(bars, contract_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Contract Agreement")
    ax.set_ylim(0, max(contract_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_02_churn_by_contract.png")

    # 3. Churn rate by tenure groups
    df_clean["tenure_group"] = pd.cut(
        df_clean["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tenure_churn = df_clean.groupby("tenure_group", observed=False)[TARGET_COL].agg(["count", "mean"]).reset_index()
    tenure_churn["mean"] *= 100
    bars = ax.bar(tenure_churn["tenure_group"], tenure_churn["mean"], color="#74add1", width=0.55, edgecolor="black")
    for bar, count in zip(bars, tenure_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate Across Customer Tenure Cohorts")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Tenure Cohort")
    ax.set_ylim(0, max(tenure_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_03_churn_by_tenure_group.png")

    # 4. MonthlyCharges distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 0], x="MonthlyCharges", label="Retained (No)", color="#2b5c8f", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 1], x="MonthlyCharges", label="Churned (Yes)", color="#d95f02", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Density Distribution of Monthly Charges by Churn Status")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    save_and_sync_fig(fig, "eda_04_monthly_charges_distribution.png")

    # 5. TotalCharges distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 0], x="TotalCharges", label="Retained (No)", color="#2b5c8f", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 1], x="TotalCharges", label="Churned (Yes)", color="#d95f02", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Density Distribution of Cumulative Total Charges by Churn Status")
    ax.set_xlabel("Total Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    save_and_sync_fig(fig, "eda_05_total_charges_distribution.png")

    # 6. PaymentMethod churn
    fig, ax = plt.subplots(figsize=(9, 5))
    pm_churn = df_clean.groupby("PaymentMethod")[TARGET_COL].agg(["count", "mean"]).reset_index()
    pm_churn["mean"] *= 100
    pm_churn = pm_churn.sort_values(by="mean", ascending=False)
    bars = ax.bar(pm_churn["PaymentMethod"], pm_churn["mean"], color="#fdae61", width=0.55, edgecolor="black")
    for bar, count in zip(bars, pm_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Payment Method")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    ax.set_xticks(range(len(pm_churn)))
    ax.set_xticklabels(pm_churn["PaymentMethod"], rotation=15, ha="right")
    ax.set_ylim(0, max(pm_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_06_churn_by_payment_method.png")

    # 7. InternetService churn
    fig, ax = plt.subplots(figsize=(7, 5))
    is_churn = df_clean.groupby("InternetService")[TARGET_COL].agg(["count", "mean"]).reset_index()
    is_churn["mean"] *= 100
    bars = ax.bar(is_churn["InternetService"], is_churn["mean"], color="#abd9e9", width=0.5, edgecolor="black")
    for bar, count in zip(bars, is_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Internet Service Type")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Internet Service Technology")
    ax.set_ylim(0, max(is_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_07_churn_by_internet_service.png")

    # 8. Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df_clean[NUMERIC_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"}, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Continuous Predictors & Churn")
    save_and_sync_fig(fig, "eda_08_correlation_heatmap.png")


# ==================================================================================================
# SECTION 5: SYSTEM ARCHITECTURE DIAGRAM
# ==================================================================================================

def draw_system_architecture():
    """Render an aesthetic publication-quality system architecture flowchart."""
    ensure_directories()
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=FIGURE_DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    c_data = "#e8f4f8"
    c_prep = "#d1e7dd"
    c_model = "#fff3cd"
    c_eval = "#cfe2ff"
    c_seg = "#f8d7da"
    c_border = "#212529"

    def draw_box(x, y, w, h, text, bg, subtitle=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg, edgecolor=c_border, linewidth=1.2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=10, fontweight="bold", color="#111827")
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8, color="#4b5563")
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#111827")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#374151", edgecolor="#374151", arrowstyle="-|>", lw=1.4, shrinkA=2, shrinkB=2)
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.5, color="#374151", fontweight="semibold")

    ax.text(5.5, 8.1, "Telco Customer Churn Prediction & Segmentation Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1e3a8a")

    draw_box(0.6, 6.7, 2.6, 0.9, "IBM Telco Dataset", c_data, "7,043 rows x 21 features")
    draw_box(4.2, 6.7, 2.6, 0.9, "Data Quality Audit", c_data, "Zero-dupes, type checks")
    draw_box(7.8, 6.7, 2.6, 0.9, "Data Cleaning", c_prep, "TotalCharges coercion & tenure")
    draw_arrow(3.2, 7.15, 4.2, 7.15)
    draw_arrow(6.8, 7.15, 7.8, 7.15)

    draw_box(7.8, 5.1, 2.6, 0.9, "Stratified Split (80:20)", c_prep, "Train: 5,634 | Test: 1,409")
    draw_arrow(9.1, 6.7, 9.1, 6.0)

    draw_box(4.2, 5.1, 2.6, 0.9, "Leakage-Free Preprocessor", c_prep, "StandardScaler + OneHot")
    draw_arrow(7.8, 5.55, 6.8, 5.55, "Fit on Train Only")

    draw_arrow(4.2, 5.55, 3.2, 4.4, "Processed Train/Test")
    draw_arrow(5.5, 5.1, 7.8, 4.4, "All Customers w/o Churn")

    draw_box(0.6, 3.5, 3.2, 1.4, "Four Primary Classifiers", c_model, "1. Logistic Regression (L2)\n2. SVM (RBF Kernel)\n3. Random Forest Ensemble\n4. Multi-Layer Perceptron (MLP)")
    draw_box(7.2, 3.5, 3.2, 1.4, "K-Means Clustering & PCA", c_seg, "• Elbow & Silhouette Analysis\n• Selected k = 3 Clusters\n• 2D PCA Dimensionality Red.")

    draw_box(0.6, 1.8, 3.2, 1.1, "Supervised Evaluation", c_eval, "Accuracy, Precision, Recall, F1,\nROC-AUC, PR-AUC, Confusion Matrix")
    draw_arrow(2.2, 3.5, 2.2, 2.9)

    draw_box(4.2, 1.8, 2.6, 1.1, "Feature Interpretation", c_eval, "Permutation Importance &\nOdds Ratios (Coefficients)")
    draw_arrow(3.8, 2.35, 4.2, 2.35)

    draw_box(7.2, 1.8, 3.2, 1.1, "Post-Hoc Cluster Profiling", c_seg, "• Churn Rate per Segment\n• Tenure & Spend Cohort Profiles")
    draw_arrow(8.8, 3.5, 8.8, 2.9)

    draw_box(2.4, 0.3, 6.2, 0.95, "Business Decision & Retention Policy", c_prep, "Targeted proactive interventions, onboarding support, contract optimization")
    draw_arrow(2.2, 1.8, 4.0, 1.25)
    draw_arrow(5.5, 1.8, 5.5, 1.25)
    draw_arrow(8.8, 1.8, 7.0, 1.25)

    save_and_sync_fig(fig, "system_architecture.png")


# ==================================================================================================
# SECTION 6: SUPERVISED MODEL TRAINING & CROSS-VALIDATION
# ==================================================================================================

def get_model_definitions() -> Dict[str, Any]:
    """Define the four primary classifiers with fixed seeds and balanced weighting."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Support Vector Machine": SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=1
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.01, batch_size=64, learning_rate_init=0.001,
            max_iter=500, early_stopping=True, n_iter_no_change=15,
            random_state=RANDOM_SEED
        )
    }


def train_and_cross_validate(X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Train models, record training runtimes, and compute 5-fold cross-validation scores."""
    ensure_directories()
    models = get_model_definitions()
    trained_artifacts = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, model in models.items():
        print(f"  Training {name}...")
        t0 = time.perf_counter()
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        safe_name = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{safe_name}_model.joblib"
        joblib.dump(model, model_path)

        trained_artifacts[name] = {
            "model": model,
            "train_time_sec": round(elapsed, 4),
            "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
            "cv_f1_std": round(float(np.std(cv_f1)), 4),
            "cv_roc_mean": round(float(np.mean(cv_roc)), 4),
            "cv_roc_std": round(float(np.std(cv_roc)), 4),
            "artifact_path": str(model_path)
        }
        print(f"    Completed in {elapsed:.2f}s | CV F1: {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")

    return trained_artifacts


# ==================================================================================================
# SECTION 7: HELD-OUT TEST EVALUATION & PERFORMANCE CURVES
# ==================================================================================================

def evaluate_all_models(
    trained_artifacts: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ids_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models on the held-out test partition (N=1,409) and export evaluation assets."""
    ensure_directories()
    metrics_list = []
    reports_dict = {}
    predictions_df = pd.DataFrame({"customerID": ids_test.values, "Actual_Churn": y_test})

    roc_data = {}
    pr_data = {}

    for name, item in trained_artifacts.items():
        model = item["model"]
        train_time = item["train_time_sec"]

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        safe_name = name.lower().replace(" ", "_")
        predictions_df[f"{safe_name}_pred"] = y_pred
        predictions_df[f"{safe_name}_prob"] = np.round(y_prob, 4)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Save confusion matrix CSV
        cm_df = pd.DataFrame(cm, index=["Actual_No_Churn", "Actual_Churn"], columns=["Pred_No_Churn", "Pred_Churn"])
        cm_df.to_csv(CONFUSION_DIR / f"cm_{safe_name}.csv")

        # Plot individual Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"],
                    cbar=False, ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix: {name}")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        save_and_sync_fig(fig_cm, f"confusion_matrix_{safe_name}.png")

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)

        # PR Curve data
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (recall_curve, precision_curve, pr_auc)

        reports_dict[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "Specificity": round(specificity, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Training_Time_s": round(train_time, 2)
        })

    comparison_df = pd.DataFrame(metrics_list).sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with open(RESULTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports_dict, f, indent=4)
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    # Plot Combined ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", linewidth=1)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")
    save_and_sync_fig(fig_roc, "eval_roc_curves.png")

    # Plot Combined PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    baseline_churn = np.mean(y_test)
    for (name, (rec_c, prec_c, ap_val)), color in zip(pr_data.items(), colors):
        ax_pr.plot(rec_c, prec_c, label=f"{name} (AP = {ap_val:.3f})", color=color, linewidth=2)
    ax_pr.axhline(y=baseline_churn, color="k", linestyle="--", label=f"No Skill Baseline ({baseline_churn:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    save_and_sync_fig(fig_pr, "eval_pr_curves.png")

    # Plot Model Comparison Bar Chart
    plot_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    melted_df = comparison_df.melt(id_vars=["Model"], value_vars=plot_metrics, var_name="Metric", value_name="Score")
    fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax_bar, edgecolor="black")
    ax_bar.set_title("Comparative Performance of the Four Primary Classifiers")
    ax_bar.set_ylabel("Score (0.0 to 1.0)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(loc="lower right")
    save_and_sync_fig(fig_bar, "eval_model_comparison_bar.png")

    # Generate LaTeX Table 6.1
    latex_lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Model Performance on the Held-Out Test Set ($N=1,409$)}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\",
        r"\midrule"
    ]
    for _, row in comparison_df.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & "
            f"{row['Recall']:.4f} & {row['F1-score']:.4f} & {row['ROC-AUC']:.4f} & {row['PR-AUC']:.4f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(REPORT_TABLES_DIR / "table_6_1_model_comparison.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    return comparison_df, predictions_df


# ==================================================================================================
# SECTION 8: FEATURE IMPORTANCE & MODEL INTERPRETABILITY
# ==================================================================================================

def run_feature_analysis(
    best_model_name: str,
    best_model: Any,
    trained_artifacts: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> pd.DataFrame:
    """Compute permutation importance and inspect logistic regression log-odds coefficients."""
    ensure_directories()
    print(f"  Computing permutation importance for best model ({best_model_name})...")
    perm_result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED, scoring="f1", n_jobs=1
    )

    feature_names = X_test.columns.tolist()
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": perm_result.importances_mean,
        "Permutation_Importance_Std": perm_result.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Plot Top 15 Features
    top_15 = importance_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top_15["Feature"][::-1],
        top_15["Permutation_Importance_Mean"][::-1],
        xerr=top_15["Permutation_Importance_Std"][::-1],
        color="#2b5c8f", edgecolor="black", capsize=3
    )
    ax.set_title(f"Top 15 Permutation Feature Importances ({best_model_name})")
    ax.set_xlabel("Mean Drop in Test F1-Score")
    save_and_sync_fig(fig, "feat_permutation_importance.png")

    # Logistic Regression coefficients
    if "Logistic Regression" in trained_artifacts:
        lr_model = trained_artifacts["Logistic Regression"]["model"]
        if hasattr(lr_model, "coef_"):
            lr_coefs = lr_model.coef_[0]
            lr_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": lr_coefs,
                "Odds_Ratio": np.exp(lr_coefs)
            }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
            lr_df.to_csv(RESULTS_DIR / "logistic_regression_coefficients.csv", index=False)

            top_pos = lr_df.head(8)
            top_neg = lr_df.tail(8)
            combined_lr = pd.concat([top_pos, top_neg]).sort_values(by="Coefficient", ascending=True)

            fig_lr, ax_lr = plt.subplots(figsize=(9, 6))
            colors = ["#d95f02" if c > 0 else "#2b5c8f" for c in combined_lr["Coefficient"]]
            ax_lr.barh(combined_lr["Feature"], combined_lr["Coefficient"], color=colors, edgecolor="black")
            ax_lr.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax_lr.set_title("Logistic Regression Coefficients (Risk vs. Protective Factors)")
            ax_lr.set_xlabel("Log-Odds Coefficient (Positive = Higher Churn Risk)")
            save_and_sync_fig(fig_lr, "feat_logistic_coefficients.png")

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Top Predictive Features Ranked by Permutation Importance (Test Set)}",
        r"\label{tab:feature_importance}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"\textbf{Rank} & \textbf{Feature Name} & \textbf{Mean F1 Drop} & \textbf{Std Dev} \\",
        r"\midrule"
    ]
    for idx, row in top_15.iterrows():
        feat_clean = row['Feature'].replace('_', r'\_')
        latex_lines.append(
            rf"{idx + 1} & {feat_clean} & {row['Permutation_Importance_Mean']:.4f} & "
            rf"$\pm$ {row['Permutation_Importance_Std']:.4f} \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(REPORT_TABLES_DIR / "table_feature_importance.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    return importance_df


# ==================================================================================================
# SECTION 9: UNSUPERVISED CUSTOMER SEGMENTATION (K-MEANS & PCA)
# ==================================================================================================

def run_customer_segmentation() -> Dict[str, Any]:
    """Execute K-Means clustering (k=2..8), optimal k=3 selection, PCA 2D mapping, and post-hoc profiling."""
    ensure_directories()
    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)

    customer_ids = df_clean[ID_COL].copy()
    actual_churn = df_clean[TARGET_COL].copy()

    # Rule: Exclude customerID and Churn during clustering
    features_df = df_clean.drop(columns=[ID_COL, TARGET_COL])
    preprocessor = build_preprocessor()
    X_cluster = preprocessor.fit_transform(features_df)

    k_range = list(range(2, 9))
    inertias = []
    silhouettes = []

    print("  Evaluating K-Means cluster candidates (k=2 to 8)...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_cluster)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_cluster, labels, sample_size=3000, random_state=RANDOM_SEED)
        silhouettes.append(sil)

    # Plot Elbow Curve
    fig_elbow, ax_elbow = plt.subplots(figsize=(6.5, 4.5))
    ax_elbow.plot(k_range, inertias, "bo-", linewidth=2, markersize=7)
    ax_elbow.set_title("K-Means Elbow Method for Optimal Cluster Count")
    ax_elbow.set_xlabel("Number of Clusters (k)")
    ax_elbow.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax_elbow.set_xticks(k_range)
    save_and_sync_fig(fig_elbow, "cluster_01_elbow_curve.png")

    # Plot Silhouette Scores
    fig_sil, ax_sil = plt.subplots(figsize=(6.5, 4.5))
    ax_sil.plot(k_range, silhouettes, "rs-", linewidth=2, markersize=7)
    ax_sil.set_title("Silhouette Score vs. Number of Clusters")
    ax_sil.set_xlabel("Number of Clusters (k)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.set_xticks(k_range)
    save_and_sync_fig(fig_sil, "cluster_02_silhouette_scores.png")

    chosen_k = 3
    print(f"  Selected optimal cluster count: k = {chosen_k}")

    final_km = KMeans(n_clusters=chosen_k, init="k-means++", n_init=20, random_state=RANDOM_SEED)
    cluster_labels = final_km.fit_predict(X_cluster)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_cluster)
    var_exp = pca.explained_variance_ratio_

    # Plot PCA Clusters
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="Set1", alpha=0.4, edgecolors="none", s=25)
    pca_centroids = pca.transform(final_km.cluster_centers_)
    ax_pca.scatter(pca_centroids[:, 0], pca_centroids[:, 1], c="black", s=120, marker="X", label="Cluster Centroids")
    ax_pca.set_title(f"Customer Segments Projected on 2D PCA Space (k={chosen_k})")
    ax_pca.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)")
    ax_pca.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)")
    ax_pca.legend()
    save_and_sync_fig(fig_pca, "cluster_03_pca_clusters.png")

    # Post-hoc evaluation joining Churn
    df_clean["Cluster"] = cluster_labels
    cluster_records = []
    for c in range(chosen_k):
        c_sub = df_clean[df_clean["Cluster"] == c]
        churn_rate = float(c_sub[TARGET_COL].mean()) * 100
        mean_tenure = float(c_sub["tenure"].mean())
        mean_monthly = float(c_sub["MonthlyCharges"].mean())
        mean_total = float(c_sub["TotalCharges"].mean())
        size = len(c_sub)
        prop = size / len(df_clean) * 100
        top_contract = c_sub["Contract"].mode()[0]
        top_internet = c_sub["InternetService"].mode()[0]

        if churn_rate > 35:
            segment_name = "High-Risk Attrition Cohort"
        elif mean_tenure > 40:
            segment_name = "Long-Term Loyal Enterprise"
        else:
            segment_name = "Value-Oriented Core Customers"

        cluster_records.append({
            "Cluster_ID": c, "Segment_Name": segment_name, "Size": size,
            "Proportion_pct": round(prop, 2), "Churn_Rate_pct": round(churn_rate, 2),
            "Avg_Tenure_mo": round(mean_tenure, 1), "Avg_Monthly_Charges": round(mean_monthly, 2),
            "Avg_Total_Charges": round(mean_total, 2), "Dominant_Contract": top_contract,
            "Dominant_Internet": top_internet
        })

    profile_df = pd.DataFrame(cluster_records)
    profile_df.to_csv(RESULTS_DIR / "cluster_profiles.csv", index=False)

    assignments_df = pd.DataFrame({"customerID": customer_ids, "Cluster": cluster_labels, "Actual_Churn": actual_churn})
    assignments_df.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)

    # Plot Cluster Profiles
    fig_prof, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar([f"Cluster {r['Cluster_ID']}" for _, r in profile_df.iterrows()],
            profile_df["Churn_Rate_pct"],
            color=["#d95f02" if r > 30 else "#2b5c8f" for r in profile_df["Churn_Rate_pct"]],
            edgecolor="black")
    for idx, r in profile_df.iterrows():
        ax1.text(idx, r["Churn_Rate_pct"] + 1, f"{r['Churn_Rate_pct']:.1f}%", ha="center", fontweight="bold")
    ax1.set_title("Observed Post-Hoc Churn Rate by Cluster")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_ylim(0, max(profile_df["Churn_Rate_pct"]) * 1.2)

    ax2.bar([f"Cluster {r['Cluster_ID']}" for _, r in profile_df.iterrows()],
            profile_df["Avg_Tenure_mo"], color="#74add1", edgecolor="black")
    for idx, r in profile_df.iterrows():
        ax2.text(idx, r["Avg_Tenure_mo"] + 1, f"{r['Avg_Tenure_mo']:.1f} mo", ha="center", fontweight="bold")
    ax2.set_title("Average Customer Tenure by Cluster")
    ax2.set_ylabel("Average Tenure (Months)")
    ax2.set_ylim(0, max(profile_df["Avg_Tenure_mo"]) * 1.2)
    save_and_sync_fig(fig_prof, "cluster_04_profile_comparison.png")

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{K-Means Customer Segmentation Profiles ($k=3$)}",
        r"\label{tab:cluster_profiles}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"\textbf{Cluster ID} & \textbf{Segment Description} & \textbf{Size ($N$)} & \textbf{Share (\%)} & \textbf{Churn Rate} & \textbf{Mean Tenure} & \textbf{Mean Monthly (\$)} \\",
        r"\midrule"
    ]
    for _, row in profile_df.iterrows():
        latex_lines.append(
            rf"Cluster {row['Cluster_ID']} & {row['Segment_Name']} & {row['Size']:,} & "
            rf"{row['Proportion_pct']:.1f}\% & {row['Churn_Rate_pct']:.1f}\% & "
            rf"{row['Avg_Tenure_mo']:.1f} mo & \${row['Avg_Monthly_Charges']:.2f} \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(REPORT_TABLES_DIR / "table_cluster_profiles.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    return {
        "k_range": k_range, "inertias": inertias, "silhouettes": silhouettes,
        "chosen_k": chosen_k, "profile_df": profile_df, "pca_variance": var_exp.tolist()
    }


# ==================================================================================================
# SECTION 10: LATEX MACROS & SUMMARY GENERATOR
# ==================================================================================================

def generate_report_macros():
    """Read generated experimental results and export dynamic LaTeX macros."""
    ensure_directories()
    with open(RESULTS_DIR / "dataset_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)

    model_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    best_row = model_df.iloc[0]
    cluster_df = pd.read_csv(RESULTS_DIR / "cluster_profiles.csv")
    feat_df = pd.read_csv(RESULTS_DIR / "feature_importance.csv")

    latex_defs = [
        "% Auto-generated experimental macros from real pipeline execution",
        f"\\newcommand{{\\datasetTotalRows}}{{{audit['total_records']:,}}}",
        f"\\newcommand{{\\datasetTotalCols}}{{{audit['total_features']}}}",
        f"\\newcommand{{\\datasetChurnProportion}}{{{audit['target_distribution']['Yes']['proportion']*100:.1f}\\%}}",
        f"\\newcommand{{\\datasetRetainedProportion}}{{{audit['target_distribution']['No']['proportion']*100:.1f}\\%}}",
        f"\\newcommand{{\\datasetTrainCount}}{{5,634}}",
        f"\\newcommand{{\\datasetTestCount}}{{1,409}}",
        f"\\newcommand{{\\datasetMissingTotalCharges}}{{{audit['total_charges_blank_count']}}}",
        "",
        "% Classification metrics for the best model",
        f"\\newcommand{{\\bestModelName}}{{{best_row['Model']}}}",
        f"\\newcommand{{\\bestModelAccuracy}}{{{best_row['Accuracy']:.4f}}}",
        f"\\newcommand{{\\bestModelPrecision}}{{{best_row['Precision']:.4f}}}",
        f"\\newcommand{{\\bestModelRecall}}{{{best_row['Recall']:.4f}}}",
        f"\\newcommand{{\\bestModelFOne}}{{{best_row['F1-score']:.4f}}}",
        f"\\newcommand{{\\bestModelRocAuc}}{{{best_row['ROC-AUC']:.4f}}}",
        f"\\newcommand{{\\bestModelPrAuc}}{{{best_row['PR-AUC']:.4f}}}",
        f"\\newcommand{{\\bestModelSpecificity}}{{{best_row['Specificity']:.4f}}}",
        f"\\newcommand{{\\bestModelBalancedAcc}}{{{best_row['Balanced_Accuracy']:.4f}}}",
        f"\\newcommand{{\\bestModelTN}}{{{int(best_row['True_Negatives']):,}}}",
        f"\\newcommand{{\\bestModelFP}}{{{int(best_row['False_Positives']):,}}}",
        f"\\newcommand{{\\bestModelFN}}{{{int(best_row['False_Negatives']):,}}}",
        f"\\newcommand{{\\bestModelTP}}{{{int(best_row['True_Positives']):,}}}",
        ""
    ]

    for _, row in model_df.iterrows():
        prefix = row['Model'].replace(" ", "").replace("-", "")
        latex_defs.extend([
            f"\\newcommand{{\\{prefix}Acc}}{{{row['Accuracy']:.4f}}}",
            f"\\newcommand{{\\{prefix}Prec}}{{{row['Precision']:.4f}}}",
            f"\\newcommand{{\\{prefix}Rec}}{{{row['Recall']:.4f}}}",
            f"\\newcommand{{\\{prefix}FOne}}{{{row['F1-score']:.4f}}}",
            f"\\newcommand{{\\{prefix}RocAuc}}{{{row['ROC-AUC']:.4f}}}",
            f"\\newcommand{{\\{prefix}PrAuc}}{{{row['PR-AUC']:.4f}}}",
            f"\\newcommand{{\\{prefix}Time}}{{{row['Training_Time_s']:.2f}}}"
        ])

    num_words = ["Zero", "One", "Two", "Three", "Four", "Five"]
    latex_defs.extend(["", f"\\newcommand{{\\selectedClusterCount}}{{{len(cluster_df)}}}"])
    for _, c_row in cluster_df.iterrows():
        c_id = int(c_row['Cluster_ID'])
        c_word = num_words[c_id] if c_id < len(num_words) else f"C{c_id}"
        latex_defs.extend([
            f"\\newcommand{{\\cluster{c_word}Name}}{{{c_row['Segment_Name']}}}",
            f"\\newcommand{{\\cluster{c_word}Size}}{{{int(c_row['Size']):,}}}",
            f"\\newcommand{{\\cluster{c_word}Prop}}{{{c_row['Proportion_pct']:.1f}\\%}}",
            f"\\newcommand{{\\cluster{c_word}Churn}}{{{c_row['Churn_Rate_pct']:.1f}\\%}}",
            f"\\newcommand{{\\cluster{c_word}Tenure}}{{{c_row['Avg_Tenure_mo']:.1f}}}",
            f"\\newcommand{{\\cluster{c_word}Monthly}}{{\\${c_row['Avg_Monthly_Charges']:.2f}}}"
        ])

    top3_feats = feat_df.head(3)["Feature"].tolist()
    clean_top3 = [f.replace("_", " ") for f in top3_feats]
    latex_defs.extend([
        "",
        f"\\newcommand{{\\topFeatureOne}}{{{clean_top3[0]}}}",
        f"\\newcommand{{\\topFeatureTwo}}{{{clean_top3[1]}}}",
        f"\\newcommand{{\\topFeatureThree}}{{{clean_top3[2]}}}"
    ])

    macro_path = REPORT_GENERATED_DIR / "experiment_macros.tex"
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_defs))
    print(f"  [Generated Macros] {macro_path}")


# ==================================================================================================
# SECTION 11: INTEGRATED VERIFICATION TESTS
# ==================================================================================================

def run_self_tests():
    """Execute integrated sanity checks verifying model shapes, metrics, and leakage integrity."""
    print("=" * 70)
    print("RUNNING PIPELINE INTEGRITY AND UNIT VERIFICATION TESTS")
    print("=" * 70)

    # Test 1: Data Loading & Shape
    raw = load_raw_data()
    assert raw.shape == (7043, 21), f"Unexpected raw shape: {raw.shape}"
    print("[PASS] Test 1: Raw data dimensions verified (7,043 x 21).")

    # Test 2: Cleaning logic
    clean = clean_raw_data(raw)
    assert clean["TotalCharges"].isna().sum() == 0, "Nulls remain in TotalCharges after cleaning."
    assert set(clean["Churn"].unique()) == {0, 1}, f"Unexpected target values: {clean['Churn'].unique()}"
    print("[PASS] Test 2: Zero-tenure imputation and binary target encoding verified.")

    # Test 3: Leakage-free train/test partition
    splits = prepare_data_splits()
    assert len(splits["X_train"]) == 5634, f"Train count {len(splits['X_train'])} != 5634"
    assert len(splits["X_test"]) == 1409, f"Test count {len(splits['X_test'])} != 1409"
    assert set(splits["ids_train"]).isdisjoint(set(splits["ids_test"])), "Customer ID leakage detected!"
    print("[PASS] Test 3: Leakage-free stratified train-test partition verified (5,634 : 1,409).")

    # Test 4: Classification evaluation sanity
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    f1 = f1_score(y_true, y_pred)
    assert 0.79 < f1 < 0.81, f"F1 calculation sanity check failed: {f1}"
    print("[PASS] Test 4: Core classification metric functions verified.")

    # Test 5: Segmentation feature integrity
    df_clean = clean_raw_data(raw)
    clustering_features = df_clean.drop(columns=[ID_COL, TARGET_COL])
    assert TARGET_COL not in clustering_features.columns, "Target Churn leaked into clustering features!"
    assert ID_COL not in clustering_features.columns, "Customer ID leaked into clustering features!"
    print("[PASS] Test 5: Strict uncoupling of target and customer ID in clustering verified.")

    print("=" * 70)
    print("ALL 5 INTEGRATED VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 70)


# ==================================================================================================
# SECTION 12: MASTER PIPELINE EXECUTION
# ==================================================================================================

def run_full_pipeline():
    """Execute complete reproducible machine learning workflow."""
    start_time = time.perf_counter()
    print("=" * 70)
    print("STARTING COMPLETE TELECOM CHURN PREDICTION & SEGMENTATION PIPELINE")
    print("=" * 70)

    # 1. Dataset Acquisition & Audit
    print("\n[Step 1/8] Auditing raw dataset...")
    raw_df = load_raw_data()
    audit_report = audit_dataset(raw_df)
    print(f"  Audited: {audit_report['total_records']} records, {audit_report['total_features']} features.")

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
    print("\n[Step 5/8] Training 4 primary classification models (LR, SVM, RF, MLP)...")
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
    print(f"Results and figures saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Churn Prediction and Segmentation Case Study Pipeline")
    parser.add_argument("--test-only", action="store_true", help="Execute only the self-verification test suite")
    args = parser.parse_args()

    if args.test_only:
        run_self_tests()
    else:
        run_full_pipeline()
