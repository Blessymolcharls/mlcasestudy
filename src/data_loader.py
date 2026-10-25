"""
Data loading and audit module for Telco Customer Churn dataset.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    FALLBACK_DATA_FILE,
    RESULTS_DIR,
    TARGET_COL,
    ID_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS
)


def get_data_filepath() -> Path:
    """Resolve the dataset filepath with fallback."""
    if RAW_DATA_FILE.exists():
        return RAW_DATA_FILE
    if FALLBACK_DATA_FILE.exists():
        return FALLBACK_DATA_FILE
    raise FileNotFoundError(
        f"Telco Churn dataset not found at {RAW_DATA_FILE} or {FALLBACK_DATA_FILE}."
    )


def load_raw_data() -> pd.DataFrame:
    """Load the raw Telco customer churn CSV."""
    filepath = get_data_filepath()
    df = pd.read_csv(filepath)
    return df


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive audit of the dataset and save the report.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0

    # Check blank strings in TotalCharges specifically
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

    # Numerical summaries (temporarily convert TotalCharges for summary)
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

    audit_file = RESULTS_DIR / "dataset_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=4)

    # Also generate a clean CSV Data Dictionary
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

    dict_df = pd.DataFrame(dictionary_records)
    dict_df.to_csv(RESULTS_DIR / "data_dictionary.csv", index=False)

    return audit_report


if __name__ == "__main__":
    raw_df = load_raw_data()
    audit = audit_dataset(raw_df)
    print(f"Audited {audit['total_records']} rows, {audit['total_features']} columns.")
    print("Audit JSON saved to results/dataset_audit.json")
