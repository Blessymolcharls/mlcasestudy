"""
Preprocessing and leakage prevention pipeline for Telco Churn.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TARGET_COL,
    ID_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS
)
from src.data_loader import load_raw_data


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame:
    - Strip whitespace from string columns
    - Convert TotalCharges safely to numeric (coercing blanks to NaN)
    - If tenure is 0 and TotalCharges is NaN, fill TotalCharges with 0.0
    - Map target Churn to binary integer (Yes: 1, No: 0)
    """
    df_clean = df.copy()

    # TotalCharges conversion
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    # Physical domain truth: customers with tenure == 0 have not completed 1 billing cycle
    zero_tenure_mask = (df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna())
    df_clean.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Convert SeniorCitizen to string or category for uniform categorical handling
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    # Encode Target
    if TARGET_COL in df_clean.columns:
        df_clean[TARGET_COL] = df_clean[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_data_splits() -> Dict[str, Any]:
    """
    Execute stratified train/test split and fit preprocessing exclusively on training data.
    Saves preprocessor and processed data splits.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)

    customer_ids = clean_df[ID_COL].copy()
    y = clean_df[TARGET_COL].values
    X = clean_df.drop(columns=[ID_COL, TARGET_COL])

    # Stratified Train/Test split
    X_train_raw, X_test_raw, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Fit preprocessor strictly on X_train to prevent leakage
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)

    # Transform both splits
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + cat_feature_names

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # Save processed splits and preprocessor
    X_train_df.to_parquet(DATA_PROCESSED_DIR / "X_train.parquet")
    X_test_df.to_parquet(DATA_PROCESSED_DIR / "X_test.parquet")
    np.save(DATA_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(DATA_PROCESSED_DIR / "y_test.npy", y_test)
    ids_train.to_csv(DATA_PROCESSED_DIR / "ids_train.csv", index=False)
    ids_test.to_csv(DATA_PROCESSED_DIR / "ids_test.csv", index=False)

    # Save feature names mapping
    pd.DataFrame({"feature": feature_names}).to_csv(
        DATA_PROCESSED_DIR / "feature_names.csv", index=False
    )

    # Persist the preprocessor
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


if __name__ == "__main__":
    splits = prepare_data_splits()
    print("Preprocessed splits successfully created:")
    print(f"Train shape: {splits['X_train'].shape}, Test shape: {splits['X_test'].shape}")
    print(f"Class distribution train: {np.bincount(splits['y_train'])}")
    print(f"Class distribution test:  {np.bincount(splits['y_test'])}")
