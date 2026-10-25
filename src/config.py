"""
Configuration and constants for Customer Churn Prediction and Segmentation case study.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORTS_DIR = BASE_DIR / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_GENERATED_DIR = REPORTS_DIR / "generated"

# Data file paths
RAW_DATA_FILE = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FALLBACK_DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Target and ID
TARGET_COL = "Churn"
ID_COL = "customerID"

# Feature definitions
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

# Visualization settings
FIGURE_DPI = 300
PALETTE_NAME = "crest"
