# Customer Churn Prediction and Segmentation for a Telecom Company
### Academic Case Study Report and Machine Learning Deliverable

---

## 1. Project and Student Details

- **Project Title:** Customer Churn Prediction and Segmentation for a Telecom Company
- **Student Name:** Blessy Mol Charls
- **Register Number:** MUT24CA029
- **Degree:** Bachelor of Technology (B.Tech)
- **Branch:** Computer Science and Engineering
- **Department:** Department of Computer Science and Engineering
- **Institution:** Muthoot Institute of Technology and Science (MITS), Varikoli, Ernakulam -- 682308
- **Affiliated University:** APJ Abdul Kalam Technological University (KTU)
- **Course Code:** CST 306 *(Editable institutional placeholder)*
- **Course Name:** Machine Learning *(Editable institutional placeholder)*
- **Faculty-in-Charge:** [FACULTY NAME] *(Editable institutional placeholder)*
- **Submission Date:** October 2026

---

## 2. Executive Summary & Core Results

This repository contains the complete, reproducible machine learning case study implemented for telecommunications subscriber churn prediction and unsupervised customer segmentation. Using the benchmark **IBM Telco Customer Churn dataset** ($N=7,043$ customers, $21$ features), the study enforces a strict **leakage-free stratified 80:20 partition** (Training split: $5,634$ records; Held-out test split: $1,409$ records).

### Four Primary Classifiers: Empirical Performance Comparison

All four classifiers were trained with class-weight balancing and evaluated on the uncorrupted held-out test split ($N=1,409$; $1,035$ retained, $374$ churners):

| Rank | Model Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | Train Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Random Forest Classifier** | **0.7559** | **0.5276** | **0.7674** | **0.6253** | **0.8427** | **0.6554** | **0.7517** | 26.13 s |
| 2 | **Support Vector Machine (RBF)** | 0.7452 | 0.5133 | 0.7727 | 0.6169 | 0.8213 | 0.5861 | 0.7353 | 235.25 s |
| 3 | **Logistic Regression (L2)** | 0.7381 | 0.5043 | **0.7834** | 0.6136 | 0.8415 | 0.6325 | 0.7217 | **2.44 s** |
| 4 | **Multi-Layer Perceptron (MLP)** | **0.8034** | **0.6678** | 0.5160 | 0.5822 | 0.8342 | 0.6225 | **0.9072** | 64.39 s |

- **Best Overall Model:** **Random Forest Classifier** achieved the highest overall balance with an **F1-score of 0.6253**, **ROC-AUC of 0.8427**, and **PR-AUC of 0.6554**.
- **Highest Sensitivity / Recall:** **Logistic Regression** captured the highest percentage of actual churners (**Recall = 78.34%**, missing only $81$ churners) at lightning-fast training speed (**2.44 s**).
- **Highest Accuracy & Precision:** **Multi-Layer Perceptron** achieved the highest accuracy (**80.34%**) and precision (**66.78%**), but missed $181$ churners due to its conservative threshold.

### Unsupervised Customer Segmentation (K-Means & PCA)

Clustering was performed **strictly without the target `Churn` or `customerID`**. Evaluated across $k \in [2, 8]$, joint Elbow and Silhouette analysis ($s=0.249$) identified $k=3$ natural customer clusters:

| Cluster | Segment Description | Size ($N$) | Share (%) | Churn Rate | Mean Tenure | Mean Monthly | Dominant Profile |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | **Value-Oriented Core Customers** | 1,526 | 21.7% | **7.4%** | 30.5 mo | $21.08 | Voice-only phone, two-year contracts, highly loyal |
| **1** | **Long-Term Loyal Enterprise** | 2,330 | 33.1% | **15.2%** | 56.9 mo | $89.16 | Fiber optic + full bundled security, high LTV |
| **2** | **High-Risk Attrition Cohort** | 3,187 | 45.2% | **44.0%** | 15.3 mo | $67.84 | Month-to-month fiber optic, no tech support |

---

## 3. Repository Architecture

```
mlcasestudy/
├── README.md                              # Master project documentation
├── requirements.txt                       # Production Python dependencies
├── pytest.ini                             # Automated test runner configuration
├── .gitignore                             # Ignored cache and build artifacts
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw benchmark dataset
├── Customer_Churn_Case_Study_Report.pdf   # 45-page compiled formal academic PDF
├── data/
│   ├── raw/                               # Raw dataset directory
│   └── processed/                         # Processed parquet train/test splits & labels
├── src/
│   ├── __init__.py
│   ├── config.py                          # Global parameters, paths, and seed (42)
│   ├── data_loader.py                     # Data loading and comprehensive audit
│   ├── preprocessing.py                   # Leakage-free ColumnTransformer & stratified split
│   ├── eda.py                             # 8 publication-grade EDA charts (300 DPI)
│   ├── generate_architecture_diagram.py   # Flowchart system architecture generator
│   ├── train_models.py                    # 4 classifiers training & 5-fold CV
│   ├── evaluate_models.py                 # Multi-metric test evaluation & ROC/PR curves
│   ├── feature_analysis.py                # Permutation importance & log-odds coefficients
│   ├── segmentation.py                    # Unsupervised K-Means (k=3) & 2D PCA projection
│   ├── report_data.py                     # LaTeX macro aggregator for report numbers
│   └── run_pipeline.py                    # Master one-click execution orchestrator
├── models/                                # Serialized model & preprocessor pipelines (.joblib)
├── results/
│   ├── dataset_audit.json                 # Machine-readable dataset audit log
│   ├── data_dictionary.csv                # Complete attribute dictionary
│   ├── model_comparison.csv               # Empirical metric summary table
│   ├── predictions.csv                    # Customer-level prediction records & probabilities
│   ├── classification_reports.json        # Class-level precision/recall/f1 breakdown
│   ├── feature_importance.csv             # Permutation importance test rankings
│   ├── logistic_regression_coefficients.csv # Log-odds coefficients & odds ratios
│   ├── cluster_profiles.csv               # Empirical cluster breakdown
│   ├── cluster_assignments.csv            # Customer cluster assignments
│   ├── experiment_summary.json            # Master run parameters and machine metrics
│   ├── confusion_matrices/                # CSV confusion matrices for each model
│   └── figures/                           # High-resolution (300 DPI) figures
├── reports/
│   ├── main.tex                           # Master LaTeX report source
│   ├── main.pdf                           # Compiled formal academic report
│   ├── references.bib                     # Verified peer-reviewed bibliography
│   ├── chapters/                          # Modular chapter sources (Ch 0 to 8 + App)
│   ├── figures/                           # Synchronized 300 DPI figures for LaTeX
│   ├── tables/                            # Auto-generated LaTeX table snippets
│   └── generated/                         # Dynamic LaTeX macros from real results
└── tests/
    ├── test_data_loader.py                # Schema & row validation tests
    ├── test_preprocessing.py              # Leakage prevention & stratification tests
    ├── test_metrics.py                    # Metric bounds & confusion matrix assertions
    └── test_segmentation.py               # Target exclusion & cluster integrity tests
```

---

## 4. Institutional Course Outcomes (CO-PO/PSO Mapping)

The project activities strictly align with the institutional Course Outcomes (CO1--CO4) and the official articulation matrix provided in `ML_CO-PO PSO MAPPING.pdf`:

- **CO1:** Apply Machine Learning concepts and basic parameter estimation methods to solve data-driven problems.
- **CO2:** Demonstrate regression and classification algorithms and performance measures to evaluate machine learning models.
- **CO3:** Illustrate the concepts of Neural Networks and Support Vector Machines in Machine learning problems.
- **CO4:** Demonstrate unsupervised learning concepts, ensemble methods and dimensionality reduction techniques to comprehend complex data patterns.

### Exact Institutional Articulation Matrix

| CO | PO1 | PO2 | PO3 | PO4 | PO5 | PO6 | PO7 | PO8 | PO9 | PO10 | PO11 | PSO1 | PSO2 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CO1** | 3 | 2 | 2 | – | – | – | – | – | – | – | 2 | 3 | 2 |
| **CO2** | 3 | 3 | 3 | 2 | 3 | – | – | – | – | – | 2 | 3 | 2 |
| **CO3** | 2 | 2 | 3 | – | 3 | – | – | – | – | – | 2 | 3 | 2 |
| **CO4** | 2 | 3 | 2 | 3 | 2 | – | – | – | – | – | 2 | 3 | 2 |

---

## 5. Sustainable Development Goals (SDGs) Addressed

Based on `SDG (1).pdf`, the case study maps to two United Nations Sustainable Development Goals:

1. **SDG 9: Industry, Innovation, Technology & Infrastructure**
   - *Target 9.5 & 9.c:* Upgrades industrial technological capabilities in telecommunications through data-driven predictive intelligence and protects customer access to affordable broadband connectivity.
2. **SDG 8: Decent Work and Economic Growth**
   - *Target 8.2:* Elevates operational productivity and optimizes customer retention workforce deployment in telecommunication enterprises.

---

## 6. Setup, Execution, and Verification Guide

### Step 1: Environment Setup (Windows PowerShell)

```powershell
# Clone or navigate to the repository directory
cd c:\Users\bless\GIT\mlcasestudy

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Complete Machine Learning Pipeline

```powershell
python -m src.run_pipeline
```

*This single command automatically executes data auditing, leakage-free preprocessing, exploratory chart generation, architecture diagram synthesis, model training (5-fold CV), test evaluation across all 4 classifiers, permutation feature analysis, K-Means clustering & PCA, and generates LaTeX macros.*

### Step 3: Run Automated Test Suite

```powershell
pytest -v
```

*Expected output: 9 passed in ~2.8 seconds (100% test pass rate).*

### Step 4: Compile Formal Academic LaTeX Report

```powershell
# Ensure MiKTeX is on PATH and run compilation in reports directory
cd reports
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
```

*The generated PDF is located at `reports/main.pdf` and mirrored at `Customer_Churn_Case_Study_Report.pdf` (45 pages).*

---

## 7. Final Submission Checklist

- [x] **Complete Python Implementation:** All modules modularized under `src/` with clear responsibilities.
- [x] **Reproducible Dependency File:** `requirements.txt` with compatible version bounds.
- [x] **Dataset Audit & Integrity:** 7,043 rows, 21 columns, zero duplicate records, safe numeric conversion of `TotalCharges`.
- [x] **Four Primary Classifiers:** Logistic Regression, Support Vector Machine (RBF), Random Forest, and Multi-Layer Perceptron.
- [x] **Leakage Prevention:** Strict stratified 80:20 split; `ColumnTransformer` fitted strictly on training data.
- [x] **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Specificity, Balanced Accuracy, Confusion Matrices.
- [x] **Interpretability Analysis:** Model-agnostic Permutation Importance on held-out test data and Logistic Regression odds ratios.
- [x] **Unsupervised Segmentation:** K-Means clustering and PCA 2D projection conducted strictly without `Churn`.
- [x] **Report Artifacts & Data Macros:** Auto-generated LaTeX macros ensuring zero transcription error between Python results and LaTeX text.
- [x] **Institutional Case Study Template:** Exact compliance with `Case_study_format_template.pdf` (Title, Certificate, Abstract + CO-PO matrix + SDG table, Chapters 1--8, References, Appendix).
- [x] **Successfully Compiled PDF:** 45-page publication-quality PDF compiled via MiKTeX with zero fatal errors.
- [x] **Automated Test Suite:** 9 unit tests passing cleanly with `pytest`.
