"""
Model training and pipeline execution for four primary classifiers:
1. Logistic Regression
2. Support Vector Machine (RBF)
3. Random Forest Classifier
4. Multi-Layer Perceptron (MLP)
"""

import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    CV_FOLDS
)


def get_model_definitions() -> Dict[str, Any]:
    """Define the 4 primary classifiers with fixed seeds and balanced weighting."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_SEED
        )
    }


def train_and_cross_validate(
    X_train: pd.DataFrame,
    y_train: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Train each classifier on the training split, measure train time,
    and compute 5-fold cross-validated F1 and ROC-AUC scores on training set.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = get_model_definitions()
    trained_artifacts = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.perf_counter()

        # Training-set cross-validation
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)

        # Fit final model on full training set
        model.fit(X_train, y_train)
        elapsed_time = time.perf_counter() - start_time

        # Check convergence for MLP
        if isinstance(model, MLPClassifier):
            print(f"  MLP completed in {model.n_iter_} iterations (Loss: {model.loss_:.4f})")

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{safe_name}_model.joblib"
        joblib.dump(model, model_path)

        trained_artifacts[name] = {
            "model": model,
            "train_time_sec": round(elapsed_time, 4),
            "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
            "cv_f1_std": round(float(np.std(cv_f1)), 4),
            "cv_roc_mean": round(float(np.mean(cv_roc)), 4),
            "cv_roc_std": round(float(np.std(cv_roc)), 4),
            "artifact_path": str(model_path)
        }
        print(f"  Done in {elapsed_time:.2f}s | CV F1: {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")

    return trained_artifacts
