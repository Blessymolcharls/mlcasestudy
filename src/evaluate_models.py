"""
Evaluation module for the four classification models on held-out test data.
Computes core metrics, saves confusion matrices, plots ROC/PR curves, and exports prediction records.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    balanced_accuracy_score
)

from src.config import (
    RESULTS_DIR,
    CONFUSION_DIR,
    FIGURES_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    FIGURE_DPI
)


def evaluate_all_models(
    trained_artifacts: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ids_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate trained models on the test set.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    reports_dict = {}
    predictions_df = pd.DataFrame({"customerID": ids_test.values, "Actual_Churn": y_test})

    # Prepare ROC and PR plot data
    roc_data = {}
    pr_data = {}

    for name, item in trained_artifacts.items():
        model = item["model"]
        train_time = item["train_time_sec"]

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        # Store predictions
        safe_name = name.lower().replace(" ", "_")
        predictions_df[f"{safe_name}_pred"] = y_pred
        predictions_df[f"{safe_name}_prob"] = np.round(y_prob, 4)

        # Core Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)

        # Confusion matrix: TN, FP, FN, TP
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Save confusion matrix CSV
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_No_Churn", "Actual_Churn"],
            columns=["Pred_No_Churn", "Pred_Churn"]
        )
        cm_df.to_csv(CONFUSION_DIR / f"cm_{safe_name}.csv")

        # Plot individual Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            cbar=False,
            ax=ax_cm
        )
        ax_cm.set_title(f"Confusion Matrix: {name}")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        fig_cm_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
        fig_cm.savefig(fig_cm_path, dpi=FIGURE_DPI, bbox_inches="tight")
        shutil.copy(fig_cm_path, REPORT_FIGURES_DIR / f"confusion_matrix_{safe_name}.png")
        plt.close(fig_cm)

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)

        # PR Curve data
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (recall_curve, precision_curve, pr_auc)

        # Classification report dict
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports_dict[name] = rep

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

    # Convert to DataFrame
    comparison_df = pd.DataFrame(metrics_list)
    comparison_df = comparison_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Save classification reports JSON
    with open(RESULTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports_dict, f, indent=4)

    # Save predictions CSV
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
    fig_roc_path = FIGURES_DIR / "eval_roc_curves.png"
    fig_roc.savefig(fig_roc_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_roc_path, REPORT_FIGURES_DIR / "eval_roc_curves.png")
    plt.close(fig_roc)

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
    fig_pr_path = FIGURES_DIR / "eval_pr_curves.png"
    fig_pr.savefig(fig_pr_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_pr_path, REPORT_FIGURES_DIR / "eval_pr_curves.png")
    plt.close(fig_pr)

    # Plot Model Comparison Bar Chart
    plot_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    melted_df = comparison_df.melt(id_vars=["Model"], value_vars=plot_metrics, var_name="Metric", value_name="Score")

    fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax_bar, edgecolor="black")
    ax_bar.set_title("Comparative Performance of the Four Primary Classifiers")
    ax_bar.set_ylabel("Score (0.0 to 1.0)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(loc="lower right")
    fig_bar_path = FIGURES_DIR / "eval_model_comparison_bar.png"
    fig_bar.savefig(fig_bar_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_bar_path, REPORT_FIGURES_DIR / "eval_model_comparison_bar.png")
    plt.close(fig_bar)

    # Generate LaTeX table for Table 6.1
    generate_latex_table_6_1(comparison_df)

    return comparison_df, predictions_df


def generate_latex_table_6_1(comparison_df: pd.DataFrame):
    """Generate LaTeX snippet for Table 6.1 required in Chapter 6."""
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Model Performance on the Held-Out Test Set ($N=1,409$)}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\",
        r"\midrule"
    ]

    for _, row in comparison_df.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & "
            f"{row['Recall']:.4f} & {row['F1-score']:.4f} & {row['ROC-AUC']:.4f} & {row['PR-AUC']:.4f} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    table_path = REPORT_TABLES_DIR / "table_6_1_model_comparison.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))


# Typing alias helper
from typing import Tuple as Tuple_Results
