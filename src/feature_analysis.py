"""
Feature importance and interpretation module.
Computes permutation importance on held-out test set and examines model coefficients.
"""

from typing import Dict, Any, List
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    FIGURE_DPI,
    RANDOM_SEED
)


def run_feature_analysis(
    best_model_name: str,
    best_model: Any,
    trained_artifacts: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compute permutation importance for the best model on test data
    and inspect linear model coefficients for interpretability.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing permutation importance for best model: {best_model_name}...")
    perm_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_SEED,
        scoring="f1",
        n_jobs=1
    )

    feature_names = X_test.columns.tolist()
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": perm_result.importances_mean,
        "Permutation_Importance_Std": perm_result.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)

    # Save to CSV
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Plot Top 15 Features by Permutation Importance
    top_15 = importance_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top_15["Feature"][::-1],
        top_15["Permutation_Importance_Mean"][::-1],
        xerr=top_15["Permutation_Importance_Std"][::-1],
        color="#2b5c8f",
        edgecolor="black",
        capsize=3
    )
    ax.set_title(f"Top 15 Permutation Feature Importances ({best_model_name})")
    ax.set_xlabel("Mean Drop in Test F1-Score")
    fig_path = FIGURES_DIR / "feat_permutation_importance.png"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_path, REPORT_FIGURES_DIR / "feat_permutation_importance.png")
    plt.close(fig)

    # Logistic Regression coefficients analysis (if available)
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

            # Plot top 10 positive and top 10 negative coefficients
            top_pos = lr_df.head(8)
            top_neg = lr_df.tail(8)
            combined_lr = pd.concat([top_pos, top_neg]).sort_values(by="Coefficient", ascending=True)

            fig_lr, ax_lr = plt.subplots(figsize=(9, 6))
            colors = ["#d95f02" if c > 0 else "#2b5c8f" for c in combined_lr["Coefficient"]]
            ax_lr.barh(combined_lr["Feature"], combined_lr["Coefficient"], color=colors, edgecolor="black")
            ax_lr.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax_lr.set_title("Logistic Regression Coefficients (Risk vs. Protective Factors)")
            ax_lr.set_xlabel("Log-Odds Coefficient (Positive = Higher Churn Risk)")
            fig_lr_path = FIGURES_DIR / "feat_logistic_coefficients.png"
            fig_lr.savefig(fig_lr_path, dpi=FIGURE_DPI, bbox_inches="tight")
            shutil.copy(fig_lr_path, REPORT_FIGURES_DIR / "feat_logistic_coefficients.png")
            plt.close(fig_lr)

    # Generate LaTeX table snippet for Top Features
    generate_latex_top_features_table(top_15)

    return importance_df


def generate_latex_top_features_table(top_df: pd.DataFrame):
    """Generate LaTeX table for top features."""
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Top Predictive Features Ranked by Permutation Importance (Test Set)}",
        r"\label{tab:feature_importance}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Rank} & \textbf{Feature Name} & \textbf{Mean F1 Drop} & \textbf{Std Dev} \\",
        r"\midrule"
    ]

    for idx, row in top_df.iterrows():
        feat_clean = row['Feature'].replace('_', r'\_')
        latex_lines.append(
            rf"{idx + 1} & {feat_clean} & {row['Permutation_Importance_Mean']:.4f} & "
            rf"$\pm$ {row['Permutation_Importance_Std']:.4f} \\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    table_path = REPORT_TABLES_DIR / "table_feature_importance.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
