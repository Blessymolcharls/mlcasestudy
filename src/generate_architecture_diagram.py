"""
Architecture diagram generator for the Telco Churn Case Study.
Produces a crisp 300 DPI system diagram for Chapter 4 of the LaTeX report.
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import (
    FIGURES_DIR,
    REPORT_FIGURES_DIR,
    FIGURE_DPI
)


def draw_system_architecture():
    """Render an aesthetic publication-quality system architecture flowchart."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=FIGURE_DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Color palette
    c_data = "#e8f4f8"       # Light blue
    c_prep = "#d1e7dd"       # Light green
    c_model = "#fff3cd"      # Light amber
    c_eval = "#cfe2ff"       # Light periwinkle
    c_seg = "#f8d7da"        # Light rose
    c_border = "#212529"

    def draw_box(x, y, w, h, text, bg, subtitle=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
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

    # Title Banner
    ax.text(5.5, 8.1, "Telco Customer Churn Prediction & Segmentation Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1e3a8a")

    # Level 1: Ingestion & Validation
    draw_box(0.6, 6.7, 2.6, 0.9, "IBM Telco Dataset", c_data, "7,043 rows x 21 features")
    draw_box(4.2, 6.7, 2.6, 0.9, "Data Quality Audit", c_data, "Zero-dupes, type checks")
    draw_box(7.8, 6.7, 2.6, 0.9, "Data Cleaning", c_prep, "TotalCharges coercion & tenure")

    draw_arrow(3.2, 7.15, 4.2, 7.15)
    draw_arrow(6.8, 7.15, 7.8, 7.15)

    # Level 2: Stratified Split & Leakage-Free Preprocessing
    draw_box(7.8, 5.1, 2.6, 0.9, "Stratified Split (80:20)", c_prep, "Train: 5,634 | Test: 1,409")
    draw_arrow(9.1, 6.7, 9.1, 6.0)

    draw_box(4.2, 5.1, 2.6, 0.9, "Leakage-Free Preprocessor", c_prep, "StandardScaler + OneHot")
    draw_arrow(7.8, 5.55, 6.8, 5.55, "Fit on Train Only")

    # Level 3: Supervised Classification (Left branch) & Unsupervised Segmentation (Right branch)
    # Split arrow from Preprocessor
    draw_arrow(4.2, 5.55, 3.2, 4.4, "Processed Train/Test")
    draw_arrow(5.5, 5.1, 7.8, 4.4, "All Customers w/o Churn")

    # Supervised Learning Box
    draw_box(0.6, 3.5, 3.2, 1.4, "Four Primary Classifiers", c_model, "1. Logistic Regression (L2)\n2. SVM (RBF Kernel)\n3. Random Forest Ensemble\n4. Multi-Layer Perceptron (MLP)")

    # Unsupervised Learning Box
    draw_box(7.2, 3.5, 3.2, 1.4, "K-Means Clustering & PCA", c_seg, "• Elbow & Silhouette Analysis\n• Selected k = 3 Clusters\n• 2D PCA Dimensionality Red.")

    # Level 4: Evaluation & Interpretation
    draw_box(0.6, 1.8, 3.2, 1.1, "Supervised Evaluation", c_eval, "Accuracy, Precision, Recall, F1,\nROC-AUC, PR-AUC, Confusion Matrix")
    draw_arrow(2.2, 3.5, 2.2, 2.9)

    draw_box(4.2, 1.8, 2.6, 1.1, "Feature Interpretation", c_eval, "Permutation Importance &\nOdds Ratios (Coefficients)")
    draw_arrow(3.8, 2.35, 4.2, 2.35)

    draw_box(7.2, 1.8, 3.2, 1.1, "Post-Hoc Cluster Profiling", c_seg, "• Churn Rate per Segment\n• Tenure & Spend Cohort Profiles")
    draw_arrow(8.8, 3.5, 8.8, 2.9)

    # Level 5: Business Intelligence & Decision Support
    draw_box(2.4, 0.3, 6.2, 0.95, "Business Decision & Retention Policy", c_prep, "Targeted proactive interventions, onboarding support, contract optimization")
    draw_arrow(2.2, 1.8, 4.0, 1.25)
    draw_arrow(5.5, 1.8, 5.5, 1.25)
    draw_arrow(8.8, 1.8, 7.0, 1.25)

    fig_path = FIGURES_DIR / "system_architecture.png"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_path, REPORT_FIGURES_DIR / "system_architecture.png")
    plt.close(fig)
    print("Saved system architecture diagram.")


if __name__ == "__main__":
    draw_system_architecture()
