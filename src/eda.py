"""
Exploratory Data Analysis (EDA) module for Telco Customer Churn.
Generates publication-quality charts (300 DPI) for report and analysis.
"""

from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    FIGURES_DIR,
    REPORT_FIGURES_DIR,
    FIGURE_DPI,
    TARGET_COL,
    NUMERIC_COLS
)
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data


def set_plot_style():
    """Configure matplotlib/seaborn aesthetic style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.autolayout": True
    })


def save_and_sync_fig(fig: plt.Figure, filename: str):
    """Save figure to results/figures and sync to reports/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    target_path = FIGURES_DIR / filename
    report_path = REPORT_FIGURES_DIR / filename

    fig.savefig(target_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(target_path, report_path)
    plt.close(fig)
    print(f"Saved: {filename}")


def run_eda():
    """Generate all required EDA visualizations."""
    set_plot_style()
    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)

    palette = {"No": "#2b5c8f", "Yes": "#d95f02"}
    palette_num = {0: "#2b5c8f", 1: "#d95f02"}

    # 1. Churn versus non-churn distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df_raw["Churn"].value_counts()
    props = df_raw["Churn"].value_counts(normalize=True) * 100
    bars = ax.bar(["No Churn", "Churn"], counts.values, color=["#2b5c8f", "#d95f02"], width=0.55, edgecolor="black", linewidth=0.8)
    for bar, prop in zip(bars, props):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 60,
                f"{int(height):,} ({prop:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)
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
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2,
                f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
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
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2,
                f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate Across Customer Tenure Cohorts")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Tenure Cohort")
    ax.set_ylim(0, max(tenure_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_03_churn_by_tenure_group.png")

    # 4. MonthlyCharges distribution by churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 0], x="MonthlyCharges", label="Retained (No)", color="#2b5c8f", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 1], x="MonthlyCharges", label="Churned (Yes)", color="#d95f02", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Density Distribution of Monthly Charges by Churn Status")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    save_and_sync_fig(fig, "eda_04_monthly_charges_distribution.png")

    # 5. TotalCharges distribution by churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 0], x="TotalCharges", label="Retained (No)", color="#2b5c8f", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=df_clean[df_clean[TARGET_COL] == 1], x="TotalCharges", label="Churned (Yes)", color="#d95f02", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Density Distribution of Cumulative Total Charges by Churn Status")
    ax.set_xlabel("Total Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    save_and_sync_fig(fig, "eda_05_total_charges_distribution.png")

    # 6. Churn rate by PaymentMethod
    fig, ax = plt.subplots(figsize=(9, 5))
    pm_churn = df_clean.groupby("PaymentMethod")[TARGET_COL].agg(["count", "mean"]).reset_index()
    pm_churn["mean"] *= 100
    pm_churn = pm_churn.sort_values(by="mean", ascending=False)
    bars = ax.bar(pm_churn["PaymentMethod"], pm_churn["mean"], color="#fdae61", width=0.55, edgecolor="black")
    for bar, count in zip(bars, pm_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2,
                f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Payment Method")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    ax.set_xticks(range(len(pm_churn)))
    ax.set_xticklabels(pm_churn["PaymentMethod"], rotation=15, ha="right")
    ax.set_ylim(0, max(pm_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_06_churn_by_payment_method.png")

    # 7. Churn rate by InternetService
    fig, ax = plt.subplots(figsize=(7, 5))
    is_churn = df_clean.groupby("InternetService")[TARGET_COL].agg(["count", "mean"]).reset_index()
    is_churn["mean"] *= 100
    bars = ax.bar(is_churn["InternetService"], is_churn["mean"], color="#abd9e9", width=0.5, edgecolor="black")
    for bar, count in zip(bars, is_churn["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2,
                f"{h:.1f}%\n(N={count:,})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn Rate by Internet Service Type")
    ax.set_ylabel("Observed Churn Rate (%)")
    ax.set_xlabel("Internet Service Technology")
    ax.set_ylim(0, max(is_churn["mean"]) * 1.22)
    save_and_sync_fig(fig, "eda_07_churn_by_internet_service.png")

    # 8. Numeric correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df_clean[NUMERIC_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"}, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Continuous Predictors & Churn")
    save_and_sync_fig(fig, "eda_08_correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()
