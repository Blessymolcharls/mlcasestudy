"""
Unsupervised Customer Segmentation module using K-Means clustering and PCA.
Excludes Churn and customerID during cluster formation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

from src.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    FIGURE_DPI,
    RANDOM_SEED,
    ID_COL,
    TARGET_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS
)
from src.data_loader import load_raw_data
from src.preprocessing import clean_raw_data, build_preprocessor


def run_customer_segmentation() -> Dict[str, Any]:
    """
    Execute end-to-end unsupervised clustering pipeline:
    1. Prepare features without customerID or Churn.
    2. Evaluate k=2..8 with Inertia and Silhouette scores.
    3. Fit optimal K-Means.
    4. Project onto 2D PCA space and plot.
    5. Post-hoc profile evaluation joining Churn.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data()
    df_clean = clean_raw_data(df_raw)

    customer_ids = df_clean[ID_COL].copy()
    actual_churn = df_clean[TARGET_COL].copy()

    # Crucial rule: Exclude Churn and customerID from clustering features
    features_df = df_clean.drop(columns=[ID_COL, TARGET_COL])

    preprocessor = build_preprocessor()
    X_cluster = preprocessor.fit_transform(features_df)

    # 1. Evaluate k from 2 to 8
    k_range = list(range(2, 9))
    inertias = []
    silhouettes = []

    print("Evaluating K-Means cluster candidates (k=2 to 8)...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_cluster)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_cluster, labels, sample_size=3000, random_state=RANDOM_SEED)
        silhouettes.append(sil)
        print(f"  k={k} | Inertia: {km.inertia_:.1f} | Silhouette: {sil:.4f}")

    # Plot Elbow Curve
    fig_elbow, ax_elbow = plt.subplots(figsize=(6.5, 4.5))
    ax_elbow.plot(k_range, inertias, "bo-", linewidth=2, markersize=7)
    ax_elbow.set_title("K-Means Elbow Method for Optimal Cluster Count")
    ax_elbow.set_xlabel("Number of Clusters (k)")
    ax_elbow.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax_elbow.set_xticks(k_range)
    fig_elbow_path = FIGURES_DIR / "cluster_01_elbow_curve.png"
    fig_elbow.savefig(fig_elbow_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_elbow_path, REPORT_FIGURES_DIR / "cluster_01_elbow_curve.png")
    plt.close(fig_elbow)

    # Plot Silhouette Scores
    fig_sil, ax_sil = plt.subplots(figsize=(6.5, 4.5))
    ax_sil.plot(k_range, silhouettes, "rs-", linewidth=2, markersize=7)
    ax_sil.set_title("Silhouette Score vs. Number of Clusters")
    ax_sil.set_xlabel("Number of Clusters (k)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.set_xticks(k_range)
    fig_sil_path = FIGURES_DIR / "cluster_02_silhouette_scores.png"
    fig_sil.savefig(fig_sil_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_sil_path, REPORT_FIGURES_DIR / "cluster_02_silhouette_scores.png")
    plt.close(fig_sil)

    # Select optimal k based on highest silhouette or sharpest elbow
    # For customer segmentation, k=3 or 4 typically offers high business interpretability
    best_k_idx = int(np.argmax(silhouettes))
    chosen_k = k_range[best_k_idx]
    # Ensure reasonable customer group count (if k=2 has highest silhouette but limited granularity, use k=3 or k=4)
    if chosen_k == 2 and silhouettes[1] > 0.15:
        chosen_k = 3

    print(f"Selected cluster count: k = {chosen_k}")

    # Fit final K-Means
    final_km = KMeans(n_clusters=chosen_k, init="k-means++", n_init=20, random_state=RANDOM_SEED)
    cluster_labels = final_km.fit_predict(X_cluster)

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_cluster)
    var_exp = pca.explained_variance_ratio_
    print(f"PCA 2D explained variance: PC1={var_exp[0]*100:.2f}%, PC2={var_exp[1]*100:.2f}% (Total: {sum(var_exp)*100:.2f}%)")

    # Plot PCA Clusters
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    scatter = ax_pca.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="Set1",
        alpha=0.4,
        edgecolors="none",
        s=25
    )
    # Plot cluster centroids in PCA space
    pca_centroids = pca.transform(final_km.cluster_centers_)
    ax_pca.scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        c="black",
        s=120,
        marker="X",
        label="Cluster Centroids"
    )
    ax_pca.set_title(f"Customer Segments Projected on 2D PCA Space (k={chosen_k})")
    ax_pca.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)")
    ax_pca.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)")
    ax_pca.legend()
    fig_pca_path = FIGURES_DIR / "cluster_03_pca_clusters.png"
    fig_pca.savefig(fig_pca_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_pca_path, REPORT_FIGURES_DIR / "cluster_03_pca_clusters.png")
    plt.close(fig_pca)

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

        # Most common contract and internet service
        top_contract = c_sub["Contract"].mode()[0]
        top_internet = c_sub["InternetService"].mode()[0]

        # Give factual descriptive name based on observed metrics
        if churn_rate > 35:
            segment_name = "High-Risk Attrition Cohort"
        elif mean_tenure > 40:
            segment_name = "Long-Term Loyal Enterprise"
        else:
            segment_name = "Value-Oriented Core Customers"

        cluster_records.append({
            "Cluster_ID": c,
            "Segment_Name": segment_name,
            "Size": size,
            "Proportion_pct": round(prop, 2),
            "Churn_Rate_pct": round(churn_rate, 2),
            "Avg_Tenure_mo": round(mean_tenure, 1),
            "Avg_Monthly_Charges": round(mean_monthly, 2),
            "Avg_Total_Charges": round(mean_total, 2),
            "Dominant_Contract": top_contract,
            "Dominant_Internet": top_internet
        })

    profile_df = pd.DataFrame(cluster_records)
    profile_df.to_csv(RESULTS_DIR / "cluster_profiles.csv", index=False)

    # Save cluster assignments
    assignments_df = pd.DataFrame({
        "customerID": customer_ids,
        "Cluster": cluster_labels,
        "Actual_Churn": actual_churn
    })
    assignments_df.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)

    # Plot Cluster Profile Comparison
    fig_prof, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(
        [f"Cluster {r['Cluster_ID']}" for _, r in profile_df.iterrows()],
        profile_df["Churn_Rate_pct"],
        color=["#d95f02" if r > 30 else "#2b5c8f" for r in profile_df["Churn_Rate_pct"]],
        edgecolor="black"
    )
    for idx, r in profile_df.iterrows():
        ax1.text(idx, r["Churn_Rate_pct"] + 1, f"{r['Churn_Rate_pct']:.1f}%", ha="center", fontweight="bold")
    ax1.set_title("Observed Post-Hoc Churn Rate by Cluster")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_ylim(0, max(profile_df["Churn_Rate_pct"]) * 1.2)

    ax2.bar(
        [f"Cluster {r['Cluster_ID']}" for _, r in profile_df.iterrows()],
        profile_df["Avg_Tenure_mo"],
        color="#74add1",
        edgecolor="black"
    )
    for idx, r in profile_df.iterrows():
        ax2.text(idx, r["Avg_Tenure_mo"] + 1, f"{r['Avg_Tenure_mo']:.1f} mo", ha="center", fontweight="bold")
    ax2.set_title("Average Customer Tenure by Cluster")
    ax2.set_ylabel("Average Tenure (Months)")
    ax2.set_ylim(0, max(profile_df["Avg_Tenure_mo"]) * 1.2)

    fig_prof_path = FIGURES_DIR / "cluster_04_profile_comparison.png"
    fig_prof.savefig(fig_prof_path, dpi=FIGURE_DPI, bbox_inches="tight")
    shutil.copy(fig_prof_path, REPORT_FIGURES_DIR / "cluster_04_profile_comparison.png")
    plt.close(fig_prof)

    # Generate LaTeX Table for Cluster Profiles
    generate_latex_cluster_table(profile_df)

    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "chosen_k": chosen_k,
        "profile_df": profile_df,
        "pca_variance": var_exp.tolist()
    }


def generate_latex_cluster_table(profile_df: pd.DataFrame):
    """Generate LaTeX table for cluster profiles in Chapter 6."""
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{K-Means Customer Segmentation Profiles ($k=3$)}",
        r"\label{tab:cluster_profiles}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Cluster ID} & \textbf{Segment Description} & \textbf{Size ($N$)} & \textbf{Share (\%)} & \textbf{Churn Rate} & \textbf{Mean Tenure} & \textbf{Mean Monthly (\$)} \\",
        r"\midrule"
    ]

    for _, row in profile_df.iterrows():
        latex_lines.append(
            rf"Cluster {row['Cluster_ID']} & {row['Segment_Name']} & {row['Size']:,} & "
            rf"{row['Proportion_pct']:.1f}\% & {row['Churn_Rate_pct']:.1f}\% & "
            rf"{row['Avg_Tenure_mo']:.1f} mo & \${row['Avg_Monthly_Charges']:.2f} \\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    table_path = REPORT_TABLES_DIR / "table_cluster_profiles.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
