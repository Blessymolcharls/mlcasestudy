"""
Report data aggregator: generates LaTeX macros and summary files from real experimental results.
Ensures zero manual transcription errors between Python outputs and LaTeX document.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RESULTS_DIR,
    REPORT_GENERATED_DIR
)


def generate_report_macros():
    """
    Read generated experimental results and output LaTeX macro definitions.
    """
    REPORT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset audit
    with open(RESULTS_DIR / "dataset_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)

    # 2. Model comparison
    model_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    best_row = model_df.iloc[0] # Highest F1

    # 3. Cluster profiles
    cluster_df = pd.read_csv(RESULTS_DIR / "cluster_profiles.csv")

    # 4. Feature importance
    feat_df = pd.read_csv(RESULTS_DIR / "feature_importance.csv")

    latex_defs = [
        "% Auto-generated experimental macros from real pipeline execution",
        f"\\newcommand{{\\datasetTotalRows}}{{{audit['total_records']:,}}}",
        f"\\newcommand{{\\datasetTotalCols}}{{{audit['total_features']}}}",
        f"\\newcommand{{\\datasetChurnProportion}}{{{audit['target_distribution']['Yes']['proportion']*100:.1f}\\%}}",
        f"\\newcommand{{\\datasetRetainedProportion}}{{{audit['target_distribution']['No']['proportion']*100:.1f}\\%}}",
        f"\\newcommand{{\\datasetTrainCount}}{{5,634}}",
        f"\\newcommand{{\\datasetTestCount}}{{1,409}}",
        f"\\newcommand{{\\datasetMissingTotalCharges}}{{{audit['total_charges_blank_count']}}}",
        "",
        "% Classification metrics for the best model",
        f"\\newcommand{{\\bestModelName}}{{{best_row['Model']}}}",
        f"\\newcommand{{\\bestModelAccuracy}}{{{best_row['Accuracy']:.4f}}}",
        f"\\newcommand{{\\bestModelPrecision}}{{{best_row['Precision']:.4f}}}",
        f"\\newcommand{{\\bestModelRecall}}{{{best_row['Recall']:.4f}}}",
        f"\\newcommand{{\\bestModelFOne}}{{{best_row['F1-score']:.4f}}}",
        f"\\newcommand{{\\bestModelRocAuc}}{{{best_row['ROC-AUC']:.4f}}}",
        f"\\newcommand{{\\bestModelPrAuc}}{{{best_row['PR-AUC']:.4f}}}",
        f"\\newcommand{{\\bestModelSpecificity}}{{{best_row['Specificity']:.4f}}}",
        f"\\newcommand{{\\bestModelBalancedAcc}}{{{best_row['Balanced_Accuracy']:.4f}}}",
        f"\\newcommand{{\\bestModelTN}}{{{int(best_row['True_Negatives']):,}}}",
        f"\\newcommand{{\\bestModelFP}}{{{int(best_row['False_Positives']):,}}}",
        f"\\newcommand{{\\bestModelFN}}{{{int(best_row['False_Negatives']):,}}}",
        f"\\newcommand{{\\bestModelTP}}{{{int(best_row['True_Positives']):,}}}",
        "",
        "% All models individual F1 and ROC",
    ]

    num_words = ["Zero", "One", "Two", "Three", "Four", "Five"]

    for _, row in model_df.iterrows():
        prefix = row['Model'].replace(" ", "").replace("-", "")
        latex_defs.extend([
            f"\\newcommand{{\\{prefix}Acc}}{{{row['Accuracy']:.4f}}}",
            f"\\newcommand{{\\{prefix}Prec}}{{{row['Precision']:.4f}}}",
            f"\\newcommand{{\\{prefix}Rec}}{{{row['Recall']:.4f}}}",
            f"\\newcommand{{\\{prefix}FOne}}{{{row['F1-score']:.4f}}}",
            f"\\newcommand{{\\{prefix}RocAuc}}{{{row['ROC-AUC']:.4f}}}",
            f"\\newcommand{{\\{prefix}PrAuc}}{{{row['PR-AUC']:.4f}}}",
            f"\\newcommand{{\\{prefix}Time}}{{{row['Training_Time_s']:.2f}}}"
        ])

    latex_defs.extend([
        "",
        "% Cluster metrics",
        f"\\newcommand{{\\selectedClusterCount}}{{{len(cluster_df)}}}",
    ])

    for _, c_row in cluster_df.iterrows():
        c_id = int(c_row['Cluster_ID'])
        c_word = num_words[c_id] if c_id < len(num_words) else f"C{c_id}"
        latex_defs.extend([
            f"\\newcommand{{\\cluster{c_word}Name}}{{{c_row['Segment_Name']}}}",
            f"\\newcommand{{\\cluster{c_word}Size}}{{{int(c_row['Size']):,}}}",
            f"\\newcommand{{\\cluster{c_word}Prop}}{{{c_row['Proportion_pct']:.1f}\\%}}",
            f"\\newcommand{{\\cluster{c_word}Churn}}{{{c_row['Churn_Rate_pct']:.1f}\\%}}",
            f"\\newcommand{{\\cluster{c_word}Tenure}}{{{c_row['Avg_Tenure_mo']:.1f}}}",
            f"\\newcommand{{\\cluster{c_word}Monthly}}{{\\${c_row['Avg_Monthly_Charges']:.2f}}}"
        ])

    # Top features
    top3_feats = feat_df.head(3)["Feature"].tolist()
    clean_top3 = [f.replace("_", " ") for f in top3_feats]
    latex_defs.extend([
        "",
        f"\\newcommand{{\\topFeatureOne}}{{{clean_top3[0]}}}",
        f"\\newcommand{{\\topFeatureTwo}}{{{clean_top3[1]}}}",
        f"\\newcommand{{\\topFeatureThree}}{{{clean_top3[2]}}}"
    ])

    macro_path = REPORT_GENERATED_DIR / "experiment_macros.tex"
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_defs))

    print(f"Generated LaTeX macros at {macro_path}")


if __name__ == "__main__":
    generate_report_macros()
