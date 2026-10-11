"""Exploratory analysis. Saves at least 8 figures under artifacts/figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURE_DIR, RANDOM_STATE  # noqa: E402
from src.data.features import engineer_features, risk_from_score  # noqa: E402
from src.data.load import load_raw_dataset  # noqa: E402

sns.set_theme(style="whitegrid")


def run_eda() -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_dataset()
    frame = engineer_features(raw)
    frame["support_need"] = frame["G3"].apply(risk_from_score)

    observations = []

    # 1. Final grade distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(frame["G3"], bins=21, kde=True, ax=ax, color="#2c6e49")
    ax.set_title("Final grade (G3) distribution")
    ax.set_xlabel("Final grade (0-20)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_g3_distribution.png", dpi=140)
    plt.close(fig)
    observations.append(
        f"Mean G3 is {frame['G3'].mean():.2f} (sd {frame['G3'].std():.2f}); "
        "the mass sits around the pass mark of 10."
    )

    # 2. Support class balance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["high_support", "medium_support", "low_support"]
    sns.countplot(data=frame, x="support_need", order=order, ax=ax, hue="support_need", legend=False)
    ax.set_title("Support-need class balance (derived from G3)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_class_balance.png", dpi=140)
    plt.close(fig)
    counts = frame["support_need"].value_counts().to_dict()
    observations.append(f"Class counts: {counts}. The classifier must not ignore the smaller high-support group.")

    # 3. G3 by study time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=frame, x="studytime", y="G3", ax=ax)
    ax.set_title("Final grade by weekly study-time band")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_g3_by_studytime.png", dpi=140)
    plt.close(fig)
    observations.append("Higher study-time bands tend to show higher median G3, with overlap that rules out a simple rule.")

    # 4. Absences vs G3
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=frame.sample(min(500, len(frame)), random_state=RANDOM_STATE), x="absences", y="G3", hue="subject", alpha=0.6, ax=ax)
    ax.set_title("Absences versus final grade")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_absences_vs_g3.png", dpi=140)
    plt.close(fig)
    observations.append("Very high absence counts are uncommon; moderate absences still appear across the grade range.")

    # 5. Failures vs G3
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=frame, x="failures", y="G3", ax=ax)
    ax.set_title("Final grade by number of past failures")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_g3_by_failures.png", dpi=140)
    plt.close(fig)
    observations.append("Past failures are one of the strongest raw separators of final grade.")

    # 6. Subject comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=frame, x="G3", hue="subject", fill=True, common_norm=False, ax=ax)
    ax.set_title("Final grade density by subject")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_g3_by_subject.png", dpi=140)
    plt.close(fig)
    observations.append(
        f"Math mean G3={frame.loc[frame.subject=='math','G3'].mean():.2f}; "
        f"Portuguese mean G3={frame.loc[frame.subject=='portuguese','G3'].mean():.2f}."
    )

    # 7. Correlation heatmap of numeric academic features
    num_cols = [
        "age",
        "Medu",
        "Fedu",
        "studytime",
        "failures",
        "absences",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "parent_edu_avg",
        "attendance_proxy",
        "engagement_score",
        "G3",
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(frame[num_cols].corr(), cmap="RdBu_r", center=0, ax=ax, annot=False)
    ax.set_title("Numeric feature correlations (including G3)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_correlation_heatmap.png", dpi=140)
    plt.close(fig)
    observations.append("G1/G2 are excluded from modelling because they correlate so strongly with G3 that they leak the target.")

    # 8. Engagement score by support need
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=frame, x="support_need", y="engagement_score", order=order, ax=ax)
    ax.set_title("Engineered engagement score by support need")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08_engagement_by_support.png", dpi=140)
    plt.close(fig)
    observations.append("The engineered engagement score is lower on average in the high-support group.")

    # 9. Wants higher education
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=frame, x="higher", y="G3", ax=ax)
    ax.set_title("Final grade by intention to pursue higher education")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "09_g3_by_higher.png", dpi=140)
    plt.close(fig)
    observations.append("Students who do not plan higher education have a visibly lower G3 distribution.")

    # 10. Missingness bar (should be near zero on UCI files)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    miss = frame.isna().mean().sort_values(ascending=False).head(12)
    miss.plot(kind="bar", ax=ax, color="#bc4749")
    ax.set_title("Fraction missing by column")
    ax.set_ylabel("fraction missing")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10_missingness.png", dpi=140)
    plt.close(fig)
    observations.append(f"Missing-cell fraction is {frame.isna().mean().mean():.4f}; imputation still exists in the pipeline for uploaded data.")

    payload = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "observations": observations,
    }
    (FIGURE_DIR / "eda_observations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", len(list(FIGURE_DIR.glob("*.png"))), "figures to", FIGURE_DIR)
    return payload


if __name__ == "__main__":
    run_eda()
