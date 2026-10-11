"""Model evaluation, charts, and responsible-use information."""

import json

import pandas as pd
import streamlit as st

from shared import _models_ready, _train_help
from src.config import FIGURE_DIR, MODEL_DIR

def page_metrics():
    st.title("Model Performance")
    st.caption("Held-out evaluation results for the reference UCI dataset. Institution models are evaluated separately after data is uploaded.")
    if not _models_ready():
        _train_help()
        return
    metrics_path = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Regression (test set)")
    reg_rows = []
    for name, vals in metrics["regression"].items():
        if isinstance(vals, dict) and "rmse" in vals:
            reg_rows.append({"model": name, **{k: vals[k] for k in ("mae", "rmse", "r2", "cv_rmse") if k in vals}})
    st.dataframe(pd.DataFrame(reg_rows).sort_values("rmse"), use_container_width=True)

    st.subheader("Classification (test set)")
    cls_rows = []
    for name, vals in metrics["classification"].items():
        if isinstance(vals, dict) and "f1_macro" in vals:
            cls_rows.append(
                {
                    "model": name,
                    "accuracy": vals.get("accuracy"),
                    "precision_macro": vals.get("precision_macro"),
                    "recall_macro": vals.get("recall_macro"),
                    "f1_macro": vals.get("f1_macro"),
                    "cv_f1_macro": vals.get("cv_f1_macro"),
                    "roc_auc_ovr": vals.get("roc_auc_ovr"),
                }
            )
    st.dataframe(pd.DataFrame(cls_rows).sort_values("f1_macro", ascending=False), use_container_width=True)

    st.subheader("Class balance")
    st.json({"train": metrics.get("class_balance_train"), "test": metrics.get("class_balance_test")})

    st.subheader("Permutation importance (regression)")
    st.dataframe(pd.DataFrame(metrics.get("permutation_importance_regression", [])), use_container_width=True)

    local_examples = metrics.get("shap", {}).get("local", [])
    if len(local_examples) >= 3:
        st.subheader("Individual prediction explanations")
        st.caption("Three held-out reference examples. Explanations use SHAP or local feature replacement; positive values indicate a higher estimate relative to the explanation baseline.")
        for number, example in enumerate(local_examples[:3], start=1):
            with st.expander(f"Example {number} · reference row {example.get('row_index', 'N/A')}"):
                st.caption(f"Method: {example.get('method', 'SHAP')}")
                st.dataframe(pd.DataFrame(example.get("contributions", [])), use_container_width=True, hide_index=True)

    st.subheader("EDA charts")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        st.image(str(path), caption=path.name, use_container_width=True)

def page_ethics():
    st.title("Limitations, privacy, and intended use")
    st.markdown(
        """
- Predictions are **probabilistic**. The project never claims 100% accuracy.
- Uploaded student records are stored in a local SQLite database and are visible according to account role.
- Passwords are stored as salted PBKDF2 hashes. Protect the host, database file, and exported reports as student data.
- Sex was excluded from the model to avoid encoding gender stereotypes as academic ability.
- G1 and G2 were excluded to prevent target leakage.
- Alcohol, health, and family fields are sensitive; treat explanations as prompts for support, not labels.
- College risk bands are transparent threshold rules. Review context with a student before taking action.
- After deployment, monitor for **model drift**: grade distributions, absence patterns, and error rates should be reviewed each term.
"""
    )
