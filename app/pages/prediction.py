"""Reference model prediction and student-facing workflow."""

import json

import pandas as pd
import streamlit as st

from shared import DEFAULTS, _form_values, _models_ready, _prediction_summary, _train_help
from reports.report_generator import _build_pdf_report
from src.config import MODEL_DIR
from src.models.inference import predict_student

def page_home():
    st.title("Student Performance & Early-Warning System")
    st.write(
        "A role-based academic support platform for reviewing student progress, identifying support needs, "
        "and planning timely interventions."
    )
    st.info(
        "Model estimates and risk bands support academic review. They should be considered with the student’s "
        "context and are not decisions about a student’s ability."
    )
    if _models_ready():
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Best regressor", metrics.get("best_regressor", "—"))
        c2.metric("Best classifier", metrics.get("best_classifier", "—"))
        c3.metric("Test rows", metrics.get("n_test", "—"))
        st.caption(metrics.get("selection_rationale", ""))
        st.success("The reference student-performance models are ready. Use the Faculty dashboard for institution records.")
    else:
        _train_help()

def page_predict():
    st.title("Reference student prediction")
    st.caption("Enter permitted academic and engagement fields only. Do not enter names, IDs, or contact details.")
    if not _models_ready():
        _train_help()
        return
    st.info("Enter the student profile fields and select **Predict** to review the estimate and supporting explanation.")
    payload = _form_values()
    if st.button("Predict", type="primary"):
        try:
            result = predict_student(payload)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        if not result["ok"]:
            st.error("Please fix the following:")
            for err in result["errors"]:
                st.write(f"- {err}")
            return
        st.session_state["last_result"] = result
        st.session_state["last_payload"] = payload

    result = st.session_state.get("last_result")
    if not result:
        return

    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted final score", result["predicted_score"])
    c2.metric("Support group", result["predicted_support"].replace("_", " "))
    c3.metric("Engagement cluster", result["engagement_cluster"])
    st.write("Class probabilities", result["class_probabilities"])
    st.warning(result["disclaimer"])

    st.subheader("Why this prediction?")
    top = result.get("explanation", {}).get("top", [])
    if top:
        st.caption(f"Explanation method: {result.get('explanation_method', 'local explanation')}")
        st.dataframe(pd.DataFrame(top), use_container_width=True)
        feature_names = [item.get("feature", "") for item in top]
        selected_feature = st.selectbox("Inspect a feature", feature_names, key="explanation_feature")
        selected = next(item for item in top if item.get("feature") == selected_feature)
        contribution = selected.get("shap", selected.get("delta_if_typical", 0))
        st.metric("Contribution to model estimate", f"{contribution:+.3f}")
        st.caption(
            "Positive `delta_if_typical` means this student’s value on that feature raised the predicted "
            "score relative to a typical training student."
        )

    st.subheader("Recommended learning actions")
    for rec in result["recommendations"]:
        st.markdown(f"**{rec['gap']}** — {rec['action']}  \nResource: *{rec['resource']}*")

    st.subheader("What-if simulator")
    st.caption("Simulation based on the trained model. It is not a guarantee that changing these factors will cause this result.")
    baseline = st.session_state.get("last_payload", payload)
    with st.expander("Explore a hypothetical change"):
        scenario = dict(baseline)
        s1, s2 = st.columns(2)
        with s1:
            scenario["absences"] = st.slider("Hypothetical absences", 0, 93, int(baseline["absences"]))
        with s2:
            scenario["studytime"] = st.slider("Hypothetical weekly study-time band", 1, 4, int(baseline["studytime"]))
        if st.button("Simulate scenario", key="simulate"):
            simulation = predict_student(scenario)
            if simulation.get("ok"):
                delta = simulation["predicted_score"] - result["predicted_score"]
                st.metric("Simulated score", simulation["predicted_score"], delta=f"{delta:+.2f} vs current")
                st.caption("The model maps absences to an attendance proxy; it does not accept attendance percentage directly.")
            else:
                st.error("Could not simulate this scenario.")

    st.subheader("Student report")
    report = _build_pdf_report(payload, result)
    st.download_button("Download PDF report", report, "student_performance_report.pdf", "application/pdf")

    st.download_button(
        "Download prediction summary (JSON)",
        data=json.dumps(_prediction_summary(result), indent=2),
        file_name="prediction_summary.json",
        mime="application/json",
    )
