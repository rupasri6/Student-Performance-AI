"""Shared institution dashboard for faculty and student views."""

import pandas as pd
import streamlit as st

from src.data.college import assess_student_risk, make_study_plan
from database import database as db

def _college_dashboard(role: str):
    records = db.students_frame()
    st.title(f"Trekverse AI Academy · {role.title()} Dashboard")
    if records.empty:
        st.info("No college records yet. Ask an administrator to import the roster.")
        return
    for col in ["Final_Marks", "Attendance", "Backlogs", "Current_CGPA"]:
        records[col] = pd.to_numeric(records[col], errors="coerce")
    sid = st.session_state["user"].get("student_id") if role == "student" else None
    if sid:
        records = records[records["Student_ID"].astype(str) == str(sid)]
        if records.empty:
            st.warning("No record is linked to this student account.")
            return
    unique = _latest_student_profiles(records)
    unique["Risk_Assessment"] = unique.apply(assess_student_risk, axis=1)
    unique["Risk"] = unique["Risk_Assessment"].map(lambda value: value["level"])
    unique["Risk_Score"] = unique["Risk_Assessment"].map(lambda value: value["score"])
    unique["Risk_Factors"] = unique["Risk_Assessment"].map(
        lambda value: ", ".join(factor["factor"] for factor in value["factors"] if factor["flagged"]) or "None flagged"
    )
    risk_by_student = unique.set_index(unique["Student_ID"].astype(str))["Risk"]
    score_by_student = unique.set_index(unique["Student_ID"].astype(str))["Risk_Score"]
    assessment_by_student = unique.set_index(unique["Student_ID"].astype(str))["Risk_Assessment"]
    factors_by_student = unique.set_index(unique["Student_ID"].astype(str))["Risk_Factors"]
    records["Risk"] = records["Student_ID"].astype(str).map(risk_by_student)
    records["Risk_Score"] = records["Student_ID"].astype(str).map(score_by_student)
    records["Risk_Assessment"] = records["Student_ID"].astype(str).map(assessment_by_student)
    records["Risk_Factors"] = records["Student_ID"].astype(str).map(factors_by_student)
    try:
        from src.models.college import segment_college_students
        segments = segment_college_students(db.students_frame())
        unique = unique.merge(segments, on="Student_ID", how="left")
    except (ValueError, KeyError):
        unique["Engagement_Segment"] = "Unavailable"
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Students", len(unique))
    k2.metric("Class average", f"{unique['Final_Marks'].mean():.1f}" if unique["Final_Marks"].notna().any() else "—")
    k3.metric("Average attendance", f"{unique['Attendance'].mean():.1f}%")
    k4.metric("At-risk students", int(unique["Risk"].isin(["High Risk", "Moderate Risk"]).sum()))
    k5.metric("Average CGPA", f"{unique['Current_CGPA'].mean():.2f}/10" if unique["Current_CGPA"].notna().any() else "—")
    if unique["Engagement_Segment"].notna().any():
        st.caption("Student segments (K-Means using academic and engagement indicators): " +
                   ", ".join(f"{name}: {count}" for name, count in unique["Engagement_Segment"].value_counts().items()))
    if role == "faculty":
        st.subheader("Current risk distribution")
        st.caption("Risk bands combine the available attendance, backlog, internal and assignment marks, study time, CGPA, quiz, submission, and learning-activity indicators. Missing factors are excluded and data coverage is shown per student.")
        risk_counts = unique["Risk"].value_counts().reindex(["High Risk", "Moderate Risk", "Low Risk"], fill_value=0)
        st.bar_chart(risk_counts)
        unique["Intervention"] = unique["Risk_Assessment"].map(_intervention_for_assessment)
        st.subheader("Faculty intervention queue")
        st.dataframe(unique.sort_values("Risk_Score", ascending=False, na_position="last")[["Student_ID", "Name", "Department", "Attendance", "Backlogs", "Final_Marks", "Risk_Score", "Risk", "Risk_Factors", "Engagement_Segment", "Intervention"]],
                     use_container_width=True, hide_index=True)
        st.subheader("Top performers")
        st.dataframe(unique.nlargest(5, "Final_Marks")[["Student_ID", "Name", "Department", "Final_Marks"]],
                     use_container_width=True, hide_index=True)
        st.subheader("Cohort subject analysis")
        cohort_subjects = records.dropna(subset=["Subject"]).groupby("Subject")["Final_Marks"].mean().sort_values(ascending=False)
        if not cohort_subjects.empty:
            st.bar_chart(cohort_subjects)
        cohort_trend = records.groupby("Semester")["Final_Marks"].mean().dropna()
        if len(cohort_trend) > 1:
            st.subheader("Cohort semester trend")
            st.line_chart(cohort_trend)
        comparison_ids = st.multiselect("Compare students", sorted(records["Student_ID"].astype(str).unique()),
                                        default=sorted(records["Student_ID"].astype(str).unique())[:2], key="compare_students")
        if comparison_ids:
            comparison = records[records["Student_ID"].astype(str).isin(comparison_ids)]
            st.dataframe(comparison[["Student_ID", "Name", "Subject", "Semester", "Attendance", "Final_Marks", "Risk_Score", "Risk"]],
                         use_container_width=True, hide_index=True)
        st.dataframe(records.sort_values("Final_Marks"), use_container_width=True, hide_index=True)
        selection = st.selectbox("Student profile", sorted(records["Student_ID"].astype(str).unique()))
        rows = records[records["Student_ID"].astype(str) == selection]
    else:
        rows = records
        selection = str(sid)
    row = rows.sort_values("Semester", ascending=False).iloc[0]
    st.subheader(f"Profile · {row.get('Name', selection)} ({selection})")
    current_cgpa = row.get("Current_CGPA")
    previous_cgpa = row.get("Previous_CGPA")
    current_cgpa_label = f"{float(current_cgpa):.2f}/10" if pd.notna(current_cgpa) else "Not provided"
    previous_cgpa_label = f"{float(previous_cgpa):.2f}/10" if pd.notna(previous_cgpa) else "Not provided"
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Current CGPA", current_cgpa_label)
    p2.metric("Previous CGPA", previous_cgpa_label)
    p3.metric("Attendance", f"{float(row.get('Attendance') or 0):.1f}%")
    p4.metric("Early-warning risk", row.get("Risk", "Unavailable"))
    risk_assessment = row["Risk_Assessment"]
    p5.metric("Risk score", f"{risk_assessment['score']:.1f}/100" if risk_assessment["score"] is not None else "Unavailable")
    st.caption(f"Risk factors used: {len(risk_assessment['factors'])}/9 · Data coverage: {risk_assessment['coverage_percent']:.0f}%")
    flagged = [factor for factor in risk_assessment["factors"] if factor["flagged"]]
    if flagged:
        st.write("**Factors for review:** " + ", ".join(f"{item['factor']} ({item['value']:g})" for item in flagged))
    else:
        st.write("No risk factors crossed the review threshold in the available data.")
    st.caption(f"Department: {row.get('Department', '')} · Semester: {row.get('Semester', '')}")
    if not rows.empty:
        trend = rows.groupby("Semester", dropna=True)["Final_Marks"].mean().dropna()
        if len(trend) > 1:
            st.subheader("Semester performance trend")
            st.line_chart(trend)
    if role == "faculty":
        with st.expander("Update attendance and marks"):
            with st.form(f"edit_record_{selection}"):
                subject_name = st.selectbox("Subject to update", sorted(rows["Subject"].dropna().astype(str).unique()) or ["Overall"])
                attendance = st.number_input("Attendance (%)", 0.0, 100.0, float(row.get("Attendance") or 0), 0.5)
                cgpa = st.number_input("Current CGPA", 0.0, 10.0,
                                       float(row.get("Current_CGPA")) if pd.notna(row.get("Current_CGPA")) else 0.0, 0.1)
                previous_cgpa_input = st.number_input("Previous CGPA", 0.0, 10.0,
                                                      float(row.get("Previous_CGPA")) if pd.notna(row.get("Previous_CGPA")) else 0.0, 0.1)
                study_hours = st.number_input("Study hours/day", 0.0, 24.0, float(row.get("Study_Hours") or 0), 0.5)
                backlogs = st.number_input("Backlogs", 0, 100, int(row.get("Backlogs") or 0))
                internal = st.number_input("Internal marks", 0.0, 100.0, float(row.get("Internal_Marks") or 0), 0.5)
                quiz = st.number_input("Quiz marks", 0.0, 100.0, float(row.get("Quiz_Marks") or 0), 0.5)
                final_marks = st.number_input("Final marks", 0.0, 100.0, float(row.get("Final_Marks") or 0), 0.5)
                assignments = st.number_input("Assignment marks", 0.0, 100.0, float(row.get("Assignment_Marks") or 0), 0.5)
                submission_rate = st.number_input("Assignment submission rate (%)", 0.0, 100.0,
                                                  float(row.get("Assignment_Submission_Rate") or 0), 0.5)
                learning_activity = st.number_input("Learning activity (0-10)", 0.0, 10.0,
                                                    float(row.get("Learning_Activity") or 0), 0.5)
                save_edit = st.form_submit_button("Save academic record")
            if save_edit:
                db.update_academic_record(selection, subject_name, int(row.get("Semester") or 1), {
                    "Attendance": attendance, "Current_CGPA": cgpa, "Previous_CGPA": previous_cgpa_input, "Study_Hours": study_hours,
                    "Backlogs": backlogs, "Internal_Marks": internal, "Quiz_Marks": quiz,
                    "Final_Marks": final_marks, "Assignment_Marks": assignments,
                    "Assignment_Submission_Rate": submission_rate,
                    "Learning_Activity": learning_activity,
                })
                st.success("Academic record updated.")
                st.rerun()
    st.subheader("Subject performance")
    subjects = rows[["Subject", "Final_Marks"]].dropna().groupby("Subject", as_index=False)["Final_Marks"].mean()
    if not subjects.empty:
        st.bar_chart(subjects.set_index("Subject"))
        weak = subjects.loc[subjects["Final_Marks"] < 65, "Subject"].astype(str).tolist()
        strong = subjects.loc[subjects["Final_Marks"] >= 65, "Subject"].astype(str).tolist()
        st.write("**Strong subjects:** " + (", ".join(strong) if strong else "No subjects at or above 65 marks yet"))
        st.write("**Needs attention:** " + (", ".join(weak) if weak else "No low subject marks recorded"))
        st.subheader("Weekly study plan")
        st.dataframe(pd.DataFrame(make_study_plan(weak, row)), use_container_width=True, hide_index=True)
    st.subheader("Performance history and prediction")
    history = db.student_history(selection)
    if not history.empty:
        history_chart = history.set_index("created_at")[["predicted_score"]].copy()
        if "risk_score" in history.columns and history["risk_score"].notna().any():
            history_chart["risk_score"] = pd.to_numeric(history["risk_score"], errors="coerce")
        st.line_chart(history_chart)
        st.dataframe(history, use_container_width=True, hide_index=True)
    if st.button("Generate and save prediction", key=f"college_predict_{role}_{selection}"):
        try:
            from src.models.college import predict_college, train_college_model
            all_rows = db.students_frame()
            # Hold this student's entire history out of training for an honest profile prediction.
            train_rows = all_rows[all_rows["Student_ID"].astype(str) != selection]
            train_college_model(train_rows)
            result = predict_college(row.to_dict())
            result["recommendations"] = make_study_plan(weak if not subjects.empty else [], row)
            st.session_state[f"college_result_{selection}"] = result
            db.save_prediction(selection, result)
        except ValueError as exc:
            st.error(str(exc))
    prediction = st.session_state.get(f"college_result_{selection}")
    if prediction:
        p1, p2 = st.columns(2)
        p1.metric("Predicted performance", f"{prediction['predicted_score']:.1f}/100")
        p2.metric("Early-warning risk", prediction["risk_level"])
        st.caption(f"Risk score: {prediction.get('risk_score', 'Unavailable')} / 100 · Risk factors use observed profile data, not the simulated score.")
        st.caption("This is a model estimate based on available historical college records, not a guaranteed outcome.")
        explanation = prediction.get("explanation", [])
        if explanation:
            st.subheader("Prediction explanation")
            st.caption(f"Method: {prediction.get('explanation_method', 'local explanation')}")
            st.dataframe(pd.DataFrame(explanation), use_container_width=True, hide_index=True)
        try:
            from src.models.college import predict_college
            latest_semester = rows["Semester"].max()
            subject_predictions = rows[(rows["Semester"] == latest_semester) & rows["Subject"].notna()].copy()
            if not subject_predictions.empty:
                subject_predictions["Predicted_Marks"] = subject_predictions.apply(
                    lambda item: predict_college(item.to_dict())["predicted_score"], axis=1)
                st.subheader("Subject-wise predicted performance")
                st.bar_chart(subject_predictions.groupby("Subject")["Predicted_Marks"].mean())
        except (ValueError, FileNotFoundError):
            pass
        st.subheader("What-if simulator")
        st.caption("Hypothetical model simulation only; changing inputs does not guarantee a real-world outcome.")
        with st.expander("Adjust academic factors"):
            scenario = dict(row)
            base_attendance = float(row.get("Attendance") or 0)
            base_hours = float(row.get("Study_Hours") or 0)
            base_submission = row.get("Assignment_Submission_Rate")
            missing_submission = pd.isna(base_submission)
            if missing_submission:
                base_submission = 75
            w1, w2, w3 = st.columns(3)
            scenario["Attendance"] = w1.slider("Attendance (%)", 0, 100, int(base_attendance), key=f"whatif_att_{selection}")
            scenario["Study_Hours"] = w2.slider("Study hours/day", 0.0, 12.0, min(12.0, base_hours), 0.5, key=f"whatif_hours_{selection}")
            scenario["Assignment_Submission_Rate"] = w3.slider("Assignment completion (%)", 0, 100, int(base_submission), key=f"whatif_assign_{selection}")
            if missing_submission:
                st.caption("No completion rate is recorded; the scenario starts at 75% while the baseline uses model imputation.")
            if st.button("Simulate changes", key=f"simulate_college_{selection}"):
                try:
                    from src.models.college import predict_college
                    scenario_result = predict_college(scenario)
                    delta = scenario_result["predicted_score"] - prediction["predicted_score"]
                    st.metric("Simulated final marks", f"{scenario_result['predicted_score']:.1f}/100", f"{delta:+.1f} vs current")
                    score = scenario_result.get("risk_score")
                    risk_detail = f"risk score {score:.1f}/100" if score is not None else "risk score unavailable"
                    st.caption(f"Simulated risk band: {scenario_result['risk_level']} ({risk_detail}) · This is a model simulation, not a guaranteed outcome.")
                except ValueError as exc:
                    st.error(str(exc))
    else:
        prediction = {
            "predicted_score": None,
            "risk_level": row["Risk"],
            "risk_score": row["Risk_Score"],
            "risk_factors": row["Risk_Assessment"]["factors"],
            "explanation": [],
        }
def _intervention_for_assessment(assessment: dict) -> str:
    flagged = {item["factor"] for item in assessment["factors"] if item["flagged"]}
    actions = []
    if "Attendance" in flagged:
        actions.append("attendance check-in")
    if "Backlogs" in flagged:
        actions.append("tutor review for backlogs")
    if flagged & {"Internal_Marks", "Assignment_Marks", "Quiz_Marks"}:
        actions.append("targeted subject practice")
    if "Assignment_Submission_Rate" in flagged:
        actions.append("assignment completion plan")
    if flagged & {"Study_Hours", "Learning_Activity"}:
        actions.append("weekly study routine")
    if not actions:
        return "Continue support and review progress next month"
    return "Faculty check-in: " + "; ".join(actions)


def _latest_student_profiles(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all subjects from each student's most recent semester."""
    latest_semester = records.groupby("Student_ID")["Semester"].transform("max")
    latest = records[records["Semester"] == latest_semester]
    aggregations = {
        "Name": "first",
        "Department": "first",
        "Semester": "max",
        "Current_CGPA": "first",
        "Previous_CGPA": "first",
        "Attendance": "mean",
        "Study_Hours": "mean",
        "Backlogs": "max",
        "Internal_Marks": "mean",
        "Assignment_Marks": "mean",
        "Quiz_Marks": "mean",
        "Final_Marks": "mean",
        "Assignment_Submission_Rate": "mean",
        "Learning_Activity": "mean",
    }
    aggregations = {column: operation for column, operation in aggregations.items() if column in latest.columns}
    return latest.groupby("Student_ID", as_index=False).agg(aggregations)
