"""Role-safe download of the selected student's academic report."""

import pandas as pd
import streamlit as st

from reports.report_generator import _build_college_pdf
from src.data.college import assess_student_risk
from database import database as db
from ._college_dashboard import _latest_student_profiles


def page_reports():
    st.title("Academic reports")
    records = db.students_frame()
    if records.empty:
        st.info("No institution records are available for a report yet.")
        return

    role = st.session_state["user"]["role"]
    if role == "STUDENT":
        student_id = st.session_state["user"].get("student_id")
        records = records[records["Student_ID"].astype(str) == str(student_id)]
        if records.empty:
            st.warning("No academic record is linked to this student account.")
            return
        student_ids = [str(student_id)]
    else:
        student_ids = sorted(records["Student_ID"].astype(str).unique())
        student_id = st.selectbox("Student", student_ids)
        records = records[records["Student_ID"].astype(str) == student_id]

    profiles = _latest_student_profiles(records)
    profile = profiles.iloc[0].to_dict()
    profile["Risk_Assessment"] = assess_student_risk(profile)
    subjects = records[records["Semester"] == profile["Semester"]].copy()
    subject_summary = subjects.dropna(subset=["Subject", "Final_Marks"]).groupby(
        "Subject", as_index=False
    )["Final_Marks"].mean()
    weak = subject_summary.loc[subject_summary["Final_Marks"] < 65, "Subject"].astype(str).tolist()
    strong = subject_summary.loc[subject_summary["Final_Marks"] >= 65, "Subject"].astype(str).tolist()
    prediction = st.session_state.get(f"college_result_{student_id}")
    if prediction is None:
        prediction = {
            "predicted_score": None,
            "risk_level": profile["Risk_Assessment"]["level"],
            "risk_score": profile["Risk_Assessment"]["score"],
            "explanation": [],
        }
        st.caption("Generate a model estimate from the Faculty dashboard before including one in this report. Recorded marks are shown separately.")

    if role != "STUDENT":
        remarks = st.text_area("Faculty remarks", max_chars=600)
    else:
        remarks = ""
    st.subheader(f"Report preview · {profile.get('Name', student_id)}")
    st.write(f"Student ID: {student_id} · Semester: {profile.get('Semester', '—')}")
    st.write(f"Early-warning band: {profile['Risk_Assessment']['level']} · Risk score: {profile['Risk_Assessment']['score'] if profile['Risk_Assessment']['score'] is not None else 'Unavailable'} / 100")
    if prediction.get("predicted_score") is not None:
        st.write(f"Latest model estimate: {prediction['predicted_score']:.1f}/100")
    else:
        st.write("Model estimate: not generated")
    pdf = _build_college_pdf(profile, prediction, subject_summary, weak, strong, remarks)
    st.download_button(
        "Download academic report (PDF)",
        pdf,
        f"student_{student_id}_report.pdf",
        "application/pdf",
        type="primary",
    )
