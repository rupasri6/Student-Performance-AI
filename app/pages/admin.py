"""Administrator workflows for records, accounts, and model training."""

from io import BytesIO
import json

import pandas as pd
import streamlit as st

from src.config import MODEL_DIR
from src.data.college import validate_college_frame
from database import database as db

def page_admin():
    st.title("Trekverse AI Academy · Administration")
    st.caption(f"SQLite records: `{db.DB_PATH}`")
    template = pd.DataFrame([{
        "Student_ID": "STU1001", "Name": "Sample Student", "Department": "Computer Science",
        "Semester": 1, "Subject": "Mathematics", "Attendance": 82, "Internal_Marks": 70,
        "Assignment_Marks": 80, "Quiz_Marks": 75, "Study_Hours": 2.5,
        "Current_CGPA": 7.4, "Previous_CGPA": 7.2, "Backlogs": 0, "Final_Marks": 74,
        "Assignment_Submission_Rate": 90, "Learning_Activity": 6,
    }])
    st.download_button("Download college CSV template", template.to_csv(index=False), "college_student_template.csv", "text/csv")
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.info(f"UCI model: {metrics.get('best_regressor', 'not trained')} · {metrics.get('n_test', 0)} held-out records")
    college_metrics_path = MODEL_DIR / "college_metrics.json"
    if college_metrics_path.exists():
        college_metrics = json.loads(college_metrics_path.read_text(encoding="utf-8"))
        st.success(
            f"Institution model: {college_metrics.get('best_model', 'unavailable')} · "
            f"{college_metrics.get('test_rows', 0)} held-out records · "
            f"RMSE {college_metrics.get('models', {}).get(college_metrics.get('best_model'), {}).get('rmse', float('nan')):.2f}"
        )
    else:
        st.caption("Institution model: not trained yet. Upload enough labeled records, then train it below.")
    uploaded = st.file_uploader("Upload institution records (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            df = pd.read_excel(BytesIO(uploaded.getvalue()), dtype={"Student_ID": str}) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(BytesIO(uploaded.getvalue()), sep=None, engine="python", dtype={"Student_ID": str})
            issues = validate_college_frame(df)
            if issues:
                for issue in issues:
                    st.error(issue)
            else:
                st.success(f"Validated {len(df)} student-subject records.")
                st.dataframe(df.head(), use_container_width=True, hide_index=True)
                if st.button("Save records to database", type="primary"):
                    st.success(f"Saved {db.import_college_records(df)} records to SQLite.")
        except Exception as exc:
            st.error(f"Could not read or import this file: {exc}")

    st.subheader("User accounts")
    st.dataframe(db.list_users(), use_container_width=True, hide_index=True)
    with st.form("create_user"):
        st.write("Create faculty or student account")
        username = st.text_input("Username")
        password = st.text_input("Temporary password (10+ characters)", type="password")
        role = st.selectbox("Role", ["ADMIN", "FACULTY", "STUDENT"])
        student_id = st.text_input("Student ID (required for student accounts)")
        submitted = st.form_submit_button("Create account")
    if submitted:
        try:
            if role == "STUDENT" and not db.student_record(student_id):
                raise ValueError("Upload this student's academic record before creating an account.")
            db.create_user(username, password, role, student_id if role == "STUDENT" else None)
            st.success("Account created.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.subheader("College model training")
    if st.button("Train / refresh model"):
        try:
            from src.models.college import train_college_model
            st.json(train_college_model())
        except ValueError as exc:
            st.error(str(exc))
