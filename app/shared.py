"""Shared form values and small Streamlit helpers."""

import streamlit as st

from src.config import MODEL_DIR, PROJECT_ROOT

DEFAULTS = {
    "school": "GP",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 2,
    "health": 3,
    "absences": 4,
    "subject": "math",
}

def _models_ready() -> bool:
    return (MODEL_DIR / "metrics.json").exists() and (MODEL_DIR / "regression_pipeline.joblib").exists()

def _prediction_summary(result: dict) -> dict:
    """Build a shareable prediction summary without student identifiers or input fields."""
    return {
        "predicted_score": result["predicted_score"],
        "support_group": result["predicted_support"],
        "class_probabilities": result["class_probabilities"],
        "key_factors": result.get("explanation", {}).get("top", [])[:5],
        "recommendations": [item["gap"] for item in result["recommendations"]],
        "disclaimer": result["disclaimer"],
    }

def _train_help() -> None:
    st.warning(
        "This app session does not see the trained model files yet. "
        "Stop Streamlit with Ctrl+C, then in the **project folder** run the commands below, "
        "start Streamlit again, and click **Rerun** in the browser."
    )
    st.code(
        f'cd "{PROJECT_ROOT}"\n'
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "python -m src.models.train\n"
        "streamlit run app/app.py",
        language="powershell",
    )
    st.caption(f"Looking for models in: `{MODEL_DIR}`  |  Project root: `{PROJECT_ROOT}`")

def _form_values() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        school = st.selectbox("School", ["GP", "MS"], index=0)
        age = st.number_input("Age", min_value=15, max_value=22, value=DEFAULTS["age"])
        address = st.selectbox("Address type", ["U", "R"], help="U = urban, R = rural")
        famsize = st.selectbox("Family size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parent cohabitation", ["T", "A"], help="T = together, A = apart")
        Medu = st.slider("Mother education (0-4)", 0, 4, DEFAULTS["Medu"])
        Fedu = st.slider("Father education (0-4)", 0, 4, DEFAULTS["Fedu"])
        Mjob = st.selectbox("Mother job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father job", ["teacher", "health", "services", "at_home", "other"])
    with c2:
        reason = st.selectbox("Reason for school choice", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.slider("Travel time band (1-4)", 1, 4, DEFAULTS["traveltime"])
        studytime = st.slider("Weekly study time band (1-4)", 1, 4, DEFAULTS["studytime"])
        failures = st.slider("Past failures", 0, 4, DEFAULTS["failures"])
        absences = st.number_input("Absences", min_value=0, max_value=93, value=DEFAULTS["absences"])
        famrel = st.slider("Family relationship quality", 1, 5, DEFAULTS["famrel"])
        freetime = st.slider("Free time", 1, 5, DEFAULTS["freetime"])
        goout = st.slider("Going out", 1, 5, DEFAULTS["goout"])
    with c3:
        Dalc = st.slider("Weekday alcohol", 1, 5, DEFAULTS["Dalc"])
        Walc = st.slider("Weekend alcohol", 1, 5, DEFAULTS["Walc"])
        health = st.slider("Health", 1, 5, DEFAULTS["health"])
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Paid extra classes", ["no", "yes"])
        activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Home internet", ["yes", "no"])
        romantic = st.selectbox("Romantic relationship", ["no", "yes"])
        subject = st.selectbox("Subject", ["math", "portuguese"])
    return {
        "school": school,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": int(Medu),
        "Fedu": int(Fedu),
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(traveltime),
        "studytime": int(studytime),
        "failures": int(failures),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": int(famrel),
        "freetime": int(freetime),
        "goout": int(goout),
        "Dalc": int(Dalc),
        "Walc": int(Walc),
        "health": int(health),
        "absences": int(absences),
        "subject": subject,
    }
