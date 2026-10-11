"""Validation and risk helpers for institution-provided academic spreadsheets."""

from __future__ import annotations

import pandas as pd

COLLEGE_REQUIRED = {
    "Student_ID", "Attendance", "Internal_Marks", "Assignment_Marks",
    "Quiz_Marks", "Study_Hours", "Previous_CGPA", "Backlogs", "Final_Marks",
}
COLLEGE_OPTIONAL = {"Name", "Department", "Semester", "Subject", "Current_CGPA", "Assignment_Submission_Rate", "Learning_Activity"}

RISK_FACTOR_WEIGHTS = {
    "Attendance": 0.18,
    "Backlogs": 0.16,
    "Internal_Marks": 0.12,
    "Assignment_Marks": 0.12,
    "Study_Hours": 0.10,
    "Previous_CGPA": 0.10,
    "Quiz_Marks": 0.08,
    "Assignment_Submission_Rate": 0.08,
    "Learning_Activity": 0.06,
}

RISK_THRESHOLDS = {"moderate": 30.0, "high": 60.0}


def validate_college_frame(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing = COLLEGE_REQUIRED - set(frame.columns)
    if missing:
        return [f"Missing required columns: {', '.join(sorted(missing))}"]
    if frame.empty:
        return ["The uploaded file has no records."]
    if frame["Student_ID"].isna().any() or frame["Student_ID"].astype(str).str.strip().eq("").any():
        errors.append("Student_ID must be present for every row.")
    keys = ["Student_ID"] + (["Semester"] if "Semester" in frame.columns else []) + (["Subject"] if "Subject" in frame.columns else [])
    if frame.duplicated(subset=keys).any():
        errors.append("Each Student_ID and Subject combination must be unique in the uploaded file.")
    for col in COLLEGE_REQUIRED - {"Student_ID"}:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            errors.append(f"{col} must contain numeric values without blanks.")
    ranges = {
        "Attendance": (0, 100), "Internal_Marks": (0, 100),
        "Assignment_Marks": (0, 100), "Quiz_Marks": (0, 100),
        "Study_Hours": (0, 24), "Previous_CGPA": (0, 10),
        "Backlogs": (0, 100), "Final_Marks": (0, 100),
    }
    for col, (low, high) in ranges.items():
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any() and ((values < low) | (values > high)).any():
            errors.append(f"{col} must be between {low} and {high}.")
    for col in ("Semester", "Backlogs"):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if ((values % 1) != 0).any():
                errors.append(f"{col} must contain whole numbers.")
    for col in COLLEGE_OPTIONAL & set(frame.columns):
        if col in {"Semester", "Current_CGPA", "Assignment_Submission_Rate", "Learning_Activity"}:
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.isna().any():
                errors.append(f"{col} must be numeric when provided.")
            if col == "Assignment_Submission_Rate" and ((values < 0) | (values > 100)).any():
                errors.append("Assignment_Submission_Rate must be between 0 and 100.")
            if col == "Current_CGPA" and ((values < 0) | (values > 10)).any():
                errors.append("Current_CGPA must be between 0 and 10.")
            if col == "Learning_Activity" and ((values < 0) | (values > 10)).any():
                errors.append("Learning_Activity must be between 0 and 10.")
    return errors


def risk_level(score: float, attendance: float, backlogs: int) -> str:
    """Transparent early-warning thresholds for a 0-100 college mark scale."""
    if score < 50 or attendance < 65 or backlogs >= 2:
        return "High Risk"
    if score < 65 or attendance < 75 or backlogs == 1:
        return "Moderate Risk"
    return "Low Risk"


def assess_student_risk(record: dict | pd.Series) -> dict:
    """Score available academic warning factors without treating missing fields as safe."""
    fields = {
        "Attendance": (0.0, 100.0, "higher_is_better"),
        "Backlogs": (0.0, 3.0, "lower_is_better"),
        "Internal_Marks": (0.0, 100.0, "higher_is_better"),
        "Assignment_Marks": (0.0, 100.0, "higher_is_better"),
        "Study_Hours": (0.0, 6.0, "higher_is_better"),
        "Previous_CGPA": (0.0, 10.0, "higher_is_better"),
        "Quiz_Marks": (0.0, 100.0, "higher_is_better"),
        "Assignment_Submission_Rate": (0.0, 100.0, "higher_is_better"),
        "Learning_Activity": (0.0, 10.0, "higher_is_better"),
    }
    factors = []
    weighted_score = 0.0
    used_weight = 0.0
    for name, (minimum, maximum, direction) in fields.items():
        value = pd.to_numeric(record.get(name), errors="coerce")
        if pd.isna(value):
            continue
        value = float(min(maximum, max(minimum, value)))
        normalized = (value - minimum) / (maximum - minimum)
        risk = (1.0 - normalized) * 100 if direction == "higher_is_better" else normalized * 100
        weight = RISK_FACTOR_WEIGHTS[name]
        weighted_score += risk * weight
        used_weight += weight
        factors.append({
            "factor": name,
            "value": value,
            "risk_score": round(risk, 1),
            "weight": weight,
            "flagged": risk >= 40.0,
        })

    if used_weight == 0:
        return {"score": None, "level": "Unavailable", "coverage_percent": 0.0, "factors": []}

    score = weighted_score / used_weight
    if score >= RISK_THRESHOLDS["high"]:
        level = "High Risk"
    elif score >= RISK_THRESHOLDS["moderate"]:
        level = "Moderate Risk"
    else:
        level = "Low Risk"
    factors.sort(key=lambda item: item["risk_score"], reverse=True)
    return {
        "score": round(score, 1),
        "level": level,
        "coverage_percent": round(used_weight * 100, 1),
        "factors": factors,
    }


def make_study_plan(
    weak_subjects: list[str], record: dict | pd.Series | None = None
) -> list[dict[str, str]]:
    """Build a practical five-day plan from recorded subject and engagement gaps."""
    record = {} if record is None else record
    subject = weak_subjects[0] if weak_subjects else str(record.get("Subject") or "your priority subject")
    submission_rate = pd.to_numeric(record.get("Assignment_Submission_Rate"), errors="coerce")
    attendance = pd.to_numeric(record.get("Attendance"), errors="coerce")
    quiz = pd.to_numeric(record.get("Quiz_Marks"), errors="coerce")
    internal = pd.to_numeric(record.get("Internal_Marks"), errors="coerce")
    study_hours = pd.to_numeric(record.get("Study_Hours"), errors="coerce")
    backlogs = pd.to_numeric(record.get("Backlogs"), errors="coerce")

    assignment_task = (
        "Complete pending assignments and submit each one before its deadline (60 min)"
        if pd.notna(submission_rate) and submission_rate < 90
        else f"{subject}: complete one assessed practice set (60 min)"
    )
    quiz_task = (
        "Review quiz errors, then retake a short retrieval quiz (45 min)"
        if (pd.notna(quiz) and quiz < 65) or (pd.notna(internal) and internal < 65)
        else f"{subject}: active recall and flashcards (30 min)"
    )
    attendance_task = (
        "Agree an attendance and catch-up plan with faculty"
        if pd.notna(attendance) and attendance < 75
        else f"{subject}: guided problem practice (45 min)"
    )
    workload_task = (
        "Book a faculty check-in and split backlog work into two small tasks"
        if pd.notna(backlogs) and backlogs > 0
        else "Review the week and schedule three focused study blocks"
    )
    if pd.notna(study_hours) and study_hours < 2:
        workload_task = "Schedule three 40-minute focused study blocks and record completed practice"
    return [
        {"day": "Monday", "task": f"{subject}: review core concepts (60 min)"},
        {"day": "Tuesday", "task": assignment_task},
        {"day": "Wednesday", "task": quiz_task},
        {"day": "Thursday", "task": attendance_task},
        {"day": "Friday", "task": workload_task},
    ]
