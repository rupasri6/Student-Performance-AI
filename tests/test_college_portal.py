import pandas as pd

from src.data.college import assess_student_risk, make_study_plan, risk_level, validate_college_frame
from src.services import database as db
from src.app.streamlit_app import _build_college_pdf


def _college_record():
    return {
        "Student_ID": "STU1001",
        "Name": "Sample Student",
        "Department": "Computer Science",
        "Semester": 1,
        "Subject": "Mathematics",
        "Attendance": 82,
        "Internal_Marks": 70,
        "Assignment_Marks": 80,
        "Quiz_Marks": 75,
        "Study_Hours": 2.5,
        "Current_CGPA": 7.4,
        "Previous_CGPA": 7.2,
        "Backlogs": 0,
        "Final_Marks": 74,
        "Assignment_Submission_Rate": 90,
        "Learning_Activity": 6,
    }


def test_college_upload_validation_and_risk_bands():
    frame = pd.DataFrame([_college_record()])
    assert validate_college_frame(frame) == []
    assert risk_level(45, 90, 0) == "High Risk"
    assert risk_level(60, 80, 0) == "Moderate Risk"
    assert risk_level(75, 85, 0) == "Low Risk"
    assert make_study_plan(["Mathematics"])[0]["task"].startswith("Mathematics:")


def test_college_upload_rejects_out_of_range_values():
    record = _college_record()
    record["Attendance"] = 110
    assert any("Attendance" in issue for issue in validate_college_frame(pd.DataFrame([record])))


def test_early_warning_uses_all_nine_factors_and_reports_coverage():
    strong_profile = {
        "Attendance": 95, "Backlogs": 0, "Internal_Marks": 88, "Assignment_Marks": 90,
        "Study_Hours": 4, "Previous_CGPA": 8.5, "Quiz_Marks": 86,
        "Assignment_Submission_Rate": 95, "Learning_Activity": 8,
    }
    strong = assess_student_risk(strong_profile)
    assert strong["level"] == "Low Risk"
    assert len(strong["factors"]) == 9
    assert strong["coverage_percent"] == 100

    weak = assess_student_risk({key: 0 for key in strong_profile})
    assert weak["level"] == "High Risk"
    assert weak["score"] > strong["score"]

    partial = assess_student_risk({"Attendance": 100})
    assert partial["level"] == "Low Risk"
    assert partial["coverage_percent"] == 18


def test_study_plan_responds_to_recorded_gaps():
    plan = make_study_plan(
        ["Mathematics"],
        {"Assignment_Submission_Rate": 50, "Attendance": 60, "Study_Hours": 1, "Backlogs": 2},
    )
    assert plan[0]["task"].startswith("Mathematics:")
    assert "pending assignments" in plan[1]["task"]
    assert "attendance" in plan[3]["task"]
    assert "focused study blocks" in plan[4]["task"]


def test_college_validation_checks_learning_activity_scale():
    record = _college_record()
    record["Learning_Activity"] = 11
    assert any("Learning_Activity" in issue for issue in validate_college_frame(pd.DataFrame([record])))


def test_academic_report_is_a_pdf():
    report = _build_college_pdf(
        _college_record(),
        {"predicted_score": 76.0, "risk_level": "Low Risk", "explanation": []},
        pd.DataFrame([{"Subject": "Mathematics", "Final_Marks": 74}]),
        [],
        ["Mathematics"],
        "Review progress next month.",
    )
    assert report.startswith(b"%PDF-")


def test_database_account_and_academic_record_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "student_records.sqlite3")
    db.create_user("faculty1", "temporary-password", "FACULTY")
    assert db.authenticate("faculty1", "temporary-password")["role"] == "FACULTY"
    assert db.authenticate("faculty1", "incorrect-password") is None

    assert db.import_college_records(pd.DataFrame([_college_record()])) == 1
    frame = db.students_frame()
    assert len(frame) == 1
    assert frame.iloc[0]["Student_ID"] == "STU1001"
    assert frame.iloc[0]["Final_Marks"] == 74
    assert frame.iloc[0]["Current_CGPA"] == 7.4
    assert frame.iloc[0]["Previous_CGPA"] == 7.2
    db.save_prediction("STU1001", {
        "predicted_score": 78.0,
        "risk_level": "Low Risk",
        "risk_score": 12.5,
        "risk_factors": [{"factor": "Attendance", "risk_score": 18.0}],
        "explanation": [],
        "recommendations": [{"day": "Monday", "task": "Review Mathematics"}],
    })
    history = db.student_history("STU1001")
    assert history.iloc[0]["risk_score"] == 12.5
    assert "Attendance" in history.iloc[0]["risk_factors_json"]
