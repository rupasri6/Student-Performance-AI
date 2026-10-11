"""SQLite persistence for accounts and academic records."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "database" / "student_records.sqlite3"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "database" / "schema.sql"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    assignment_columns = {row[1] for row in conn.execute("PRAGMA table_info(assignments)")}
    for column, sql_type in (("marks", "REAL"), ("submission_rate", "REAL"), ("learning_activity", "REAL")):
        if column not in assignment_columns:
            conn.execute(f"ALTER TABLE assignments ADD COLUMN {column} {sql_type}")
    if "marks" not in assignment_columns and "submitted" in assignment_columns:
        conn.execute("UPDATE assignments SET marks=submitted WHERE total=100")
    student_columns = {row[1] for row in conn.execute("PRAGMA table_info(students)")}
    if "previous_cgpa" not in student_columns:
        conn.execute("ALTER TABLE students ADD COLUMN previous_cgpa REAL")
        conn.execute("UPDATE students SET previous_cgpa=current_cgpa")
    prediction_columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "risk_score" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN risk_score REAL")
    if "risk_factors_json" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN risk_factors_json TEXT NOT NULL DEFAULT '[]'")
    return conn


@contextmanager
def connection():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return f"{salt.hex()}:{digest.hex()}"


def _check_password(password: str, encoded: str) -> bool:
    salt_hex, expected = encoded.split(":", 1)
    return hmac.compare_digest(_password_hash(password, bytes.fromhex(salt_hex)).split(":", 1)[1], expected)


def account_count() -> int:
    with connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def create_user(username: str, password: str, role: str, student_id: str | None = None) -> None:
    if not username.strip():
        raise ValueError("Username cannot be empty.")
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters.")
    if role not in {"ADMIN", "FACULTY", "STUDENT"}:
        raise ValueError("Unknown role.")
    with connection() as conn:
        conn.execute(
            "INSERT INTO users(username,password_hash,role,student_id) VALUES(?,?,?,?)",
            (username.strip(), _password_hash(password), role, student_id.strip() if student_id else None),
        )


def authenticate(username: str, password: str) -> dict | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    if row and _check_password(password, row["password_hash"]):
        return {"username": row["username"], "role": row["role"], "student_id": row["student_id"]}
    return None


def list_users() -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql_query("SELECT id, username, role, student_id, created_at FROM users ORDER BY role, username", conn)


def import_college_records(frame: pd.DataFrame) -> int:
    """Upsert college roster and semester marks into normalized SQLite tables."""
    with connection() as conn:
        for row in frame.to_dict(orient="records"):
            sid = str(row["Student_ID"]).strip()
            name = str(row.get("Name") if pd.notna(row.get("Name")) else sid).strip()
            dept = str(row.get("Department", "")).strip()
            sem = int(row.get("Semester", 1))
            cgpa = row.get("Previous_CGPA")
            current_cgpa = row.get("Current_CGPA")
            attendance_pct = float(row["Attendance"])
            study_hours = row.get("Study_Hours")
            backlogs = int(row.get("Backlogs", 0))
            conn.execute(
                """INSERT INTO students(student_id,name,department,semester,current_cgpa,previous_cgpa,attendance,study_hours,backlogs)
                   VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(student_id) DO UPDATE SET
                   name=excluded.name,department=excluded.department,
                   current_cgpa=CASE WHEN excluded.semester>=students.semester THEN excluded.current_cgpa ELSE students.current_cgpa END,
                   previous_cgpa=CASE WHEN excluded.semester>=students.semester THEN excluded.previous_cgpa ELSE students.previous_cgpa END,
                   attendance=CASE WHEN excluded.semester>=students.semester THEN excluded.attendance ELSE students.attendance END,
                   study_hours=CASE WHEN excluded.semester>=students.semester THEN excluded.study_hours ELSE students.study_hours END,
                   backlogs=CASE WHEN excluded.semester>=students.semester THEN excluded.backlogs ELSE students.backlogs END,
                   semester=MAX(students.semester,excluded.semester),updated_at=CURRENT_TIMESTAMP""",
                (sid, name, dept, sem, _float_or_none(current_cgpa), _float_or_none(cgpa), attendance_pct, _float_or_none(study_hours), backlogs),
            )
            conn.execute("INSERT INTO attendance(student_id,semester,percentage) VALUES(?,?,?)", (sid, sem, attendance_pct))
            subject = str(row.get("Subject", "Overall")).strip() or "Overall"
            conn.execute("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (subject,))
            subject_id = conn.execute("SELECT id FROM subjects WHERE name=?", (subject,)).fetchone()[0]
            conn.execute("DELETE FROM marks WHERE student_id=? AND subject_id=? AND semester=?", (sid, subject_id, sem))
            conn.execute(
                "INSERT INTO marks(student_id,subject_id,semester,internal,quiz,final) VALUES(?,?,?,?,?,?)",
                (sid, subject_id, sem, _float_or_none(row.get("Internal_Marks")), _float_or_none(row.get("Quiz_Marks")), _float_or_none(row.get("Final_Marks"))),
            )
            if "Assignment_Marks" in row:
                conn.execute("DELETE FROM assignments WHERE student_id=? AND semester=?", (sid, sem))
                conn.execute("INSERT INTO assignments(student_id,semester,submitted,total,marks,submission_rate,learning_activity) VALUES(?,?,?,?,?,?,?)",
                             (sid, sem, 0, 0, float(row["Assignment_Marks"]),
                              _float_or_none(row.get("Assignment_Submission_Rate")),
                              _float_or_none(row.get("Learning_Activity"))))
    return len(frame)


def _float_or_none(value):
    if value is None or pd.isna(value) or value == "":
        return None
    return float(value)


def students_frame() -> pd.DataFrame:
    sql = """SELECT s.student_id AS Student_ID,s.name AS Name,s.department AS Department,
      COALESCE(m.semester,s.semester) AS Semester,s.current_cgpa AS Current_CGPA,s.previous_cgpa AS Previous_CGPA,
      COALESCE((SELECT at.percentage FROM attendance at WHERE at.student_id=s.student_id
        AND at.semester=COALESCE(m.semester,s.semester) ORDER BY at.id DESC LIMIT 1),s.attendance) AS Attendance,
      s.study_hours AS Study_Hours,s.backlogs AS Backlogs,sub.name AS Subject,
      m.internal AS Internal_Marks,m.quiz AS Quiz_Marks,m.final AS Final_Marks,
      a.marks AS Assignment_Marks,a.submission_rate AS Assignment_Submission_Rate,
      a.learning_activity AS Learning_Activity
      FROM students s LEFT JOIN marks m ON m.student_id=s.student_id
      LEFT JOIN subjects sub ON sub.id=m.subject_id
      LEFT JOIN assignments a ON a.id=(SELECT MAX(a2.id) FROM assignments a2
        WHERE a2.student_id=s.student_id AND a2.semester=COALESCE(m.semester,s.semester))
      ORDER BY s.student_id,COALESCE(m.semester,s.semester) DESC,sub.name"""
    with connection() as conn:
        return pd.read_sql_query(sql, conn)


def student_record(student_id: str) -> dict | None:
    frame = students_frame()
    match = frame[frame["Student_ID"].astype(str) == str(student_id)]
    return None if match.empty else match.iloc[0].to_dict()


def student_history(student_id: str) -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql_query(
            """SELECT p.predicted_score,p.risk_level,p.risk_score,p.risk_factors_json,p.created_at FROM predictions p
               WHERE p.student_id=? ORDER BY p.created_at DESC""", conn, params=(student_id,)
        )


def update_academic_record(student_id: str, subject: str, semester: int, values: dict) -> None:
    with connection() as conn:
        conn.execute("UPDATE students SET semester=?,attendance=?,current_cgpa=?,previous_cgpa=?,study_hours=?,backlogs=?,updated_at=CURRENT_TIMESTAMP WHERE student_id=?",
                     (int(semester), float(values["Attendance"]), _float_or_none(values.get("Current_CGPA")),
                      _float_or_none(values.get("Previous_CGPA")),
                      _float_or_none(values.get("Study_Hours")), int(values.get("Backlogs", 0)), student_id))
        conn.execute("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (subject,))
        subject_id = conn.execute("SELECT id FROM subjects WHERE name=?", (subject,)).fetchone()[0]
        conn.execute("DELETE FROM marks WHERE student_id=? AND subject_id=? AND semester=?", (student_id, subject_id, int(semester)))
        conn.execute("INSERT INTO marks(student_id,subject_id,semester,internal,quiz,final) VALUES(?,?,?,?,?,?)",
          (student_id, subject_id, int(semester), values.get("Internal_Marks"), values.get("Quiz_Marks"), values.get("Final_Marks")))
        conn.execute("INSERT INTO attendance(student_id,semester,percentage) VALUES(?,?,?)",
                     (student_id, int(semester), float(values["Attendance"])))
        conn.execute("INSERT INTO assignments(student_id,semester,submitted,total,marks,submission_rate,learning_activity) VALUES(?,?,?,?,?,?,?)",
                     (student_id, int(semester), 0, 0, float(values.get("Assignment_Marks", 0)),
                      _float_or_none(values.get("Assignment_Submission_Rate")), _float_or_none(values.get("Learning_Activity"))))


def save_prediction(student_id: str, result: dict) -> None:
    explanation = result.get("explanation", [])
    if isinstance(explanation, dict):
        explanation = explanation.get("top", [])
    with connection() as conn:
        cur = conn.execute(
            """INSERT INTO predictions(
                   student_id,predicted_score,risk_level,risk_score,risk_factors_json,explanation_json
               ) VALUES(?,?,?,?,?,?)""",
            (student_id, result["predicted_score"], result["risk_level"], result.get("risk_score"),
             json.dumps(result.get("risk_factors", [])), json.dumps(explanation)),
        )
        pid = cur.lastrowid
        conn.executemany("INSERT INTO recommendations(prediction_id,recommendation) VALUES(?,?)",
                         [(pid, r.get("action", r.get("task", "Review with faculty"))) for r in result.get("recommendations", [])])
