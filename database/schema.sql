PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('ADMIN', 'FACULTY', 'STUDENT')),
    student_id TEXT UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT '',
    semester INTEGER NOT NULL DEFAULT 1,
    current_cgpa REAL,
    previous_cgpa REAL,
    attendance REAL,
    study_hours REAL,
    backlogs INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    semester INTEGER NOT NULL,
    percentage REAL NOT NULL CHECK (percentage BETWEEN 0 AND 100),
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS marks (
    id INTEGER PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    semester INTEGER NOT NULL,
    internal REAL,
    quiz REAL,
    final REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    semester INTEGER NOT NULL,
    submitted REAL NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    marks REAL,
    submission_rate REAL,
    learning_activity REAL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    predicted_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score REAL,
    risk_factors_json TEXT NOT NULL DEFAULT '[]',
    explanation_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
