# Project Overview

## Student Performance and Early-Warning System

**Organization:** Trekverse
**Application:** Trekverse AI Academy

## Purpose

Provide administrators, faculty, and students with a clear view of academic progress and timely support needs. The system combines student records, performance estimates, transparent risk rules, and practical study recommendations in one role-based application.

## Functional scope

- **Admin:** manage accounts, validate and import institution data, and train the institution model.
- **Faculty:** review cohort performance, attendance, risk bands, semester trends, subject results, student comparisons, and intervention suggestions; update academic records and prepare student reports.
- **Student:** review their own profile, performance history, subject results, risk band, study plan, model estimate, and report.
- **Prediction and support:** compare candidate models, explain estimates, run what-if simulations, and provide study recommendations.
- **Reporting:** evaluate reference models using held-out metrics and cross-validation; export an academic report as PDF.

## Data and modeling

The UCI Student Performance dataset provides a reproducible reference dataset. Institution records can be imported from CSV or Excel using the template available in the application. The institution model uses attendance, internal and assignment marks, quiz results, study hours, previous CGPA, backlogs, assignment submission, and learning activity when available. Current CGPA can also be supplied for the student profile and report.

The project reports measured performance rather than claiming certainty. Institution risk bands use a weighted score from up to nine available academic and engagement factors: attendance, backlogs, internal and assignment marks, study hours, previous CGPA, quiz marks, assignment submission, and learning activity. Missing factors reduce reported data coverage rather than being treated as safe. Risk bands prompt human review and support; they do not make academic decisions.

## Technical components

- **Interface:** Streamlit
- **Data processing and models:** pandas, scikit-learn, XGBoost, LightGBM, CatBoost, and SHAP where available
- **Persistence:** SQLite with separate tables for users, students, subjects, attendance, marks, assignments, predictions, and recommendations
- **Reports:** PDF generation with ReportLab

## Limitations

The local SQLite authentication and storage support a project demonstration. A production deployment requires institution-approved authentication, hosting, access control, backups, and privacy procedures. Model quality on the UCI data does not demonstrate quality on institution records; local data must be representative and evaluated before operational use.
