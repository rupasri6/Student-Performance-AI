# Trekverse AI Academy

## Student Performance and Early-Warning System

A Streamlit application for reviewing academic performance, identifying students who may need support, and planning follow-up. The project combines a reference model trained on the UCI Student Performance dataset with an institution-data workflow for authorized administrators, faculty, and students.

## Capabilities

- Admin, Faculty, and Student accounts with role-specific views.
- CSV and Excel import for institution records, with schema and value validation.
- Student profiles, subject performance, semester history, and faculty cohort summaries.
- Reference-model and institution-model prediction workflows, with explanations and model metrics.
- Nine-factor weighted early-warning bands, with missing-input coverage shown.
- Subject-aware score estimates, what-if simulations, data-personalized study plans, and downloadable PDF reports.
- SQLite persistence for users, academic records, predictions, and recommendations.

Risk bands and model outputs are decision-support information. They should be reviewed with the student and do not determine academic outcomes.

## Run on Windows

Open this project folder in VS Code, then run these commands in its integrated PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data.download
python -m src.models.train
streamlit run app/app.py
```

If PowerShell blocks virtual-environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

On first launch, create the administrator account in the app. The administrator can download the college data template, import records, train the institution model, and create Faculty or Student accounts. Student accounts must be linked to an imported `Student_ID`.

## Institution data

Upload CSV or `.xlsx` files from **Admin**. Required columns:

`Student_ID`, `Attendance`, `Internal_Marks`, `Assignment_Marks`, `Quiz_Marks`, `Study_Hours`, `Previous_CGPA`, `Backlogs`, `Final_Marks`.

Optional columns: `Name`, `Department`, `Semester`, `Subject`, `Current_CGPA`, `Assignment_Submission_Rate`, and `Learning_Activity`. `Previous_CGPA` is required for prediction; `Current_CGPA` is displayed when supplied.

Use marks and attendance on a 0-100 scale and CGPA on a 0-10 scale. Institution model training requires at least 12 labeled records from at least 4 students. For a student-specific prediction, the selected student's records are held out from that training run, so the remaining records must still meet this minimum.

The faculty risk summary uses the latest semester and all available attendance, backlogs, internal and assignment marks, study hours, previous CGPA, quiz performance, assignment completion, and learning activity. Risk scores are weighted decision-support rules; model score estimates are reported separately.

## Model evaluation

The UCI pipeline compares regression and classification models, selects models using cross-validation on training data, and reports held-out results in `artifacts/models/metrics.json`. The institution pipeline compares Linear Regression, Random Forest, and Gradient Boosting using grouped validation by student. Review metrics on the **Model Performance** page before interpreting estimates. Performance on the public reference dataset does not establish performance on Trekverse or any institution's data.

The UCI data-quality report records missing values, duplicate rows, range violations, IQR outlier counts, and unexpected category values. Numeric and categorical imputers are fitted inside each model pipeline. The EDA and evaluation artifacts include more than eight charts, including residual and confusion-matrix plots.

## Data and privacy

The UCI files are stored in `data/raw/`. Uploaded institution data and password hashes are stored locally in `database/student_records.sqlite3`. Keep that database and exported reports private. The local SQLite sign-in is intended for a project demonstration; production deployment requires managed authentication, access controls, backups, and institution-approved data handling.

UCI Student Performance dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). Cite Cortez, P. (2008), "Student Performance," UCI Machine Learning Repository, and Cortez & Silva (2008), "Using Data Mining to Predict Secondary School Student Performance," FUBUTEC 2008.

## Project structure

```text
app/              Streamlit entry point, role pages, and shared helpers
src/data/         Dataset loading, validation, and feature preparation
src/models/       Training, evaluation, prediction, and explanations
src/services/     SQLite persistence and account management
database/         Database schema and application database interface
reports/          Student PDF report generation
artifacts/        Trained models, metrics, and EDA figures
tests/            Automated checks
docs/             Proposal, data dictionary, ethics, and project notes
deliverables/     Technical report (PDF and Markdown) and project presentation
```

See [test results](docs/test_results.md) for the latest recorded verification run and [explainability notes](docs/explainability_report.md) for interpretation guidance.
