# Technical Report

## Executive summary

Trekverse AI Academy is a role-based student performance and early-warning application. Admin, Faculty, and Student workflows support academic record management, subject and semester review, performance estimates, transparent risk scoring, explanations, study plans, what-if simulations, and PDF reports.

## Data and preparation

The reference model uses the UCI Student Performance Mathematics and Portuguese files, combined into 1,044 course records. The pipeline checks missing cells, duplicates, numeric range violations, IQR outlier candidates, and inconsistent categories. Numeric and categorical imputation is fitted inside each training pipeline. Student sex and prior-period grades G1/G2 are excluded from model features to reduce sensitive-attribute use and target leakage.

Administrators can import institution records from CSV or Excel after schema and value validation. Required input includes Student_ID, attendance, internal, assignment and quiz marks, study hours, previous CGPA, backlogs, and final marks. Assignment submission rate and learning activity are optional. The institution model requires at least 12 labeled records from at least four students. Institution performance must be assessed on representative local records before operational use.

## Modeling and evaluation

The reference pipelines compare a mean baseline, linear regression, decision tree, random forest, gradient boosting, support vector regression, XGBoost, LightGBM, and CatBoost. A tuned Random Forest is selected for regression using training-fold cross-validation RMSE. XGBoost is selected for support classification using training-fold macro-F1. The held-out test split is reserved for final evaluation. K-Means uses academic and engagement inputs without the final-grade target.

The institution score model compares Linear Regression, Random Forest, and Gradient Boosting using student-grouped splits and cross-validation. Subject is an input feature. The application reports model metrics, global permutation importance, SHAP or local feature-replacement explanations, residual analysis, and missing-input robustness results.

## Early warning and student support

Institution risk assessment computes a weighted 0-100 score from up to nine available signals: attendance, backlogs, internal marks, assignment marks, study hours, previous CGPA, quiz marks, assignment submission rate, and learning activity. A score below 30 is Low Risk, 30 to below 60 is Moderate Risk, and 60 or above is High Risk. Missing indicators are excluded from the calculation; the dashboard reports data coverage and flagged factors. These thresholds are transparent decision-support rules, separate from predicted marks.

Student and Faculty dashboards show profile and academic history, semester trends, subject performance, risk, and support actions. Faculty can review cohort averages, attendance, at-risk counts, top performers, subject results, comparisons, and an intervention queue. The five-day study plan uses recorded weak subjects and engagement gaps. What-if controls simulate changes to attendance, study hours, and assignment completion. The interface labels the output as a model simulation, not a guaranteed real-world result.

## Reference results

A fixed-seed split contains 835 training records and 209 held-out records. The tuned Random Forest regressor achieved held-out MAE 2.336, RMSE 3.300, and R2 0.268; training-fold CV RMSE was 3.205. XGBoost support classification achieved accuracy 0.579, macro-F1 0.533, and ROC-AUC 0.741; training-fold CV macro-F1 was 0.547. Full candidate metrics and the confusion matrix are stored in `artifacts/models/metrics.json`.

## Application and persistence

The Streamlit interface supports role-specific workflows. SQLite stores users, student profiles, subjects, attendance, marks, assignments, predictions, and recommendations. Passwords use salted PBKDF2 hashes. Faculty can review cohort and student trends, update records, compare students, and prepare reports. Students can review their own linked records and support tools. Current CGPA is displayed when institution data supplies it; previous CGPA remains a model input. Prediction exports omit names, IDs, and raw personal data.

## Limitations and responsible use

The public reference data represents two Portuguese schools and does not establish performance on Trekverse institution records. Regression error remains material on a 0-20 scale, and the classifier makes errors across support groups. Predictions and risk bands should prompt human review; they must not determine academic outcomes. The local SQLite authentication and direct service layer support a local project demonstration. The application does not provide a separate REST API or PostgreSQL/MySQL deployment, and its login is not production identity management.

See [test results](test_results.md), [data dictionary](data_dictionary.md), [explainability notes](explainability_report.md), and [ethics and limitations](ethics_and_limitations.md) for supporting details.
