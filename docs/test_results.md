# Test and Model Verification Results

**Run date:** 11 October 2026  
**Environment:** Project virtual environment; Windows; Python 3.11

## Automated tests

Command: `python -m pytest -q`

Result: **22 passed, 3 warnings**. The warnings are non-blocking SHAP/matplotlib pending-deprecation notices.

Coverage includes raw-data quality checks, feature engineering, leakage exclusions, inference validation, reference-model inference, recommendation triggers, college CSV validation, weighted risk scoring and missing-data coverage, personalized study plans, PDF generation, privacy-safe export, SQLite authentication, academic-record persistence, and risk-history persistence.

## Reference UCI model

The combined Mathematics and Portuguese dataset contains 1,044 course records. A fixed random seed separates 835 training records from 209 held-out test records. Model selection and hyperparameter search use cross-validation on the training portion.

| Task | Selected model | Held-out result | Cross-validation |
| --- | --- | --- | --- |
| Score regression (0-20) | Tuned Random Forest | MAE 2.336; RMSE 3.300; R2 0.268 | RMSE 3.205 |
| Support classification | XGBoost | Accuracy 0.579; macro-F1 0.533; ROC-AUC 0.741 | Macro-F1 0.547 |

These values describe performance on the public UCI dataset only. They do not establish performance on Trekverse or institution data.

## Data-quality checks

The reference dataset run recorded 1,044 rows, no missing cells, no full-row duplicates, no out-of-range final grades, no negative absence counts, and no unexpected categorical values. IQR outlier candidates are recorded in `artifacts/models/metrics.json`; the pipeline does not automatically remove them.

## Institution data status

No institution student records are included in the submission database. CSV and Excel validation, SQLite persistence, student-grouped model training, and institution-model workflows are implemented. Institution performance cannot be reported until representative labeled records are imported and evaluated.
