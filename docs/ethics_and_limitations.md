# Ethics, limitations, bias, and future work

## Intended use

The application is a teacher- and student-facing decision-support tool. It may help staff notice students who might need tutoring, attendance mentoring, or wellbeing referral. It must not be used alone for streaming, expulsion, scholarship denial, or any irreversible academic judgement.

## What the system must not claim

- It does not measure intelligence, character, or future success.
- Test metrics will not be 100%. Real MAE/RMSE/R² and macro-F1 are stored in `artifacts/models/metrics.json`.
- A high-support label means “consider extra help,” not “this student will fail.”

## Privacy

- The public UCI files have no names or IDs. College imports can include identifying details and are stored in the local SQLite database.
- User passwords are stored as salted PBKDF2 hashes. Restrict access to the host and database file; local SQLite is a demo deployment, not a hardened production service.
- Exported PDF reports contain academic details. Share them only with authorized people.

## Bias and fairness risks

- Historical failures and extra-support flags can **re-encode disadvantage**: students who already received help may look “risky” forever.
- Parent education and job are socioeconomic proxies.
- Alcohol, romantic relationship, and family structure fields can invite stereotyped interpretations. Treat them as wellbeing context, not moral scores.
- Sex is excluded from predictors.
- Only two Portuguese schools from 2008 appear in the data. Transfer to another country or decade is not guaranteed.

## Leakage

G1 and G2 are excluded. If they were included, error would collapse and the product would stop being an early-warning system.

## Uncertainty and wrong predictions

- Residual plots and RMSE quantify typical score error (often around 2 points on a 0–20 scale; check the live metrics file).
- Class probabilities are shown so staff can see when the model is unsure.
- Missing `studytime` is tested as a robustness check during training.
- Incorrect predictions should be overridden by teachers; log disagreements if this were production.

## Model drift after deployment

Each term, compare:

1. Input distributions (absences, failures, studytime).
2. Predicted vs actual G3 once grades exist.
3. False-negative rate on the high-support group (missed students).

Retrain if those checks degrade. Do not silently reuse a 2008 model on a 2026 LMS dump without a new validation set.

## Future improvements

- Deep learning tabular models (TabNet) after a larger multi-school dataset exists.
- An LLM assistant that turns the explanation JSON into a parent-friendly letter, with a teacher in the loop.
- Learning-management clickstream instead of self-reported studytime.
- Fairness audits (equalized odds by school or address type) on a dataset that is large enough to support them.
