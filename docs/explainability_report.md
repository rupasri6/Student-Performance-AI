# Model Explainability

## Global explanations

The training pipeline reports permutation importance for regression and classification. It measures how model performance changes when one feature is shuffled. SHAP values are calculated when supported by the selected estimator and available library versions.

## Individual predictions

The prediction views show the most influential features for an individual estimate. When SHAP is unavailable, the application compares the estimate with a reference value for each feature and labels the explanation method accordingly.

## Recommendations

`src/models/recommend.py` maps selected academic and engagement signals to learning resources. Institution study plans use recorded weak subjects to prioritize practice activities.

## Interpretation

Feature importance describes model behavior; it does not establish cause and effect. For example, additional school support may be associated with a lower estimate because support was offered after a student began struggling. Explanations should guide discussion and support, not label students.
