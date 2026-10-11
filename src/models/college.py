"""Train and serve a college-schema model from persisted institution records."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.config import MODEL_DIR
from src.data.college import assess_student_risk
from src.services.database import students_frame

NUMERIC_FEATURES = [
    "Attendance", "Internal_Marks", "Assignment_Marks", "Quiz_Marks", "Study_Hours",
    "Previous_CGPA", "Backlogs", "Assignment_Submission_Rate", "Learning_Activity",
]
FEATURES = NUMERIC_FEATURES + ["Subject"]
MODEL_PATH = MODEL_DIR / "college_regression.joblib"
METRICS_PATH = MODEL_DIR / "college_metrics.json"


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.reindex(columns=FEATURES).copy()
    for name in NUMERIC_FEATURES:
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    prepared["Subject"] = prepared["Subject"].fillna("Overall").astype(str)
    return prepared


def _build_pipeline(estimator):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Overall"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("subject", categorical, ["Subject"])],
        verbose_feature_names_out=False,
    )
    return make_pipeline(preprocess, estimator)


def train_college_model(frame: pd.DataFrame | None = None) -> dict:
    frame = frame if frame is not None else students_frame()
    frame = frame.dropna(subset=["Final_Marks"])
    if len(frame) < 12 or frame["Student_ID"].nunique() < 4:
        raise ValueError("At least 12 labeled records from 4 or more students are needed to train this model.")
    X = _prepare_features(frame)
    y = pd.to_numeric(frame["Final_Marks"], errors="coerce")
    valid = y.notna()
    X, y, groups = X.loc[valid], y.loc[valid], frame.loc[valid, "Student_ID"].astype(str)
    if len(X) < 12 or groups.nunique() < 4:
        raise ValueError("At least 12 labeled records from 4 or more students are needed.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    scores = {}
    best_name, best_model, best_rmse = None, None, float("inf")
    if groups_train.nunique() < 2:
        raise ValueError("At least two students must remain in the training split for cross-validation.")
    folds = GroupKFold(n_splits=min(5, groups_train.nunique()))
    for name, estimator in candidates.items():
        model = _build_pipeline(estimator)
        cv_rmse = -cross_val_score(model, X_train, y_train, groups=groups_train, cv=folds,
                                   scoring="neg_root_mean_squared_error").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {"mae": float(mean_absolute_error(y_test, pred)),
                   "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                   "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
                   "cv_rmse": float(cv_rmse)}
        scores[name] = metrics
        if cv_rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, cv_rmse
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    summary = {"best_model": best_name, "models": scores, "train_rows": len(X_train), "test_rows": len(X_test),
               "target_scale": "0-100 college marks", "features": FEATURES}
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_college(record: dict) -> dict:
    if not MODEL_PATH.exists():
        train_college_model()
    model = joblib.load(MODEL_PATH)
    features = {key: record.get(key) for key in FEATURES}
    features["Subject"] = record.get("Subject") or "Overall"
    features = _prepare_features(pd.DataFrame([features])).iloc[0].to_dict()
    score = float(np.clip(model.predict(pd.DataFrame([features]))[0], 0, 100))
    risk = assess_student_risk(record)
    explanation, method = explain_college(model, features)
    return {"predicted_score": round(score, 2), "risk_level": risk["level"],
            "risk_score": risk["score"], "risk_coverage_percent": risk["coverage_percent"],
            "risk_factors": risk["factors"],
            "explanation": explanation, "explanation_method": method}


def explain_college(model, features: dict) -> tuple[list[dict], str]:
    row = _prepare_features(pd.DataFrame([features]))
    try:
        import shap
        estimator = model[-1]
        transformed = model[:-1].transform(row)
        names = list(model[:-1].get_feature_names_out())
        values = np.asarray(shap.TreeExplainer(estimator).shap_values(transformed)).reshape(-1)
        order = np.argsort(np.abs(values))[::-1][:7]
        return ([{"feature": names[i], "shap": float(values[i])} for i in order], "SHAP")
    except Exception:
        # Model replacement deltas keep explanations available for non-tree models or SHAP incompatibility.
        reference = students_frame()
        contributions = []
        current = float(model.predict(row)[0])
        for feature in FEATURES:
            values = reference.get(feature, pd.Series(dtype=object)).dropna()
            if feature == "Subject":
                typical = str(values.mode().iloc[0]) if not values.empty else "Overall"
            else:
                values = pd.to_numeric(values, errors="coerce").dropna()
                typical = float(values.median()) if not values.empty else 0.0
            altered = row.copy()
            altered[feature] = typical
            contributions.append({"feature": feature, "delta_if_typical": current - float(model.predict(altered)[0])})
        contributions.sort(key=lambda item: abs(item["delta_if_typical"]), reverse=True)
        return contributions[:7], "local feature replacement"


def segment_college_students(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cluster student engagement features without using final marks as an input."""
    frame = frame if frame is not None else students_frame()
    frame = frame.sort_values("Semester", ascending=False).drop_duplicates("Student_ID").copy()
    if len(frame) < 3:
        return pd.DataFrame(columns=["Student_ID", "Engagement_Segment"])
    X = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    X = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=min(3, len(frame)), n_init=10, random_state=42).fit_predict(X)
    return pd.DataFrame({"Student_ID": frame["Student_ID"].astype(str).to_numpy(),
                         "Engagement_Segment": [f"Segment {label + 1}" for label in labels]})
