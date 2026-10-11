import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.app.streamlit_app import DEFAULTS
from src.data.features import engineer_features, risk_from_score
from src.data.preprocess import build_preprocessor
from src.models.inference import predict_student
from src.models.recommend import recommend_for_student


def _tiny_training_frame(n=40):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        row = dict(DEFAULTS)
        row["studytime"] = int(rng.integers(1, 5))
        row["failures"] = int(rng.integers(0, 3))
        row["absences"] = int(rng.integers(0, 20))
        row["age"] = int(rng.integers(15, 20))
        row["G3"] = max(0, min(20, 14 - 2 * row["failures"] + row["studytime"] - row["absences"] * 0.1))
        rows.append(row)
    return pd.DataFrame(rows)


def test_predict_student_with_injected_artifacts():
    frame = _tiny_training_frame()
    X = engineer_features(frame)
    y_reg = frame["G3"]
    y_cls = y_reg.apply(risk_from_score)
    pre = build_preprocessor()
    reg = Pipeline([("preprocess", pre), ("model", RandomForestRegressor(n_estimators=20, random_state=0))])
    clf = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=20, random_state=0)),
        ]
    )
    # clustering stubs
    from sklearn.cluster import KMeans

    cluster_pre = build_preprocessor()
    Xt = cluster_pre.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(Xt)
    reg.fit(X, y_reg)
    clf.fit(X, y_cls)
    artifacts = {
        "reg": reg,
        "clf": clf,
        "kmeans": kmeans,
        "cluster_pre": cluster_pre,
        "background": X.head(10),
    }
    result = predict_student(dict(DEFAULTS), artifacts=artifacts)
    assert result["ok"] is True
    assert 0 <= result["predicted_score"] <= 20
    assert result["predicted_support"] in {"high_support", "medium_support", "low_support"}
    assert "recommendations" in result


def test_predict_student_rejects_invalid():
    bad = dict(DEFAULTS)
    bad["age"] = 9
    result = predict_student(bad, artifacts={})
    assert result["ok"] is False
    assert result["errors"]


def test_recommendations_for_high_risk_flags():
    payload = dict(DEFAULTS)
    payload["failures"] = 3
    payload["absences"] = 20
    payload["studytime"] = 1
    recs = recommend_for_student(payload)
    gaps = " ".join(r["gap"].lower() for r in recs)
    assert "fail" in gaps
    assert "absence" in gaps
