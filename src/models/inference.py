"""Load saved pipelines and run a single-student prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.data.features import engineer_features, risk_from_score
from src.data.validate import validate_inference_payload
from src.models.explain import local_shap_contributions, simple_local_contributions
from src.models.recommend import recommend_for_student

_CACHE: dict[str, Any] = {}


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    if "reg" in _CACHE:
        return _CACHE
    reg_path = model_dir / "regression_pipeline.joblib"
    if not reg_path.exists():
        raise FileNotFoundError(
            "Trained model not found. From the project folder run: python -m src.models.train"
        )
    _CACHE["reg"] = joblib.load(reg_path)
    _CACHE["clf"] = joblib.load(model_dir / "classification_pipeline.joblib")
    _CACHE["kmeans"] = joblib.load(model_dir / "kmeans.joblib")
    _CACHE["cluster_pre"] = joblib.load(model_dir / "cluster_preprocessor.joblib")
    bg_path = model_dir / "explanation_background.csv"
    _CACHE["background"] = pd.read_csv(bg_path) if bg_path.exists() else None
    return _CACHE


def predict_student(payload: dict[str, Any], artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    errors = validate_inference_payload(payload)
    if errors:
        return {"ok": False, "errors": errors}

    arts = artifacts or load_artifacts()
    row = pd.DataFrame([payload])
    row = engineer_features(row)
    pred_score = float(arts["reg"].predict(row)[0])
    pred_score = max(0.0, min(20.0, pred_score))
    pred_class = str(arts["clf"].predict(row)[0])
    proba = arts["clf"].predict_proba(row)[0]
    classes = list(arts["clf"].classes_)
    cluster = int(arts["kmeans"].predict(arts["cluster_pre"].transform(row))[0])
    background = arts.get("background")
    if background is None or background.empty:
        background = row
    needed = [c for c in background.columns if c in row.columns]
    contrib = local_shap_contributions(arts["reg"], row[needed])
    explanation_method = "SHAP" if contrib else "local feature replacement (SHAP fallback)"
    contrib = contrib or simple_local_contributions(arts["reg"], row[needed], background[needed])
    recs = recommend_for_student(payload, contrib[0]["top"] if contrib else None)
    return {
        "ok": True,
        "predicted_score": round(pred_score, 2),
        "score_band_from_regression": risk_from_score(pred_score),
        "predicted_support": pred_class,
        "class_probabilities": {str(c): round(float(p), 3) for c, p in zip(classes, proba)},
        "engagement_cluster": cluster,
        "explanation": contrib[0] if contrib else {},
        "explanation_method": explanation_method,
        "recommendations": recs,
        "disclaimer": (
            "This is a decision-support estimate, not a judgement of ability or future. "
            "It can be wrong. A teacher should review the explanation before acting."
        ),
    }
