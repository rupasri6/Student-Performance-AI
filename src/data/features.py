"""Derived academic and engagement features. No target columns are used here."""

from __future__ import annotations

import pandas as pd

from src.config import ABSENCE_CAP, TARGET_CLASS, TARGET_REGRESSION


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["parent_edu_avg"] = (out["Medu"] + out["Fedu"]) / 2.0
    capped_absences = out["absences"].clip(lower=0, upper=ABSENCE_CAP)
    out["attendance_proxy"] = 1.0 - (capped_absences / ABSENCE_CAP)
    out["alcohol_index"] = (out["Dalc"] + out["Walc"]) / 2.0
    out["study_vs_travel"] = out["studytime"] / out["traveltime"].clip(lower=1)
    out["support_count"] = (
        _yes_no_to_int(out["schoolsup"])
        + _yes_no_to_int(out["famsup"])
        + _yes_no_to_int(out["paid"])
    )
    out["engagement_score"] = (
        out["studytime"]
        + _yes_no_to_int(out["activities"])
        + _yes_no_to_int(out["internet"])
        + _yes_no_to_int(out["higher"])
    )
    out["social_load"] = (out["goout"] + out["freetime"] + out["romantic"].map({"yes": 1, "no": 0}).fillna(0)) / 3.0
    return out


def add_support_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Map final grade to support bands. Labels are derived only from G3."""
    out = frame.copy()
    out[TARGET_CLASS] = out[TARGET_REGRESSION].apply(risk_from_score)
    return out


def risk_from_score(score: float) -> str:
    if score < 10:
        return "high_support"
    if score < 14:
        return "medium_support"
    return "low_support"
