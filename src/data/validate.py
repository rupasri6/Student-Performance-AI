"""Quality checks for educational records. Raises ValueError on hard failures."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import TARGET_REGRESSION


REQUIRED_COLUMNS = [
    "school",
    "age",
    "studytime",
    "failures",
    "absences",
    "G3",
    "subject",
]


def validate_raw_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    numeric_outliers = {}
    for column in [name for name in INFERENCE_RANGES if name in frame.columns] + [TARGET_REGRESSION]:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            numeric_outliers[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        numeric_outliers[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr else 0

    inconsistent_categories = {}
    for column, allowed in ALLOWED.items():
        if column in frame.columns:
            unexpected = sorted(set(frame[column].dropna().astype(str)) - allowed)
            if unexpected:
                inconsistent_categories[column] = unexpected

    report = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "full_row_duplicates": int(frame.duplicated().sum()),
        "g3_out_of_range": int(((frame[TARGET_REGRESSION] < 0) | (frame[TARGET_REGRESSION] > 20)).sum()),
        "negative_absences": int((frame["absences"] < 0).sum()) if "absences" in frame.columns else 0,
        "numeric_outliers_iqr": numeric_outliers,
        "inconsistent_categories": inconsistent_categories,
        "missing_value_treatment": "Median imputation for numeric features; most-frequent imputation for categorical features.",
    }
    if report["g3_out_of_range"]:
        raise ValueError("Found final grades outside the 0-20 UCI scale.")
    if report["negative_absences"]:
        raise ValueError("Found negative absence counts.")
    return report


INFERENCE_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
}

YES_NO = {"yes", "no"}
ALLOWED = {
    "school": {"GP", "MS"},
    "address": {"U", "R"},
    "famsize": {"LE3", "GT3"},
    "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": YES_NO,
    "famsup": YES_NO,
    "paid": YES_NO,
    "activities": YES_NO,
    "nursery": YES_NO,
    "higher": YES_NO,
    "internet": YES_NO,
    "romantic": YES_NO,
    "subject": {"math", "portuguese"},
}


def validate_inference_payload(payload: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors. Empty list means the payload is usable."""
    errors: list[str] = []
    for field, (low, high) in INFERENCE_RANGES.items():
        if field not in payload or payload[field] is None or payload[field] == "":
            errors.append(f"{field} is required.")
            continue
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue
        if value < low or value > high:
            errors.append(f"{field} must be between {low} and {high}.")

    for field, allowed in ALLOWED.items():
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"{field} is required.")
        elif str(payload[field]) not in allowed:
            errors.append(f"{field} must be one of: {sorted(allowed)}.")
    return errors
