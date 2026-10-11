import pandas as pd
import pytest

from src.data.features import engineer_features, risk_from_score
from src.data.load import load_raw_dataset
from src.data.preprocess import prepare_modelling_frame, split_xy
from src.data.validate import validate_inference_payload, validate_raw_frame


def test_load_has_expected_columns():
    frame = load_raw_dataset()
    assert "G3" in frame.columns
    assert "subject" in frame.columns
    assert set(frame["subject"].unique()) == {"math", "portuguese"}
    assert len(frame) > 800


def test_validate_raw_ok():
    report = validate_raw_frame(load_raw_dataset())
    assert report["n_rows"] > 0
    assert report["g3_out_of_range"] == 0
    assert "numeric_outliers_iqr" in report
    assert "inconsistent_categories" in report
    assert report["full_row_duplicates"] == 0


def test_validate_raw_rejects_bad_grade():
    frame = load_raw_dataset().head(3).copy()
    frame.loc[frame.index[0], "G3"] = 99
    with pytest.raises(ValueError):
        validate_raw_frame(frame)


def test_validate_raw_reports_category_inconsistency():
    frame = load_raw_dataset().head(5).copy()
    frame.loc[frame.index[0], "school"] = "UNKNOWN"
    report = validate_raw_frame(frame)
    assert report["inconsistent_categories"]["school"] == ["UNKNOWN"]


def test_engineer_features_adds_columns():
    frame = engineer_features(load_raw_dataset().head(5))
    for col in ["parent_edu_avg", "attendance_proxy", "engagement_score", "support_count"]:
        assert col in frame.columns
    assert frame["attendance_proxy"].between(0, 1).all()


def test_risk_bands():
    assert risk_from_score(8) == "high_support"
    assert risk_from_score(12) == "medium_support"
    assert risk_from_score(16) == "low_support"


def test_no_leaky_or_sex_features_in_xy():
    prepared = prepare_modelling_frame(load_raw_dataset().head(20))
    X, y_reg, y_cls = split_xy(prepared)
    assert "G1" not in X.columns
    assert "G2" not in X.columns
    assert "sex" not in X.columns
    assert "G3" not in X.columns
    assert len(y_reg) == len(X)


def test_inference_payload_missing_field():
    errors = validate_inference_payload({"age": 17})
    assert errors


def test_inference_payload_out_of_range():
    from src.app.streamlit_app import DEFAULTS

    payload = dict(DEFAULTS)
    payload["absences"] = -3
    errors = validate_inference_payload(payload)
    assert any("absences" in e for e in errors)
