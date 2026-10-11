import json

from src.app.streamlit_app import DEFAULTS, _prediction_summary, pages
from src.data.validate import validate_inference_payload


def test_default_form_payload_is_valid():
    assert validate_inference_payload(DEFAULTS) == []


def test_app_pages_registered():
    assert set(pages) >= {"Home", "Predict", "Performance", "Admin", "Ethics"}


def test_prediction_summary_is_serializable_and_omits_identifiers():
    result = {
        "predicted_score": 12.5,
        "predicted_support": "medium_support",
        "class_probabilities": {"high_support": 0.2, "medium_support": 0.6, "low_support": 0.2},
        "explanation": {"top": [{"feature": "studytime", "shap": 0.3}]},
        "recommendations": [{"gap": "Study routine", "action": "Plan weekly practice"}],
        "disclaimer": "Decision support only.",
        "name": "Private Student",
        "student_id": "STU1234",
    }
    encoded = json.dumps(_prediction_summary(result))
    assert "Private Student" not in encoded
    assert "STU1234" not in encoded
    assert "studytime" in encoded
