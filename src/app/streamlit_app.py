"""Compatibility exports for earlier imports from src.app.streamlit_app."""

from app.auth import _login_screen
from app.pages.admin import page_admin
from app.pages.analytics import page_ethics, page_metrics
from app.pages._college_dashboard import _college_dashboard
from app.pages.faculty import page_faculty
from app.pages.prediction import page_home, page_predict
from app.pages.reports import page_reports
from app.pages.student import page_student
from app.shared import DEFAULTS, _form_values, _models_ready, _prediction_summary, _train_help
from reports.report_generator import _build_college_pdf, _build_pdf_report

pages = {
    "Home": page_home,
    "Predict": page_predict,
    "Performance": page_metrics,
    "Admin": page_admin,
    "Faculty": page_faculty,
    "Student": page_student,
    "Reports": page_reports,
    "Ethics": page_ethics,
}
