"""Role-based Streamlit application entry point."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, APP_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from auth import _login_screen
from pages.admin import page_admin
from pages.analytics import page_ethics, page_metrics
from pages.faculty import page_faculty
from pages.prediction import page_home, page_predict
from pages.reports import page_reports
from pages.student import page_student
from database import database as db

st.set_page_config(page_title="Trekverse AI Academy", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f5f8fc; }
  [data-testid="stSidebar"] { background: #102a43; }
  [data-testid="stSidebar"] * { color: #f4f8fb !important; }
  h1, h2, h3 { color: #102a43; }
  div[data-testid="stMetric"] { background: white; padding: 14px 16px; border-radius: 12px; box-shadow: 0 2px 10px #17324d12; }
  div.stButton > button[kind="primary"] { background: #168c83; border-color: #168c83; }
</style>
""", unsafe_allow_html=True)

db.account_count()
if "user" not in st.session_state:
    selected_page = st.navigation([st.Page(_login_screen, title="Sign in")], position="sidebar")
else:
    user = st.session_state["user"]
    st.sidebar.title("Trekverse AI Academy")
    st.sidebar.caption(f"Signed in as {user['username']} · {user['role']}")
    if st.sidebar.button("Sign out"):
        del st.session_state["user"]
        st.rerun()

    if user["role"] == "ADMIN":
        available_pages = {
            "Home": page_home, "Admin": page_admin, "Faculty": page_faculty,
            "Student": page_student, "Predict": page_predict,
            "Performance": page_metrics, "Reports": page_reports, "Ethics": page_ethics,
        }
    elif user["role"] == "FACULTY":
        available_pages = {
            "Faculty": page_faculty, "Predict": page_predict,
            "Performance": page_metrics, "Reports": page_reports, "Ethics": page_ethics,
        }
    else:
        available_pages = {"Student": page_student, "Reports": page_reports, "Ethics": page_ethics}

    # Explicit navigation disables Streamlit's legacy automatic pages discovery.
    selected_page = st.navigation(
        [st.Page(handler, title=title) for title, handler in available_pages.items()],
        position="sidebar",
    )

selected_page.run()
