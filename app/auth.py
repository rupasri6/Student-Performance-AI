"""Local demonstration sign-in and first administrator setup."""

import streamlit as st

from database import database as db

def _login_screen():
    st.title("Trekverse AI Academy")
    st.caption("Student Intelligence Platform")
    if db.account_count() == 0:
        st.info("First launch: create the administrator account. Additional admin accounts can only be provisioned by an administrator.")
        with st.form("initial_admin"):
            username = st.text_input("Administrator username")
            password = st.text_input("Password (10+ characters)", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create administrator")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    db.create_user(username, password, "ADMIN")
                    st.success("Administrator created. Sign in below.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        return
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        user = db.authenticate(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Username or password is incorrect.")
