"""
app.py — Landing page: Login / Register

Routes to:
  pages/patient.py  — Patient portal (Tier 1 + history)
  pages/doctor.py   — Doctor portal  (patient list + Tier 2)

Run:  streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Respiratory Triage AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from database.auth_store import AuthStore

auth = AuthStore()

# ── Session state init ────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── Already logged in → redirect ─────────────────────────────────────────────
if st.session_state.user:
    role = st.session_state.user.get("role")
    if role == "doctor":
        st.switch_page("pages/doctor.py")
    else:
        st.switch_page("pages/patient.py")

# ── Landing UI ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>Respiratory Triage AI</h1>"
    "<p style='text-align:center; color:gray;'>Multimodal AI-powered respiratory screening</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab_login, tab_register = st.tabs(["Login", "Register (Patient)"])

# ── Login ─────────────────────────────────────────────────────────────────────
with tab_login:
    st.subheader("Sign In")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", type="primary", use_container_width=True):
        if not username or not password:
            st.error("Please enter username and password.")
        else:
            user = auth.login(username, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome, {user['full_name'] or user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.caption("Doctor account: username `doctor` / password `doctor123`")

# ── Register ──────────────────────────────────────────────────────────────────
with tab_register:
    st.subheader("Create Patient Account")
    col1, col2 = st.columns(2)
    with col1:
        reg_name     = st.text_input("Full Name")
        reg_user     = st.text_input("Username")
    with col2:
        reg_pass     = st.text_input("Password", type="password")
        reg_pass2    = st.text_input("Confirm Password", type="password")

    if st.button("Create Account", type="primary", use_container_width=True):
        if not all([reg_name, reg_user, reg_pass, reg_pass2]):
            st.error("Please fill in all fields.")
        elif reg_pass != reg_pass2:
            st.error("Passwords do not match.")
        elif len(reg_pass) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            result = auth.register_user(reg_user, reg_pass, 'patient', reg_name)
            if result['success']:
                st.success("Account created! Please log in.")
            else:
                st.error(result.get('error', 'Registration failed.'))

st.markdown("---")
st.caption("For research purposes only. Not a medical device.")
