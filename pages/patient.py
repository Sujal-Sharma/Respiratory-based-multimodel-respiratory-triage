"""
pages/patient.py — Patient portal.

Sections:
  1. My Profile   — view/edit medical profile
  2. Self-Screen  — Tier 1 symptom-based triage
  3. My History   — past sessions + trend chart + deterioration alerts
"""

import os
import sys
import json
import tempfile
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Patient Portal — Respiratory Triage AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if "user" not in st.session_state or st.session_state.user is None:
    st.switch_page("app.py")

user = st.session_state.user
if user.get("role") != "patient":
    st.switch_page("pages/doctor.py")

from database.auth_store   import AuthStore
from database.session_store import SessionStore

auth_store    = AuthStore()
session_store = SessionStore()
patient_id    = f"patient_{user['id']}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {user.get('full_name') or user['username']}")
    st.caption(f"Patient ID: {patient_id}")
    st.markdown("---")
    page = st.radio("Navigation", ["Self-Screen", "My History", "My Profile"])
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.switch_page("app.py")
    st.caption("Powered by OPERA-CT + LangGraph")
    st.caption("For research purposes only.")

profile = auth_store.get_profile(user["id"]) or {}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: My Profile
# ══════════════════════════════════════════════════════════════════════════════
if page == "My Profile":
    st.title("My Profile")
    st.markdown("---")

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            age    = st.number_input("Age", 1, 120, int(profile.get("age") or 25))
            gender = st.selectbox("Gender", ["male", "female", "other"],
                                  index=["male","female","other"].index(
                                      profile.get("gender","male")))
        with col2:
            resp_cond = st.checkbox("Known Respiratory Condition",
                                    value=bool(profile.get("respiratory_condition", 0)))
            smoking   = st.checkbox("Smoker / Ex-smoker",
                                    value=bool(profile.get("smoking", 0)))
        notes = st.text_area("Medical Notes (optional)", value=profile.get("notes",""))

        if st.form_submit_button("Save Profile", type="primary"):
            auth_store.update_profile(user["id"], age, gender, resp_cond, smoking, notes)
            st.success("Profile saved.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Self-Screen (Tier 1)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Self-Screen":
    st.title("Symptom Self-Screening")
    st.info("Answer the questions below to get an instant AI-based respiratory risk assessment.")
    st.markdown("---")

    # Pre-fill from profile
    col_form, col_info = st.columns([2, 1])
    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            age    = st.number_input("Age", 1, 120, int(profile.get("age") or 25))
        with c2:
            gender = st.selectbox("Gender", ["male","female","other"],
                                  index=["male","female","other"].index(
                                      profile.get("gender","male")))

        st.markdown("**Current Symptoms**")
        c1, c2 = st.columns(2)
        with c1:
            fever    = st.checkbox("Fever / Muscle Pain")
            dyspnea  = st.checkbox("Difficulty Breathing")
            wheezing = st.checkbox("Wheezing")
        with c2:
            resp_cond  = st.checkbox("Known Respiratory Condition",
                                     value=bool(profile.get("respiratory_condition", 0)))
            congestion = st.checkbox("Nasal Congestion")
            cough_sev  = st.slider("Cough Severity (0=none, 1=severe)", 0.0, 1.0, 0.3, 0.1)

    with col_info:
        st.markdown("**How this works**")
        st.info(
            "Tier 1 screening uses your symptoms to estimate respiratory risk "
            "using GOLD 2024 and BTS 2023 clinical guidelines.\n\n"
            "For full audio-based analysis, ask your doctor to run a Tier 2 assessment."
        )

    symptom_list = []
    if fever:      symptom_list.append("fever/muscle pain")
    if dyspnea:    symptom_list.append("difficulty breathing")
    if wheezing:   symptom_list.append("wheezing")
    if resp_cond:  symptom_list.append("respiratory condition")
    if congestion: symptom_list.append("nasal congestion")

    st.markdown("---")
    if st.button("Run Screening", type="primary", use_container_width=True):
        patient_info = {
            "age": age, "gender": gender,
            "symptoms": symptom_list,
            "fever_muscle_pain": fever,
            "respiratory_condition": resp_cond,
            "cough_detected": cough_sev,
            "dyspnea": dyspnea,
            "wheezing": wheezing,
            "congestion": congestion,
        }

        with st.spinner("Analysing symptoms..."):
            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info,
                                cough_audio_path="",
                                lung_audio_path="",
                                patient_id=patient_id)

        decision = result.get("triage_decision", {})
        severity = decision.get("severity", "UNKNOWN")

        severity_colors = {"LOW": "green", "MODERATE": "orange",
                           "HIGH": "red", "CRITICAL": "red"}
        color = severity_colors.get(severity, "gray")

        st.markdown("---")
        st.subheader("Screening Result")
        st.markdown(
            f'<div style="background:{color};padding:16px;border-radius:8px;'
            f'text-align:center;margin-bottom:16px;">'
            f'<h2 style="color:white;margin:0;">Severity: {severity}</h2></div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Diagnosis",   decision.get("diagnosis","N/A"))
            st.metric("Confidence",  f"{decision.get('confidence',0):.0%}")
        with c2:
            st.metric("Referral",    decision.get("referral_urgency","none").upper())
            st.metric("Action",      decision.get("recommended_action","—"))

        st.markdown("**Clinical Reasoning**")
        st.info(decision.get("reasoning","—"))

        # Symptom agent details
        sym = result.get("symptom_result", {})
        with st.expander("Symptom Analysis Details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Symptomatic Risk",    f"{sym.get('symptomatic_probability',0):.0%}")
            c2.metric("COPD Hint",           f"{sym.get('copd_probability_hint',0):.0%}")
            c3.metric("Pneumonia Hint",      f"{sym.get('pneumonia_probability_hint',0):.0%}")

        # Deterioration alerts
        session_res = result.get("session_result", {})
        alerts = session_res.get("deterioration_alerts")
        if alerts:
            for alert in alerts:
                st.error(f"Deterioration Alert: {alert['message']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: My History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "My History":
    st.title("My Screening History")
    st.markdown("---")

    history = session_store.get_sessions(patient_id, n=20)

    if not history:
        st.info("No screening sessions yet. Go to 'Self-Screen' to run your first assessment.")
    else:
        # Summary metrics
        df = pd.DataFrame(history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sessions", len(df))
        c2.metric("Latest Severity", df.iloc[0]["severity"])
        c3.metric("Avg COPD Risk",   f"{df['copd_confidence'].mean():.0%}")
        c4.metric("Avg Pneumonia Risk", f"{df['pneu_confidence'].mean():.0%}")

        # Trend chart
        st.subheader("Risk Trend Over Time")
        df_plot = df.iloc[::-1].reset_index(drop=True)  # chronological
        df_plot["session"] = range(1, len(df_plot) + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_plot["session"], df_plot["copd_confidence"],
                marker='o', label="COPD Risk", color="#EF5350")
        ax.plot(df_plot["session"], df_plot["pneu_confidence"],
                marker='s', label="Pneumonia Risk", color="#42A5F5")
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label="COPD threshold")
        ax.axhline(0.64, color='red', linestyle='--', alpha=0.5, label="Pneumonia threshold")
        ax.set_xlabel("Session")
        ax.set_ylabel("Confidence")
        ax.set_title("Longitudinal Risk Monitoring")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Deterioration check
        alerts = session_store.check_deterioration(patient_id)
        if alerts:
            for alert in alerts:
                st.error(f"Deterioration Alert — {alert['message']}")

        # Session table
        st.subheader("Session Log")
        display_df = df[["timestamp","tier","severity","diagnosis",
                          "copd_confidence","pneu_confidence","sound_type","action"]].copy()
        display_df.columns = ["Time","Tier","Severity","Diagnosis",
                               "COPD Risk","Pneu Risk","Sound","Action"]
        display_df["COPD Risk"] = display_df["COPD Risk"].apply(lambda x: f"{x:.0%}")
        display_df["Pneu Risk"] = display_df["Pneu Risk"].apply(lambda x: f"{x:.0%}")
        st.dataframe(display_df, use_container_width=True)
