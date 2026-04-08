"""
pages/doctor_patient.py — Doctor view: individual patient detail + Tier 2.

Sections:
  1. Patient Summary    — profile + latest severity + deterioration alerts
  2. Session History    — table + trend chart
  3. Tier 2 Assessment  — upload stethoscope recording → full AI evaluation
"""

import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Patient Detail — Doctor Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if "user" not in st.session_state or st.session_state.user is None:
    st.switch_page("app.py")

user = st.session_state.user
if user.get("role") != "doctor":
    st.switch_page("pages/patient.py")

if "selected_patient_id" not in st.session_state:
    st.switch_page("pages/doctor.py")

from database.auth_store    import AuthStore
from database.session_store import SessionStore

auth_store    = AuthStore()
session_store = SessionStore()

sel_user_id  = st.session_state.selected_patient_id
sel_pid      = st.session_state.selected_patient_pid
sel_name     = st.session_state.selected_patient_name
sel_profile  = auth_store.get_profile(sel_user_id) or {}
sel_user     = auth_store.get_user_by_id(sel_user_id) or {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### Dr. {user.get('full_name') or user['username']}")
    st.caption("Doctor Portal")
    st.markdown("---")
    if st.button("Back to Patient List", use_container_width=True):
        st.switch_page("pages/doctor.py")
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.pop("selected_patient_id", None)
        st.switch_page("app.py")
    st.caption("Powered by OPERA-CT + LangGraph")
    st.caption("For research purposes only.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Patient: {sel_name}")
st.caption(f"ID: {sel_pid} | Username: {sel_user.get('username','—')}")
st.markdown("---")

history = session_store.get_sessions(sel_pid, n=50)
alerts  = session_store.check_deterioration(sel_pid)

# ── Section 1: Patient Summary ────────────────────────────────────────────────
st.subheader("Patient Summary")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Age",     sel_profile.get("age") or "—")
col2.metric("Gender",  (sel_profile.get("gender") or "—").capitalize())
col3.metric("Resp. Condition", "Yes" if sel_profile.get("respiratory_condition") else "No")
col4.metric("Smoker",  "Yes" if sel_profile.get("smoking") else "No")
col5.metric("Sessions", len(history))

if sel_profile.get("notes"):
    st.info(f"Notes: {sel_profile['notes']}")

if alerts:
    for alert in alerts:
        st.error(f"Deterioration Alert — {alert['message']}")

st.markdown("---")

# ── Section 2: History + Trend Chart ─────────────────────────────────────────
st.subheader("Session History")

if not history:
    st.info("No sessions recorded yet for this patient.")
else:
    df = pd.DataFrame(history)

    # Trend chart
    df_plot = df.iloc[::-1].reset_index(drop=True)
    df_plot["session"] = range(1, len(df_plot) + 1)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df_plot["session"], df_plot["copd_confidence"],
            marker='o', label="COPD Risk", color="#EF5350")
    ax.plot(df_plot["session"], df_plot["pneu_confidence"],
            marker='s', label="Pneumonia Risk", color="#42A5F5")
    ax.axhline(0.5,  color='orange', linestyle='--', alpha=0.5, label="COPD threshold (0.50)")
    ax.axhline(0.64, color='red',    linestyle='--', alpha=0.5, label="Pneumonia threshold (0.64)")

    # Mark Tier 2 sessions
    t2 = df_plot[df_plot["tier"] == 2]
    if not t2.empty:
        ax.scatter(t2["session"], t2["copd_confidence"],
                   marker='*', s=120, color='darkred', zorder=5, label="Tier 2 session")

    ax.set_xlabel("Session"); ax.set_ylabel("Risk Confidence")
    ax.set_title(f"Longitudinal Risk — {sel_name}")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Session table
    display_df = df[["timestamp","tier","severity","diagnosis",
                      "copd_confidence","pneu_confidence","sound_type","action"]].copy()
    display_df.columns = ["Time","Tier","Severity","Diagnosis",
                           "COPD Risk","Pneu Risk","Sound","Action"]
    display_df["COPD Risk"] = display_df["COPD Risk"].apply(lambda x: f"{x:.0%}")
    display_df["Pneu Risk"] = display_df["Pneu Risk"].apply(lambda x: f"{x:.0%}")
    st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# ── Section 3: Tier 2 Assessment ─────────────────────────────────────────────
st.subheader("Tier 2 Clinical Assessment")
st.info(
    "Upload a stethoscope lung sound recording for this patient. "
    "The full AI pipeline (COPD + Pneumonia + Sound agents) will run "
    "and the result will be saved to the patient's history."
)

col_upload, col_form = st.columns([1, 1])

with col_upload:
    lung_file = st.file_uploader(
        "Stethoscope Lung Sound Recording",
        type=["wav", "mp3", "webm", "ogg", "m4a", "flac", "mp4"],
        help="8-15 seconds of lung auscultation."
    )
    if lung_file:
        st.audio(lung_file, format="audio/wav")

with col_form:
    st.markdown("**Patient Symptoms (as reported)**")
    c1, c2 = st.columns(2)
    with c1:
        fever    = st.checkbox("Fever / Muscle Pain",     key="d_fever")
        dyspnea  = st.checkbox("Difficulty Breathing",    key="d_dyspnea")
        wheezing = st.checkbox("Wheezing",                key="d_wheeze")
    with c2:
        resp_cond  = st.checkbox("Respiratory Condition",
                                 value=bool(sel_profile.get("respiratory_condition",0)),
                                 key="d_resp")
        congestion = st.checkbox("Nasal Congestion",      key="d_cong")
        cough_sev  = st.slider("Cough Severity", 0.0, 1.0, 0.3, 0.1, key="d_cough")

    age    = int(sel_profile.get("age") or 45)
    gender = sel_profile.get("gender", "male")

st.markdown("---")
run_disabled = lung_file is None

if st.button("Run Tier 2 Assessment", type="primary",
             disabled=run_disabled, use_container_width=True):

    symptom_list = []
    if fever:      symptom_list.append("fever/muscle pain")
    if dyspnea:    symptom_list.append("difficulty breathing")
    if wheezing:   symptom_list.append("wheezing")
    if resp_cond:  symptom_list.append("respiratory condition")
    if congestion: symptom_list.append("nasal congestion")

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

    with tempfile.TemporaryDirectory() as tmpdir:
        lung_path = os.path.join(tmpdir, f"lung.{lung_file.name.split('.')[-1]}")
        with open(lung_path, "wb") as f:
            f.write(lung_file.getbuffer())

        with st.spinner("Running full Tier 2 analysis..."):
            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info,
                                cough_audio_path=lung_path,
                                lung_audio_path=lung_path,
                                patient_id=sel_pid)

    decision = result.get("triage_decision", {})
    severity = decision.get("severity", "UNKNOWN")
    severity_colors = {"LOW": "green", "MODERATE": "orange",
                       "HIGH": "red", "CRITICAL": "red"}
    color = severity_colors.get(severity, "gray")

    st.markdown("---")
    st.subheader("Tier 2 Result")
    st.markdown(
        f'<div style="background:{color};padding:16px;border-radius:8px;'
        f'text-align:center;margin-bottom:16px;">'
        f'<h2 style="color:white;margin:0;">Severity: {severity}</h2></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnosis",       decision.get("diagnosis","N/A"))
    c2.metric("Confidence",      f"{decision.get('confidence',0):.0%}")
    c3.metric("Referral",        decision.get("referral_urgency","none").upper())

    st.markdown("**Clinical Reasoning**")
    st.info(decision.get("reasoning","—"))
    st.markdown("**Recommended Action**")
    st.success(decision.get("recommended_action","Consult specialist."))

    # Agent breakdown
    with st.expander("Agent Details", expanded=True):
        a1, a2, a3 = st.columns(3)
        copd_r = result.get("copd_result", {})
        pneu_r = result.get("pneumonia_result", {})
        snd_r  = result.get("sound_result", {})

        with a1:
            st.markdown("**COPD Agent**")
            detected = copd_r.get("detected", False)
            st.write(f"Detected: **{'Yes' if detected else 'No'}**")
            st.metric("Probability", f"{copd_r.get('probability',0):.1%}")
            st.metric("Confidence",  f"{copd_r.get('confidence',0):.1%}")

        with a2:
            st.markdown("**Pneumonia Agent**")
            detected = pneu_r.get("detected", False)
            st.write(f"Detected: **{'Yes' if detected else 'No'}**")
            st.metric("Probability", f"{pneu_r.get('probability',0):.1%}")
            st.metric("Confidence",  f"{pneu_r.get('confidence',0):.1%}")

        with a3:
            st.markdown("**Sound Agent**")
            if snd_r:
                st.write(f"Type: **{snd_r.get('sound_type','N/A')}**")
                st.metric("Confidence", f"{snd_r.get('confidence',0):.1%}")
                probs = snd_r.get("all_probabilities", {})
                if probs:
                    st.json(probs)
            else:
                st.write("Not available")

    # Deterioration alerts
    session_res = result.get("session_result", {})
    det_alerts  = session_res.get("deterioration_alerts")
    if det_alerts:
        for alert in det_alerts:
            st.error(f"Deterioration Alert: {alert['message']}")

    st.success("Result saved to patient history.")

elif run_disabled:
    st.warning("Please upload a stethoscope recording to run Tier 2.")
