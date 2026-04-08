"""
pages/doctor_patient.py — Doctor view: individual patient detail + Tier 2.

Sections:
  1. Patient Summary    — profile + baseline + latest scores + alerts
  2. Longitudinal Chart — symptom/voice/longitudinal trend over all sessions
  3. Session History    — full session log table
  4. Tier 2 Assessment  — stethoscope upload → full AI evaluation
"""

import os, sys, tempfile
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Patient Detail — Doctor Portal",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded",
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if "user" not in st.session_state or not st.session_state.user:
    st.switch_page("app.py")
user = st.session_state.user
if user.get("role") != "doctor":
    st.switch_page("pages/patient.py")
if "selected_patient_id" not in st.session_state:
    st.switch_page("pages/doctor.py")

from database.auth_store    import AuthStore
from database.session_store import SessionStore
from pipeline.longitudinal  import interpret_score

auth_store    = AuthStore()
session_store = SessionStore()

sel_id      = st.session_state.selected_patient_id
sel_pid     = st.session_state.selected_patient_pid
sel_name    = st.session_state.selected_patient_name
sel_profile = auth_store.get_profile(sel_id) or {}
sel_user    = auth_store.get_user_by_id(sel_id) or {}

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

# ── Data ──────────────────────────────────────────────────────────────────────
history  = session_store.get_sessions(sel_pid, n=50)
alerts   = session_store.check_deterioration(sel_pid)
baseline = session_store.get_baseline(sel_pid)
latest   = history[0] if history else {}

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Patient: {sel_name}")
st.caption(f"ID: {sel_pid}  |  Username: @{sel_user.get('username','—')}")
st.markdown("---")

# ── Section 1: Summary ────────────────────────────────────────────────────────
st.subheader("Patient Summary")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Age",       sel_profile.get("age") or "—")
c2.metric("Gender",    (sel_profile.get("gender") or "—").capitalize())
c3.metric("Resp. Cond","Yes" if sel_profile.get("respiratory_condition") else "No")
c4.metric("Smoker",    "Yes" if sel_profile.get("smoking") else "No")
c5.metric("Sessions",  len(history))
c6.metric("Latest Severity", latest.get("severity","—"))

if sel_profile.get("notes"):
    st.info(f"Clinical Notes: {sel_profile['notes']}")

# Latest scores row
if history:
    long_score = latest.get("longitudinal_score", 0.0)
    interp     = interpret_score(long_score)
    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    sc1.metric("Longitudinal Score", f"{long_score:.0%}",
               help="Combined risk: 50% symptom + 35% voice + 15% cough drift")
    sc2.metric("Symptom Index",      f"{latest.get('symptom_index',0):.0%}")
    sc3.metric("Voice Index",        f"{latest.get('voice_index',0):.0%}",
               help="Acoustic deviation from patient's baseline")
    sc4.metric("COPD Risk",          f"{latest.get('copd_confidence',0):.0%}")
    sc5.metric("Pneumonia Risk",     f"{latest.get('pneu_confidence',0):.0%}")

# Voice baseline
if baseline and baseline.get("voice_features"):
    vf = baseline["voice_features"]
    with st.expander("Voice Baseline (established first session)"):
        bc1,bc2,bc3,bc4 = st.columns(4)
        bc1.metric("Jitter",   f"{vf.get('jitter',0):.4f}")
        bc2.metric("Shimmer",  f"{vf.get('shimmer',0):.4f}")
        bc3.metric("HNR (dB)", f"{vf.get('hnr',0):.1f}")
        bc4.metric("Duration", f"{vf.get('phonation_duration',0):.1f}s")
        st.caption(f"Baseline created: {baseline.get('created_at','—')[:10]}")

# Deterioration alerts
if alerts:
    for alert in alerts:
        st.error(f"Deterioration Alert — {alert['message']}")

st.markdown("---")

# ── Section 2: Longitudinal Chart ─────────────────────────────────────────────
st.subheader("Longitudinal Risk Trend")

if history:
    df = pd.DataFrame(history)
    df_plot = df.iloc[::-1].reset_index(drop=True)
    df_plot["session"] = range(1, len(df_plot)+1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: longitudinal composite
    ax1.fill_between(df_plot["session"],
                     df_plot.get("longitudinal_score", 0), alpha=0.15, color="#7E57C2")
    ax1.plot(df_plot["session"],
             df_plot.get("longitudinal_score", pd.Series([0]*len(df_plot))),
             marker='D', color="#7E57C2", lw=2, label="Longitudinal Score")
    ax1.plot(df_plot["session"],
             df_plot.get("symptom_index", pd.Series([0]*len(df_plot))),
             marker='o', color="#EF5350", lw=1.5, ls='--', label="Symptom Index")
    ax1.plot(df_plot["session"],
             df_plot.get("voice_index", pd.Series([0]*len(df_plot))),
             marker='s', color="#42A5F5", lw=1.5, ls='--', label="Voice Index")
    ax1.axhline(0.40, color='orange', ls='--', alpha=0.5)
    ax1.axhline(0.60, color='red',    ls='--', alpha=0.5)

    t2 = df_plot[df_plot["tier"] == 2]
    if not t2.empty:
        ax1.scatter(t2["session"],
                    t2.get("longitudinal_score", pd.Series([0]*len(t2))),
                    marker='*', s=150, color='darkred', zorder=5, label="Tier 2")
    ax1.set_ylabel("Score (0–1)")
    ax1.set_title(f"Longitudinal Monitoring — {sel_name}")
    ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3); ax1.set_ylim(-0.02, 1.05)

    # Bottom: COPD / Pneumonia audio confidence
    ax2.plot(df_plot["session"], df_plot["copd_confidence"],
             marker='o', color="#EF5350", lw=2, label="COPD Confidence")
    ax2.plot(df_plot["session"], df_plot["pneu_confidence"],
             marker='s', color="#42A5F5", lw=2, label="Pneumonia Confidence")
    ax2.axhline(0.5,  color='orange', ls='--', alpha=0.5, label="COPD threshold")
    ax2.axhline(0.64, color='red',    ls='--', alpha=0.5, label="Pneu threshold")
    ax2.set_xlabel("Session"); ax2.set_ylabel("Confidence (0–1)")
    ax2.set_title("Audio Agent Confidence (Tier 2 sessions)")
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3); ax2.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    st.pyplot(fig); plt.close()
else:
    st.info("No sessions yet.")

st.markdown("---")

# ── Section 3: Session Log ────────────────────────────────────────────────────
st.subheader("Full Session History")
if history:
    df = pd.DataFrame(history)
    show_cols = ["timestamp","tier","severity","diagnosis",
                 "longitudinal_score","symptom_index","voice_index","drift_score",
                 "copd_confidence","pneu_confidence","sound_type","action"]
    show_cols = [c for c in show_cols if c in df.columns]
    disp = df[show_cols].copy()
    for col in ["longitudinal_score","symptom_index","voice_index","drift_score",
                "copd_confidence","pneu_confidence"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{x:.0%}")
    disp.columns = [c.replace("_"," ").title() for c in disp.columns]
    st.dataframe(disp, use_container_width=True)

st.markdown("---")

# ── Section 4: Tier 2 Assessment ─────────────────────────────────────────────
st.subheader("Tier 2 Clinical Assessment")
st.info(
    "Upload a stethoscope lung sound recording for this patient. "
    "The full AI pipeline (COPD + Pneumonia + Sound agents) will run "
    "and the result will be saved to the patient's history."
)

col_up, col_sym = st.columns([1,1])

with col_up:
    lung_file = st.file_uploader(
        "Stethoscope Lung Sound Recording",
        type=["wav","mp3","webm","ogg","m4a","flac","mp4"],
        help="8-15 seconds of lung auscultation."
    )
    if lung_file:
        st.audio(lung_file, format="audio/wav")

with col_sym:
    st.markdown("**Reported Symptoms**")
    c1,c2 = st.columns(2)
    with c1:
        fever    = st.checkbox("Fever / Muscle Pain",    key="d_fever")
        dyspnea  = st.checkbox("Difficulty Breathing",   key="d_dyspnea")
        wheezing = st.checkbox("Wheezing",               key="d_wheeze")
    with c2:
        resp_cond  = st.checkbox("Respiratory Condition",
                                 value=bool(sel_profile.get("respiratory_condition",0)),
                                 key="d_resp")
        congestion = st.checkbox("Nasal Congestion",     key="d_cong")
        cough_sev  = st.slider("Cough Severity", 0.0, 1.0, 0.3, 0.1, key="d_cough")
    dyspnea_level = st.select_slider(
        "Breathlessness (mMRC)",
        options=[0,1,2,3,4],
        format_func=lambda x: {0:"0-None",1:"1-Mild",2:"2-Moderate",
                                3:"3-Severe",4:"4-Very Severe"}[x],
        key="d_mmrc"
    )

age    = int(sel_profile.get("age") or 45)
gender = sel_profile.get("gender","male")

st.markdown("---")
if st.button("Run Tier 2 Assessment", type="primary",
             disabled=lung_file is None, use_container_width=True):

    symptom_list = []
    if fever:      symptom_list.append("fever/muscle pain")
    if dyspnea:    symptom_list.append("difficulty breathing")
    if wheezing:   symptom_list.append("wheezing")
    if resp_cond:  symptom_list.append("respiratory condition")
    if congestion: symptom_list.append("nasal congestion")

    patient_info = {
        "age": age, "gender": gender,
        "symptoms": symptom_list,
        "fever_muscle_pain":     fever,
        "respiratory_condition": resp_cond,
        "cough_detected":        cough_sev,
        "cough_severity":        cough_sev * 10,
        "dyspnea":               dyspnea or dyspnea_level >= 2,
        "dyspnea_level":         dyspnea_level,
        "wheezing":              wheezing,
        "congestion":            congestion,
        "chest_tightness": 0, "sleep_quality": 0,
        "energy_level": 0, "sputum": 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        lung_path = os.path.join(tmpdir, f"lung.{lung_file.name.split('.')[-1]}")
        with open(lung_path,"wb") as f:
            f.write(lung_file.getbuffer())

        with st.spinner("Running full Tier 2 analysis..."):
            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info,
                                cough_audio_path = lung_path,
                                lung_audio_path  = lung_path,
                                vowel_audio_path = "",
                                patient_id       = sel_pid)

    decision   = result.get("triage_decision", {})
    severity   = decision.get("severity", "UNKNOWN")
    long_score = result.get("longitudinal_score", 0.0)
    interp     = interpret_score(long_score)

    sev_colors = {"LOW":"green","MODERATE":"orange","HIGH":"red","CRITICAL":"red"}
    color = sev_colors.get(severity, "gray")

    st.markdown("---")
    st.subheader("Tier 2 Result")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(
            f'<div style="background:{color};padding:16px;border-radius:8px;'
            f'text-align:center;">'
            f'<h2 style="color:white;margin:0;">Severity: {severity}</h2></div>',
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            f'<div style="background:{interp["color"]};padding:16px;'
            f'border-radius:8px;text-align:center;">'
            f'<h3 style="color:white;margin:0;">{interp["label"]}</h3>'
            f'<p style="color:white;margin:4px 0 0;">{interp["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    rc1,rc2,rc3 = st.columns(3)
    rc1.metric("Diagnosis",  decision.get("diagnosis","N/A"))
    rc2.metric("Confidence", f"{decision.get('confidence',0):.0%}")
    rc3.metric("Referral",   decision.get("referral_urgency","none").upper())

    st.markdown("**Clinical Reasoning**")
    st.info(decision.get("reasoning","—"))
    st.markdown("**Recommended Action**")
    st.success(decision.get("recommended_action","Consult specialist."))

    with st.expander("Agent Details", expanded=True):
        a1, a2, a3 = st.columns(3)
        copd_r = result.get("copd_result", {})
        pneu_r = result.get("pneumonia_result", {})
        snd_r  = result.get("sound_result",  {})

        with a1:
            st.markdown("**COPD Agent**")
            st.write(f"Detected: **{'Yes' if copd_r.get('detected') else 'No'}**")
            st.metric("Probability", f"{copd_r.get('probability',0):.1%}")
            st.metric("Confidence",  f"{copd_r.get('confidence',0):.1%}")
            st.progress(float(copd_r.get('probability', 0)))

        with a2:
            st.markdown("**Pneumonia Agent**")
            st.write(f"Detected: **{'Yes' if pneu_r.get('detected') else 'No'}**")
            st.metric("Probability", f"{pneu_r.get('probability',0):.1%}")
            st.metric("Confidence",  f"{pneu_r.get('confidence',0):.1%}")
            st.progress(float(pneu_r.get('probability', 0)))

        with a3:
            st.markdown("**Sound Agent**")
            if snd_r:
                st.write(f"Type: **{snd_r.get('sound_type','N/A')}**")
                st.metric("Confidence", f"{snd_r.get('confidence',0):.1%}")
                probs = snd_r.get("all_probabilities", {})
                if probs:
                    for label, prob in probs.items():
                        st.write(f"{label}: {prob:.1%}")
                        st.progress(float(prob))
            else:
                st.write("Not available")

    session_res = result.get("session_result", {})
    det_alerts  = session_res.get("deterioration_alerts")
    if det_alerts:
        for alert in det_alerts:
            st.error(f"Deterioration Alert: {alert['message']}")

    st.success("Result saved to patient history.")

elif lung_file is None:
    st.warning("Upload a stethoscope recording to run Tier 2.")
