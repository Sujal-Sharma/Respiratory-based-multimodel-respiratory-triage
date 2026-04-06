"""
app.py — Streamlit web app for Multimodal Respiratory Triage.

Two-tier interface:
  Tier 1 (Patient): Cough/lung recording + symptoms -> COPD & Pneumonia risk screening
  Tier 2 (Clinician): + Stethoscope lung recording  -> adds lung sound analysis

Architecture:
  OPERA-CT embeddings -> COPD Agent + Pneumonia Agent + Sound Agent (Tier 2)
  -> Deterministic Rule Engine -> Triage Decision

Run:  streamlit run app.py
"""

import os
import json
import tempfile
import streamlit as st

st.set_page_config(
    page_title="Respiratory Triage AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Respiratory Triage AI")
    st.markdown("---")

    tier = st.radio(
        "Select Tier",
        ["Tier 1 -- Patient Self-Screening", "Tier 2 -- Clinician Confirmation"],
        help="Tier 1: audio + symptoms only. Tier 2: adds stethoscope lung sound analysis."
    )
    is_tier2 = "Tier 2" in tier

    st.markdown("---")
    st.markdown("**How it works**")
    if not is_tier2:
        st.info(
            "Upload a cough or breathing recording and fill in symptoms. "
            "The AI screens for COPD and Pneumonia risk using OPERA-CT embeddings "
            "and clinical rule-based reasoning."
        )
    else:
        st.info(
            "Upload an audio recording + a stethoscope lung sound recording. "
            "The AI runs all 3 specialist agents (COPD, Pneumonia, Sound) "
            "and applies BTS/GOLD/GINA clinical guidelines."
        )

    st.markdown("---")
    st.caption("Powered by OPERA-CT + LangGraph")
    st.caption("For research purposes only. Not a medical device.")


# ── Main content ─────────────────────────────────────────────────────────────
st.header("Tier 2 -- Clinician Confirmation" if is_tier2 else "Tier 1 -- Patient Self-Screening")

col_audio, col_symptoms = st.columns([1, 1])

# ── Audio uploads ─────────────────────────────────────────────────────────────
with col_audio:
    st.subheader("Audio Recording")

    audio_file = st.file_uploader(
        "Upload Cough or Breathing Recording",
        type=["wav", "mp3", "webm", "ogg", "m4a", "flac"],
        help="Record 3-15 seconds of coughing or breathing using your phone or microphone."
    )
    if audio_file:
        st.audio(audio_file, format="audio/wav")

    lung_file = None
    if is_tier2:
        st.markdown("---")
        lung_file = st.file_uploader(
            "Upload Stethoscope Lung Sound Recording",
            type=["wav", "mp3", "webm", "ogg", "m4a", "flac"],
            help="Record lung sounds using a stethoscope (8-15 seconds)."
        )
        if lung_file:
            st.audio(lung_file, format="audio/wav")

# ── Symptom form ──────────────────────────────────────────────────────────────
with col_symptoms:
    st.subheader("Patient Information")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with c2:
        gender = st.selectbox("Gender", ["male", "female", "other"])

    st.markdown("**Symptoms**")
    c1, c2 = st.columns(2)
    with c1:
        fever       = st.checkbox("Fever / Muscle Pain")
        dyspnea     = st.checkbox("Difficulty Breathing")
        wheezing    = st.checkbox("Wheezing")
    with c2:
        resp_cond   = st.checkbox("Known Respiratory Condition")
        congestion  = st.checkbox("Nasal Congestion")
        cough_severity = st.slider("Cough Severity", 0.0, 1.0, 0.5, 0.1,
                                   help="0 = no cough, 1 = severe/persistent cough")

    symptom_list = []
    if fever:        symptom_list.append("fever/muscle pain")
    if dyspnea:      symptom_list.append("difficulty breathing")
    if wheezing:     symptom_list.append("wheezing")
    if resp_cond:    symptom_list.append("respiratory condition")
    if congestion:   symptom_list.append("nasal congestion")

# ── Run triage ────────────────────────────────────────────────────────────────
st.markdown("---")

run_disabled = audio_file is None
if is_tier2 and lung_file is None:
    run_disabled = True

run_label = "Run Full Triage" if is_tier2 else "Screen for Risk"
if st.button(run_label, type="primary", disabled=run_disabled, use_container_width=True):

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"audio.{audio_file.name.split('.')[-1]}")
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        lung_path = ""
        if is_tier2 and lung_file:
            lung_path = os.path.join(tmpdir, f"lung.{lung_file.name.split('.')[-1]}")
            with open(lung_path, "wb") as f:
                f.write(lung_file.getbuffer())

        patient_info = {
            "age":                  age,
            "gender":               gender,
            "symptoms":             symptom_list,
            "fever_muscle_pain":    fever,
            "respiratory_condition": resp_cond,
            "cough_detected":       cough_severity,
            "dyspnea":              dyspnea,
            "wheezing":             wheezing,
            "congestion":           congestion,
        }

        with st.spinner("Running triage pipeline..."):
            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info, audio_path, lung_path)

    # ── Display results ───────────────────────────────────────────────────
    decision = result.get("triage_decision", {})
    severity = decision.get("severity", "UNKNOWN")

    severity_colors = {
        "LOW":      "green",
        "MODERATE": "orange",
        "HIGH":     "red",
        "CRITICAL": "red",
    }
    color = severity_colors.get(severity, "gray")

    st.markdown("---")
    st.subheader("Triage Decision")

    st.markdown(
        f'<div style="background-color:{color}; padding:16px; border-radius:8px; '
        f'text-align:center; margin-bottom:16px;">'
        f'<h2 style="color:white; margin:0;">Severity: {severity}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Diagnosis",        decision.get("diagnosis", "N/A"))
        st.metric("Confidence",       f"{decision.get('confidence', 0):.0%}")
        st.metric("Referral Urgency", decision.get("referral_urgency", "N/A").upper())
    with res_col2:
        st.metric("Tier",             f"Tier {decision.get('tier', result.get('tier', '?'))}")
        st.metric("Rule Applied",     decision.get("rule_applied", "N/A"))
        st.metric("Agent Agreement",  decision.get("agents_agreement", "N/A"))

    st.markdown("**Clinical Reasoning**")
    st.info(decision.get("reasoning", "No reasoning available."))

    st.markdown("**Recommended Action**")
    st.success(decision.get("recommended_action", "Consult a healthcare professional."))

    # Agent details
    with st.expander("Agent Details", expanded=False):
        ag_col1, ag_col2, ag_col3 = st.columns(3)

        with ag_col1:
            st.markdown("**COPD Agent**")
            copd_res = result.get("copd_result", {})
            st.write(f"Prediction: **{copd_res.get('label', 'N/A')}**")
            st.write(f"Probability: {copd_res.get('copd_probability', 0):.2%}")
            st.write(f"Confidence: {copd_res.get('confidence', 0):.2%}")

        with ag_col2:
            st.markdown("**Pneumonia Agent**")
            pneu_res = result.get("pneumonia_result", {})
            st.write(f"Prediction: **{pneu_res.get('label', 'N/A')}**")
            st.write(f"Probability: {pneu_res.get('pneumonia_probability', 0):.2%}")
            st.write(f"Confidence: {pneu_res.get('confidence', 0):.2%}")

        with ag_col3:
            st.markdown("**Sound Agent**")
            snd_res = result.get("sound_result", {})
            if snd_res and not snd_res.get("skipped"):
                st.write(f"Sound Type: **{snd_res.get('sound_type', 'N/A')}**")
                st.write(f"Confidence: {snd_res.get('confidence', 0):.2%}")
                if snd_res.get("all_probabilities"):
                    st.json(snd_res["all_probabilities"])
            else:
                st.write("Not available (Tier 1 mode)")

    # Session / deterioration alerts
    alerts = result.get("session_alerts", {})
    if alerts.get("deterioration_detected"):
        st.warning(f"Deterioration detected over {alerts.get('sessions_analyzed', '?')} sessions. "
                   f"Slope: {alerts.get('slope', 0):.3f}")

    with st.expander("Raw JSON Response", expanded=False):
        st.json(result)

elif run_disabled:
    if audio_file is None:
        st.warning("Please upload an audio recording to proceed.")
    elif is_tier2 and lung_file is None:
        st.warning("Please upload both an audio recording and a lung sound recording for Tier 2.")
