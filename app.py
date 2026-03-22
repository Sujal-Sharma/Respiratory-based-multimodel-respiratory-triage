"""
app.py — Streamlit web app for Multimodal Respiratory Triage.

Two-tier interface:
  Tier 1 (Patient): Cough recording + symptom form -> preliminary risk screening
  Tier 2 (Clinician): + Lung recording via stethoscope -> full triage diagnosis

Run:  streamlit run app.py
"""

import os
import json
import tempfile
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Respiratory Triage AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Respiratory Triage AI")
    st.markdown("---")

    tier = st.radio(
        "Select Tier",
        ["Tier 1 -- Patient Self-Screening", "Tier 2 -- Clinician Confirmation"],
        help="Tier 1: cough + symptoms only. Tier 2: adds lung sound analysis."
    )
    is_tier2 = "Tier 2" in tier

    st.markdown("---")
    st.markdown("**How it works**")
    if not is_tier2:
        st.info(
            "Upload a cough recording and fill in symptoms. "
            "The AI will screen for respiratory risk using cough analysis "
            "and symptom classification."
        )
    else:
        st.info(
            "Upload cough + lung recordings and fill in symptoms. "
            "The AI will perform full triage using all 3 specialist agents "
            "plus LLM clinical reasoning."
        )

    st.markdown("---")
    st.caption("Powered by LangGraph + Groq (Llama 3.3 70B)")
    st.caption("For research purposes only. Not a medical device.")


# ── Main content ────────────────────────────────────────────────────────────

st.header("Tier 2 -- Clinician Confirmation" if is_tier2 else "Tier 1 -- Patient Self-Screening")

col_audio, col_symptoms = st.columns([1, 1])

# ── Audio uploads ───────────────────────────────────────────────────────────

with col_audio:
    st.subheader("Audio Recordings")

    cough_file = st.file_uploader(
        "Upload Cough Recording",
        type=["wav", "mp3", "webm", "ogg", "m4a", "flac", "mp4", "3gp", "aac"],
        help="Record a 3-5 second cough using your phone or microphone."
    )

    if cough_file:
        st.audio(cough_file, format="audio/wav")

    lung_file = None
    if is_tier2:
        st.markdown("---")
        lung_file = st.file_uploader(
            "Upload Lung Sound Recording",
            type=["wav", "mp3", "webm", "ogg", "m4a", "flac", "mp4", "3gp", "aac"],
            help="Record lung sounds using a stethoscope (8-15 seconds)."
        )
        if lung_file:
            st.audio(lung_file, format="audio/wav")

# ── Symptom form ────────────────────────────────────────────────────────────

with col_symptoms:
    st.subheader("Patient Information")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with c2:
        gender = st.selectbox("Gender", ["male", "female", "other"])

    st.markdown("**Symptoms**")
    c1, c2 = st.columns(2)
    with c1:
        fever = st.checkbox("Fever / Muscle Pain")
        dyspnea = st.checkbox("Difficulty Breathing")
        wheezing = st.checkbox("Wheezing")
    with c2:
        resp_cond = st.checkbox("Known Respiratory Condition")
        congestion = st.checkbox("Nasal Congestion")
        cough_severity = st.slider("Cough Severity", 0.0, 1.0, 0.5, 0.1,
                                   help="0 = no cough, 1 = severe cough")

    # Build symptoms list for display
    symptom_list = []
    if fever:
        symptom_list.append("fever/muscle pain")
    if dyspnea:
        symptom_list.append("difficulty breathing")
    if wheezing:
        symptom_list.append("wheezing")
    if resp_cond:
        symptom_list.append("respiratory condition")
    if congestion:
        symptom_list.append("nasal congestion")

# ── Run triage ──────────────────────────────────────────────────────────────

st.markdown("---")

run_disabled = cough_file is None
if is_tier2 and lung_file is None:
    run_disabled = True

run_label = "Run Full Triage" if is_tier2 else "Screen for Risk"
if st.button(run_label, type="primary", disabled=run_disabled, use_container_width=True):

    # Save uploaded files to temp paths
    with tempfile.TemporaryDirectory() as tmpdir:
        cough_path = os.path.join(tmpdir, f"cough.{cough_file.name.split('.')[-1]}")
        with open(cough_path, "wb") as f:
            f.write(cough_file.getbuffer())

        lung_path = ""
        if is_tier2 and lung_file:
            lung_path = os.path.join(tmpdir, f"lung.{lung_file.name.split('.')[-1]}")
            with open(lung_path, "wb") as f:
                f.write(lung_file.getbuffer())

        patient_info = {
            "age": age,
            "gender": gender,
            "symptoms": symptom_list,
            "fever_muscle_pain": fever,
            "respiratory_condition": resp_cond,
            "cough_detected": cough_severity,
            "dyspnea": dyspnea,
            "wheezing": wheezing,
            "congestion": congestion,
        }

        with st.spinner("Running triage pipeline..."):
            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info, cough_path, lung_path)

    # ── Display results ─────────────────────────────────────────────────

    decision = result.get("triage_decision", {})
    severity = decision.get("severity", "UNKNOWN")

    severity_colors = {
        "LOW": "green",
        "MODERATE": "orange",
        "HIGH": "red",
        "CRITICAL": "red",
    }
    color = severity_colors.get(severity, "gray")

    st.markdown("---")
    st.subheader("Triage Decision")

    # Severity banner
    st.markdown(
        f'<div style="background-color:{color}; padding:16px; border-radius:8px; '
        f'text-align:center; margin-bottom:16px;">'
        f'<h2 style="color:white; margin:0;">Severity: {severity}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Main results
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Diagnosis", decision.get("diagnosis", "N/A"))
        st.metric("Confidence", f"{decision.get('confidence', 0):.0%}")
        st.metric("Referral Urgency", decision.get("referral_urgency", "N/A").upper())

    with res_col2:
        st.metric("Tier", f"Tier {decision.get('tier', result.get('tier', '?'))}")
        st.metric("LLM Provider", decision.get("llm_provider", "N/A").capitalize())
        st.metric("Agent Agreement", decision.get("agents_agreement", "N/A"))

    # Reasoning
    st.markdown("**Clinical Reasoning**")
    st.info(decision.get("reasoning", "No reasoning available."))

    st.markdown("**Recommended Action**")
    st.success(decision.get("recommended_action", "Consult a healthcare professional."))

    # Agent details (expandable)
    with st.expander("Agent Details", expanded=False):
        agent_col1, agent_col2, agent_col3 = st.columns(3)

        with agent_col1:
            st.markdown("**Cough Agent (LightCoughCNN)**")
            cough_res = result.get("cough_result", {})
            label = cough_res.get("label", cough_res.get("top_prediction", "N/A"))
            conf = cough_res.get("confidence", 0)
            st.write(f"Prediction: **{label}**")
            st.write(f"Confidence: {conf:.2%}")
            probs = cough_res.get("probabilities", cough_res.get("scores", {}))
            if probs:
                st.json(probs)

        with agent_col2:
            st.markdown("**Symptom Agent (XGBoost)**")
            sym_res = result.get("symptom_result", {})
            st.write(f"Prediction: **{sym_res.get('label', 'N/A')}**")
            st.write(f"Confidence: {sym_res.get('confidence', 0):.2%}")
            if sym_res.get("probabilities"):
                st.json(sym_res["probabilities"])

        with agent_col3:
            st.markdown("**Lung Agent (MultiTaskEfficientNet)**")
            lung_res = result.get("lung_result", {})
            if lung_res and "disease" in lung_res:
                st.write(f"Disease: **{lung_res['disease']['label']}** "
                         f"({lung_res['disease']['confidence']:.2%})")
                st.write(f"Sound: **{lung_res['sound']['label']}** "
                         f"({lung_res['sound']['confidence']:.2%})")
                if lung_res['disease'].get('probabilities'):
                    st.json(lung_res['disease']['probabilities'])
            else:
                st.write("Not available (Tier 1 mode)")

    # Raw JSON (expandable)
    with st.expander("Raw JSON Response", expanded=False):
        st.json(decision)

elif run_disabled:
    if cough_file is None:
        st.warning("Please upload a cough recording to proceed.")
    elif is_tier2 and lung_file is None:
        st.warning("Please upload both cough and lung recordings for Tier 2.")
