"""
pages/patient.py — Patient portal.

Sections:
  1. Self-Screen  — Tier 1: CAT-style symptoms + vowel recording + optional cough
  2. My History   — trend chart (symptom/voice/longitudinal) + session log + alerts
  3. My Profile   — view/edit medical profile
"""

import os, sys, tempfile
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Patient Portal — Respiratory Triage AI",
    page_icon="🫁", layout="wide", initial_sidebar_state="expanded",
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if "user" not in st.session_state or not st.session_state.user:
    st.switch_page("app.py")
user = st.session_state.user
if user.get("role") != "patient":
    st.switch_page("pages/doctor.py")

from database.auth_store    import AuthStore
from database.session_store import SessionStore
from pipeline.longitudinal  import interpret_score

auth_store    = AuthStore()
session_store = SessionStore()
patient_id    = f"patient_{user['id']}"
profile       = auth_store.get_profile(user["id"]) or {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {user.get('full_name') or user['username']}")
    st.caption(f"ID: {patient_id}")
    st.markdown("---")
    page = st.radio("Navigation", ["Self-Screen", "My History", "My Profile"])
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.switch_page("app.py")
    st.caption("Powered by OPERA-CT + LangGraph")
    st.caption("For research purposes only.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Self-Screen
# ══════════════════════════════════════════════════════════════════════════════
if page == "Self-Screen":
    st.title("Symptom Self-Screening")
    st.markdown("---")

    col_form, col_voice = st.columns([3, 2])

    with col_form:
        st.subheader("Patient Information")
        c1, c2 = st.columns(2)
        with c1:
            age    = st.number_input("Age", 1, 120, int(profile.get("age") or 25))
        with c2:
            gender = st.selectbox("Gender", ["male","female","other"],
                                  index=["male","female","other"].index(
                                      profile.get("gender","male")))

        st.markdown("**Symptoms**")
        c1, c2 = st.columns(2)
        with c1:
            fever    = st.checkbox("Fever / Muscle Pain")
            wheezing = st.checkbox("Wheezing")
            resp_cond= st.checkbox("Known Respiratory Condition",
                                   value=bool(profile.get("respiratory_condition",0)))
        with c2:
            congestion = st.checkbox("Nasal Congestion")
            cough_sev  = st.slider("Cough Severity", 0.0, 1.0, 0.3, 0.1,
                                   help="0=none, 1=severe/persistent")

        st.markdown("**Breathing & Functional Status (mMRC + CAT)**")
        dyspnea_level = st.select_slider(
            "Breathlessness Level (mMRC scale)",
            options=[0,1,2,3,4],
            value=0,
            format_func=lambda x: {
                0: "0 — None",
                1: "1 — Only on strenuous activity",
                2: "2 — Walking fast or uphill",
                3: "3 — Stops after 100m on flat",
                4: "4 — Too breathless to leave house",
            }[x]
        )
        dyspnea = dyspnea_level >= 2

        c1, c2, c3 = st.columns(3)
        with c1:
            chest_tightness = st.slider("Chest Tightness", 0, 4, 0,
                                        help="0=none, 4=very tight")
        with c2:
            sleep_quality = st.slider("Sleep Quality (4=very poor)", 0, 4, 0)
        with c3:
            energy_level  = st.slider("Energy Level (4=no energy)", 0, 4, 0)

        sputum = st.select_slider(
            "Sputum / Phlegm",
            options=[0,1,2,3],
            value=0,
            format_func=lambda x: {0:"None",1:"Clear",2:"Coloured",3:"Thick/Dark"}[x]
        )

    with col_voice:
        st.subheader("Voice & Cough Recordings")
        st.info(
            "**Vowel recording (recommended)**\n\n"
            "Say **'Ahhh'** clearly for 5 seconds into your microphone. "
            "This captures voice biomarkers (jitter, shimmer, HNR) for "
            "longitudinal respiratory monitoring.\n\n"
            "**First session** — saves your personal baseline.\n"
            "**Future sessions** — measures acoustic change from baseline."
        )
        vowel_file = st.file_uploader(
            "Upload Vowel Recording ('Ahhh' — 5 sec)",
            type=["wav","mp3","webm","ogg","m4a","flac","mp4"],
            key="vowel_upload",
        )
        if vowel_file:
            st.audio(vowel_file)
            st.success("Vowel recording ready.")

        st.markdown("---")
        st.caption("Optional: upload a cough recording for acoustic drift tracking.")
        cough_file = st.file_uploader(
            "Cough Recording (optional)",
            type=["wav","mp3","webm","ogg","m4a","flac","mp4"],
            key="cough_upload",
        )
        if cough_file:
            st.audio(cough_file)

    st.markdown("---")
    symptom_list = []
    if fever:      symptom_list.append("fever/muscle pain")
    if dyspnea:    symptom_list.append("difficulty breathing")
    if wheezing:   symptom_list.append("wheezing")
    if resp_cond:  symptom_list.append("respiratory condition")
    if congestion: symptom_list.append("nasal congestion")

    if st.button("Run Screening", type="primary", use_container_width=True):
        patient_info = {
            "age": age, "gender": gender,
            "symptoms": symptom_list,
            "fever_muscle_pain":     fever,
            "respiratory_condition": resp_cond,
            "cough_detected":        cough_sev,
            "cough_severity":        cough_sev * 10,
            "dyspnea":               dyspnea,
            "dyspnea_level":         dyspnea_level,
            "wheezing":              wheezing,
            "congestion":            congestion,
            "chest_tightness":       chest_tightness,
            "sleep_quality":         sleep_quality,
            "energy_level":          energy_level,
            "sputum":                sputum,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            vowel_path = ""
            cough_path = ""

            if vowel_file:
                vowel_path = os.path.join(tmpdir,
                    f"vowel.{vowel_file.name.split('.')[-1]}")
                with open(vowel_path, "wb") as f:
                    f.write(vowel_file.getbuffer())

            if cough_file:
                cough_path = os.path.join(tmpdir,
                    f"cough.{cough_file.name.split('.')[-1]}")
                with open(cough_path, "wb") as f:
                    f.write(cough_file.getbuffer())

            with st.spinner("Analysing symptoms and voice biomarkers..."):
                from pipeline.triage_graph import run_triage
                result = run_triage(
                    patient_info,
                    cough_audio_path = cough_path,
                    lung_audio_path  = "",
                    vowel_audio_path = vowel_path,
                    patient_id       = patient_id,
                )

        decision   = result.get("triage_decision", {})
        severity   = decision.get("severity", "UNKNOWN")
        long_score = result.get("longitudinal_score", 0.0)
        interp     = interpret_score(long_score)

        sev_colors = {"LOW":"green","MODERATE":"orange","HIGH":"red","CRITICAL":"red"}
        color = sev_colors.get(severity, "gray")

        st.markdown("---")
        st.subheader("Screening Result")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div style="background:{color};padding:16px;border-radius:8px;'
                f'text-align:center;">'
                f'<h2 style="color:white;margin:0;">Severity: {severity}</h2></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="background:{interp["color"]};padding:16px;'
                f'border-radius:8px;text-align:center;">'
                f'<h3 style="color:white;margin:0;">Risk Score: {long_score:.0%}</h3>'
                f'<p style="color:white;margin:4px 0 0 0;">{interp["label"]}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis",  decision.get("diagnosis","N/A"))
        c2.metric("Referral",   decision.get("referral_urgency","none").upper())
        c3.metric("Action",     decision.get("recommended_action","—"))

        st.markdown("**Clinical Reasoning**")
        st.info(decision.get("reasoning","—"))

        # Signal breakdown
        with st.expander("Signal Breakdown", expanded=True):
            sc1, sc2, sc3 = st.columns(3)
            sym = result.get("symptom_result", {})
            vr  = result.get("voice_result",   {})

            sc1.metric("Symptom Index",
                       f"{result.get('symptom_index',0):.0%}",
                       help="CAT-style validated symptom severity score")
            sc2.metric("Voice Health Index",
                       f"{result.get('voice_index',0):.0%}",
                       help="Acoustic deviation from your personal baseline (0=stable)")
            sc3.metric("Cough Drift",
                       f"{result.get('drift_score',0):.0%}",
                       help="OPERA-CT embedding drift from baseline cough")

            if vr.get("features"):
                feat = vr["features"]
                st.markdown("**Voice Biomarkers**")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Jitter",   f"{feat.get('jitter',0):.4f}",
                           help="F0 cycle-to-cycle variation")
                fc2.metric("Shimmer",  f"{feat.get('shimmer',0):.4f}",
                           help="Amplitude variation")
                fc3.metric("HNR (dB)", f"{feat.get('hnr',0):.1f}",
                           help="Harmonics-to-noise ratio")
                fc4.metric("Duration", f"{feat.get('phonation_duration',0):.1f}s",
                           help="Phonation duration")

            if vr.get("is_baseline"):
                st.info("First session — voice baseline saved. "
                        "Future sessions will track changes from today.")
            elif vr.get("error"):
                st.warning(f"Voice analysis: {vr['error']}")

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
    history = session_store.get_sessions(patient_id, n=30)

    if not history:
        st.info("No sessions yet. Run a Self-Screen first.")
    else:
        df = pd.DataFrame(history)

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sessions",   len(df))
        c2.metric("Latest Severity",  df.iloc[0]["severity"])
        c3.metric("Latest Risk Score",f"{df.iloc[0].get('longitudinal_score',0):.0%}")
        c4.metric("Avg COPD Risk",    f"{df['copd_confidence'].mean():.0%}")
        c5.metric("Avg Pneu Risk",    f"{df['pneu_confidence'].mean():.0%}")

        # Longitudinal trend chart
        st.subheader("Longitudinal Risk Trend")
        df_plot = df.iloc[::-1].reset_index(drop=True)
        df_plot["session"] = range(1, len(df_plot)+1)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(df_plot["session"],
                        df_plot.get("longitudinal_score", 0),
                        alpha=0.15, color="#7E57C2")
        ax.plot(df_plot["session"], df_plot.get("longitudinal_score",
                pd.Series([0]*len(df_plot))),
                marker='D', color="#7E57C2", lw=2, label="Longitudinal Score")
        ax.plot(df_plot["session"], df_plot.get("symptom_index",
                pd.Series([0]*len(df_plot))),
                marker='o', color="#EF5350", lw=1.5, ls='--', label="Symptom Index")
        ax.plot(df_plot["session"], df_plot.get("voice_index",
                pd.Series([0]*len(df_plot))),
                marker='s', color="#42A5F5", lw=1.5, ls='--', label="Voice Index")

        ax.axhline(0.05, color='gray', ls=':', alpha=0.5, label="MCID (0.05)")
        ax.axhline(0.40, color='orange', ls='--', alpha=0.4, label="Moderate threshold")
        ax.axhline(0.60, color='red', ls='--', alpha=0.4, label="High threshold")

        # Mark Tier 2 sessions
        t2 = df_plot[df_plot["tier"] == 2]
        if not t2.empty:
            ax.scatter(t2["session"], t2.get("longitudinal_score",
                       pd.Series([0]*len(t2))),
                       marker='*', s=150, color='darkred', zorder=5,
                       label="Tier 2 (doctor)")

        ax.set_xlabel("Session"); ax.set_ylabel("Score (0–1)")
        ax.set_title("Longitudinal Respiratory Risk Monitoring")
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3); ax.set_ylim(-0.02, 1.05)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Deterioration alerts
        alerts = session_store.check_deterioration(patient_id)
        if alerts:
            for alert in alerts:
                st.error(f"Deterioration Alert — {alert['message']}")

        # Session table
        st.subheader("Session Log")
        show_cols = ["timestamp","tier","severity","diagnosis",
                     "longitudinal_score","symptom_index","voice_index",
                     "drift_score","copd_confidence","pneu_confidence",
                     "sound_type","action"]
        show_cols = [c for c in show_cols if c in df.columns]
        disp = df[show_cols].copy()
        for col in ["longitudinal_score","symptom_index","voice_index",
                    "drift_score","copd_confidence","pneu_confidence"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:.0%}")
        disp.columns = [c.replace("_"," ").title() for c in disp.columns]
        st.dataframe(disp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: My Profile
# ══════════════════════════════════════════════════════════════════════════════
elif page == "My Profile":
    st.title("My Profile")
    st.markdown("---")

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            age    = st.number_input("Age", 1, 120, int(profile.get("age") or 25))
            gender = st.selectbox("Gender", ["male","female","other"],
                                  index=["male","female","other"].index(
                                      profile.get("gender","male")))
        with c2:
            resp_cond = st.checkbox("Known Respiratory Condition",
                                    value=bool(profile.get("respiratory_condition",0)))
            smoking   = st.checkbox("Smoker / Ex-smoker",
                                    value=bool(profile.get("smoking",0)))
        notes = st.text_area("Medical Notes (optional)", value=profile.get("notes",""))

        if st.form_submit_button("Save Profile", type="primary"):
            auth_store.update_profile(user["id"], age, gender,
                                      resp_cond, smoking, notes)
            st.success("Profile saved.")
            st.rerun()

    # Show voice baseline status
    st.markdown("---")
    st.subheader("Voice Baseline Status")
    baseline = session_store.get_baseline(patient_id)
    if baseline and baseline.get("voice_features"):
        vf = baseline["voice_features"]
        st.success(f"Voice baseline established on {baseline['created_at'][:10]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Jitter",  f"{vf.get('jitter',0):.4f}")
        c2.metric("Baseline Shimmer", f"{vf.get('shimmer',0):.4f}")
        c3.metric("Baseline HNR",     f"{vf.get('hnr',0):.1f} dB")
        if st.button("Reset Voice Baseline", type="secondary"):
            session_store.save_baseline(patient_id, {}, None)
            st.warning("Baseline reset. Next screening will set a new baseline.")
            st.rerun()
    else:
        st.info("No voice baseline yet. Record a vowel ('Ahhh') "
                "during your next Self-Screen to establish your personal baseline.")
