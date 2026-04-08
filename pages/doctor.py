"""
pages/doctor.py — Doctor portal: patient list.

Shows all registered patients with their latest severity,
COPD/Pneumonia risk, and session count. Doctor clicks a patient
to go to the Tier 2 detail page.
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Doctor Portal — Respiratory Triage AI",
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

from database.auth_store    import AuthStore
from database.session_store import SessionStore

auth_store    = AuthStore()
session_store = SessionStore()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### Dr. {user.get('full_name') or user['username']}")
    st.caption("Doctor Portal")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("- Patient List (current)")
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.pop("selected_patient_id", None)
        st.switch_page("app.py")
    st.caption("Powered by OPERA-CT + LangGraph")
    st.caption("For research purposes only.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Patient Dashboard")
st.markdown("Select a patient to view their history and run a Tier 2 clinical assessment.")
st.markdown("---")

patients = auth_store.get_all_patients()

if not patients:
    st.info("No patients registered yet.")
else:
    # Build enriched table with latest session data
    rows = []
    for p in patients:
        pid = f"patient_{p['id']}"
        latest = session_store.get_latest_session(pid)
        total  = len(session_store.get_sessions(pid, n=100))
        alerts = session_store.check_deterioration(pid)

        rows.append({
            "id":            p["id"],
            "patient_id":    pid,
            "Name":          p["full_name"] or p["username"],
            "Username":      p["username"],
            "Age":           p.get("age") or "—",
            "Gender":        (p.get("gender") or "—").capitalize(),
            "Sessions":      total,
            "Last Severity": latest["severity"] if latest else "—",
            "Risk Score":    f"{latest.get('longitudinal_score',0):.0%}" if latest else "—",
            "COPD Risk":     f"{latest['copd_confidence']:.0%}" if latest else "—",
            "Pneu Risk":     f"{latest['pneu_confidence']:.0%}" if latest else "—",
            "Voice Index":   f"{latest.get('voice_index',0):.0%}" if latest else "—",
            "Last Visit":    (latest["timestamp"][:10]) if latest else "Never",
            "Alert":         "YES" if alerts else "—",
        })

    df = pd.DataFrame(rows)

    # Search/filter
    search = st.text_input("Search patient by name or username", "")
    if search:
        mask = (df["Name"].str.contains(search, case=False) |
                df["Username"].str.contains(search, case=False))
        df = df[mask]

    # Colour-code severity — use border accent, keep text always dark
    severity_border = {"HIGH": "#EF5350", "CRITICAL": "#EF5350",
                       "MODERATE": "#FFA726", "LOW": "#66BB6A", "—": "#90A4AE"}

    st.subheader(f"Registered Patients ({len(df)})")

    for _, row in df.iterrows():
        border = severity_border.get(row["Last Severity"], "#90A4AE")
        alert_badge = "  🔴 DETERIORATION ALERT" if row["Alert"] == "YES" else ""

        with st.container():
            st.markdown(
                f'<div style="border-left:5px solid {border};padding:12px 16px;'
                f'border-radius:6px;margin-bottom:8px;border-top:1px solid #eee;'
                f'border-right:1px solid #eee;border-bottom:1px solid #eee;">'
                f'<span style="font-size:1.05em;font-weight:600;color:#212121;">'
                f'{row["Name"]}</span> '
                f'<span style="color:#555;">(@{row["Username"]})</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<span style="color:#333;">Age: {row["Age"]} &nbsp; {row["Gender"]}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Sessions: {row["Sessions"]}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Last visit: {row["Last Visit"]}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Severity: <b style="color:#212121;">{row["Last Severity"]}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Risk: <b>{row["Risk Score"]}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;COPD: <b>{row["COPD Risk"]}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Pneu: <b>{row["Pneu Risk"]}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Voice: <b>{row["Voice Index"]}</b>'
                f'</span>'
                f'<span style="color:#c62828;font-weight:600;">{alert_badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"View Patient — {row['Name']}", key=f"btn_{row['id']}"):
                st.session_state.selected_patient_id  = row["id"]
                st.session_state.selected_patient_pid = row["patient_id"]
                st.session_state.selected_patient_name = row["Name"]
                st.switch_page("pages/doctor_patient.py")
