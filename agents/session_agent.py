"""
agents/session_agent.py — Longitudinal session memory agent.

Stores every triage result and autonomously monitors for deterioration
trends across sessions. This is the agentic memory component —
it triggers alerts without being asked.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.session_store import SessionStore, DB_PATH

logger = logging.getLogger(__name__)


class SessionAgent:
    """
    Agentic memory component.

    Records each triage session to SQLite and checks for deteriorating
    confidence trends using linear regression. Returns deterioration
    alerts if a patient is getting worse over multiple sessions.
    """

    AGENT_NAME = 'Session Memory Agent'

    def __init__(self, db_path: str = DB_PATH):
        self.store = SessionStore(db_path)

    def record_and_check(self,
                         patient_id: str,
                         triage_result: dict,
                         copd_conf: float,
                         pneu_conf: float,
                         tier: int = 1,
                         sound_type: str = 'Normal',
                         cough_severity: int = 0) -> dict:
        """
        Record this session and check for deterioration trend.

        Parameters
        ----------
        patient_id     : unique patient identifier
        triage_result  : final triage decision dict from rule_engine
        copd_conf      : COPD probability from COPDAgent
        pneu_conf      : Pneumonia probability from PneumoniaAgent
        tier           : 1 (patient self-screen) or 2 (clinical)
        sound_type     : sound classification result
        cough_severity : 0-10

        Returns
        -------
        dict with:
            agent, session_saved, deterioration_alerts,
            session_history, total_sessions
        """
        self.store.save_session(
            patient_id=patient_id,
            triage_result=triage_result,
            copd_conf=copd_conf,
            pneu_conf=pneu_conf,
            tier=tier,
            sound_type=sound_type,
            cough_severity=cough_severity,
        )

        alerts  = self.store.check_deterioration(patient_id)
        history = self.store.get_sessions(patient_id, n=10)

        if alerts:
            for alert in alerts:
                logger.warning("Deterioration alert", extra={'alert_message': alert.get('message', '')})

        return {
            'agent':                self.AGENT_NAME,
            'session_saved':        True,
            'deterioration_alerts': alerts,
            'session_history':      history,
            'total_sessions':       len(history),
        }

    def get_history(self, patient_id: str, n: int = 10) -> list:
        """Retrieve session history for a patient."""
        return self.store.get_sessions(patient_id, n=n)
