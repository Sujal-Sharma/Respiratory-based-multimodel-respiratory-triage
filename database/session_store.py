"""
database/session_store.py — SQLite-based longitudinal session storage.

Stores every triage result per patient and detects deterioration trends
over time using linear regression on confidence scores.

The SessionAgent autonomously triggers alerts when a worsening trend
is detected — no human has to ask for this check.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np

DB_PATH = './data/sessions.db'


class SessionStore:
    """
    Persistent SQLite store for patient triage sessions.

    Each session records: patient_id, timestamp, tier, COPD confidence,
    Pneumonia confidence, severity, diagnosis, sound type, cough severity,
    and recommended action.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id      TEXT    NOT NULL,
                    timestamp       TEXT    NOT NULL,
                    tier            INTEGER DEFAULT 1,
                    copd_confidence REAL    DEFAULT 0.0,
                    pneu_confidence REAL    DEFAULT 0.0,
                    severity        TEXT    DEFAULT 'LOW',
                    diagnosis       TEXT    DEFAULT '',
                    sound_type      TEXT    DEFAULT 'Normal',
                    cough_severity  INTEGER DEFAULT 0,
                    action          TEXT    DEFAULT ''
                )
            """)
            conn.commit()

    def save_session(self,
                     patient_id: str,
                     triage_result: dict,
                     copd_conf: float,
                     pneu_conf: float,
                     tier: int = 1,
                     sound_type: str = 'Normal',
                     cough_severity: int = 0):
        """Insert a new triage session record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions
                    (patient_id, timestamp, tier, copd_confidence, pneu_confidence,
                     severity, diagnosis, sound_type, cough_severity, action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                datetime.now().isoformat(),
                tier,
                round(copd_conf, 4),
                round(pneu_conf, 4),
                triage_result.get('severity', 'LOW'),
                triage_result.get('diagnosis', ''),
                sound_type,
                cough_severity,
                triage_result.get('recommended_action', ''),
            ))
            conn.commit()

    def get_sessions(self, patient_id: str, n: int = 10) -> list:
        """
        Retrieve the N most recent sessions for a patient (newest first).
        Returns list of row dicts.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM sessions
                WHERE patient_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (patient_id, n))
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def check_deterioration(self,
                            patient_id: str,
                            window: int = 5,
                            slope_threshold: float = 0.05,
                            conf_threshold: float = 0.65) -> list | None:
        """
        Detect a deteriorating trend over the last `window` sessions.

        Uses linear regression on confidence scores. An alert is triggered
        when slope > slope_threshold AND current confidence > conf_threshold.

        Returns list of alert dicts, or None if no deterioration detected.
        """
        sessions = self.get_sessions(patient_id, n=window)
        if len(sessions) < 3:
            return None  # Need at least 3 data points for a meaningful trend

        # Reverse to chronological order (oldest first for regression)
        sessions = list(reversed(sessions))

        alerts = []
        for disease, col in [('COPD', 'copd_confidence'),
                              ('Pneumonia', 'pneu_confidence')]:
            confidences  = [s[col] for s in sessions]
            current_conf = confidences[-1]

            x     = np.arange(len(confidences), dtype=float)
            slope = float(np.polyfit(x, confidences, 1)[0])

            if slope > slope_threshold and current_conf > conf_threshold:
                alerts.append({
                    'disease':             disease,
                    'current_confidence':  round(current_conf, 4),
                    'trend_slope':         round(slope, 4),
                    'sessions_analysed':   len(sessions),
                    'message': (
                        f"DETERIORATION ALERT: {disease} risk increasing. "
                        f"Current confidence: {current_conf:.0%}. "
                        f"Rising +{slope:.3f} per session over "
                        f"{len(sessions)} sessions. "
                        "Urgent clinical review recommended."
                    ),
                })

        return alerts if alerts else None
