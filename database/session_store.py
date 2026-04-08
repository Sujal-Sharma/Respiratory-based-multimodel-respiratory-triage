"""
database/session_store.py — SQLite-based longitudinal session storage.

Stores every triage result per patient and detects deterioration trends
over time using linear regression on confidence scores.

The SessionAgent autonomously triggers alerts when a worsening trend
is detected — no human has to ask for this check.
"""

import os
import sqlite3
import json
from datetime import datetime

import numpy as np

DB_PATH = './data/sessions.db'


class SessionStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id          TEXT    NOT NULL,
                    timestamp           TEXT    NOT NULL,
                    tier                INTEGER DEFAULT 1,
                    copd_confidence     REAL    DEFAULT 0.0,
                    pneu_confidence     REAL    DEFAULT 0.0,
                    severity            TEXT    DEFAULT 'LOW',
                    diagnosis           TEXT    DEFAULT '',
                    sound_type          TEXT    DEFAULT 'Normal',
                    cough_severity      REAL    DEFAULT 0.0,
                    action              TEXT    DEFAULT '',
                    symptom_index       REAL    DEFAULT 0.0,
                    voice_index         REAL    DEFAULT 0.0,
                    drift_score         REAL    DEFAULT 0.0,
                    longitudinal_score  REAL    DEFAULT 0.0
                )
            """)
            # Add new columns to existing DB if upgrading
            for col, typedef in [
                ("symptom_index",      "REAL DEFAULT 0.0"),
                ("voice_index",        "REAL DEFAULT 0.0"),
                ("drift_score",        "REAL DEFAULT 0.0"),
                ("longitudinal_score", "REAL DEFAULT 0.0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # column already exists

            # Patient baselines table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_baselines (
                    patient_id          TEXT    PRIMARY KEY,
                    voice_features_json TEXT    DEFAULT '{}',
                    cough_embedding     BLOB,
                    created_at          TEXT    NOT NULL,
                    updated_at          TEXT    NOT NULL
                )
            """)
            conn.commit()

    # ── Baseline management ───────────────────────────────────────────────────

    def get_baseline(self, patient_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM patient_baselines WHERE patient_id = ?",
                (patient_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            d = dict(zip(cols, row))
            d['voice_features'] = json.loads(d.get('voice_features_json') or '{}')
            if d.get('cough_embedding'):
                d['cough_embedding'] = np.frombuffer(d['cough_embedding'],
                                                     dtype=np.float32)
            return d

    def save_baseline(self, patient_id: str,
                      voice_features: dict,
                      cough_embedding: np.ndarray | None = None):
        now = datetime.now().isoformat()
        emb_blob = cough_embedding.astype(np.float32).tobytes() \
            if cough_embedding is not None else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO patient_baselines
                    (patient_id, voice_features_json, cough_embedding,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    voice_features_json = excluded.voice_features_json,
                    cough_embedding     = excluded.cough_embedding,
                    updated_at          = excluded.updated_at
            """, (patient_id, json.dumps(voice_features), emb_blob, now, now))
            conn.commit()

    # ── Session management ────────────────────────────────────────────────────

    def save_session(self,
                     patient_id: str,
                     triage_result: dict,
                     copd_conf: float,
                     pneu_conf: float,
                     tier: int = 1,
                     sound_type: str = 'Normal',
                     cough_severity: float = 0.0,
                     symptom_index: float = 0.0,
                     voice_index: float = 0.0,
                     drift_score: float = 0.0,
                     longitudinal_score: float = 0.0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions
                    (patient_id, timestamp, tier, copd_confidence, pneu_confidence,
                     severity, diagnosis, sound_type, cough_severity, action,
                     symptom_index, voice_index, drift_score, longitudinal_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                datetime.now().isoformat(),
                tier,
                round(copd_conf, 4),
                round(pneu_conf, 4),
                triage_result.get('severity', 'LOW'),
                triage_result.get('diagnosis', ''),
                sound_type,
                round(float(cough_severity), 4),
                triage_result.get('recommended_action', ''),
                round(float(symptom_index), 4),
                round(float(voice_index), 4),
                round(float(drift_score), 4),
                round(float(longitudinal_score), 4),
            ))
            conn.commit()

    def get_sessions(self, patient_id: str, n: int = 10) -> list:
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

    def get_latest_session(self, patient_id: str) -> dict | None:
        sessions = self.get_sessions(patient_id, n=1)
        return sessions[0] if sessions else None

    def get_all_patient_ids(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT patient_id FROM sessions ORDER BY patient_id"
            )
            return [row[0] for row in cursor.fetchall()]

    def check_deterioration(self,
                            patient_id: str,
                            window: int = 5,
                            slope_threshold: float = 0.05,
                            conf_threshold: float = 0.45) -> list | None:
        sessions = self.get_sessions(patient_id, n=window)
        if len(sessions) < 3:
            return None

        sessions = list(reversed(sessions))
        alerts = []

        # Check COPD / Pneumonia audio confidence (Tier 2 sessions)
        for disease, col in [('COPD', 'copd_confidence'),
                              ('Pneumonia', 'pneu_confidence')]:
            confidences  = [s[col] for s in sessions]
            current_conf = confidences[-1]
            x     = np.arange(len(confidences), dtype=float)
            slope = float(np.polyfit(x, confidences, 1)[0])
            if slope > slope_threshold and current_conf > conf_threshold:
                alerts.append({
                    'disease':            disease,
                    'current_confidence': round(current_conf, 4),
                    'trend_slope':        round(slope, 4),
                    'sessions_analysed':  len(sessions),
                    'message': (
                        f"DETERIORATION ALERT: {disease} risk increasing. "
                        f"Current: {current_conf:.0%}. "
                        f"Rising +{slope:.3f}/session over {len(sessions)} sessions. "
                        "Urgent clinical review recommended."
                    ),
                })

        # Check longitudinal score trend (Tier 1 sessions)
        long_scores = [s.get('longitudinal_score', 0.0) for s in sessions]
        if any(s > 0 for s in long_scores):
            x     = np.arange(len(long_scores), dtype=float)
            slope = float(np.polyfit(x, long_scores, 1)[0])
            current = long_scores[-1]
            if slope > slope_threshold and current > conf_threshold:
                alerts.append({
                    'disease':            'Overall Health',
                    'current_confidence': round(current, 4),
                    'trend_slope':        round(slope, 4),
                    'sessions_analysed':  len(sessions),
                    'message': (
                        f"DETERIORATION ALERT: Overall respiratory risk increasing. "
                        f"Longitudinal score: {current:.0%}. "
                        f"Rising +{slope:.3f}/session. Doctor review recommended."
                    ),
                })

        return alerts if alerts else None
