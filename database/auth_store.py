"""
database/auth_store.py — User authentication and profile storage.

Manages patient and doctor accounts in SQLite.
Passwords are hashed with sha-256 + salt.

Tables:
  users    — account credentials + role (patient / doctor)
  profiles — patient medical profile (age, gender, conditions)
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime

DB_PATH = './data/sessions.db'


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


class AuthStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    username     TEXT    NOT NULL UNIQUE,
                    password_hash TEXT   NOT NULL,
                    salt         TEXT    NOT NULL,
                    role         TEXT    NOT NULL DEFAULT 'patient',
                    full_name    TEXT    DEFAULT '',
                    created_at   TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    user_id      INTEGER PRIMARY KEY,
                    age          INTEGER DEFAULT 0,
                    gender       TEXT    DEFAULT 'male',
                    respiratory_condition INTEGER DEFAULT 0,
                    smoking      INTEGER DEFAULT 0,
                    notes        TEXT    DEFAULT '',
                    updated_at   TEXT    NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            conn.commit()

        # Create default doctor account if none exists
        if not self.get_user_by_username('doctor'):
            self.register_user('doctor', 'doctor123', 'doctor', 'Dr. Admin')

    def register_user(self, username: str, password: str,
                      role: str = 'patient', full_name: str = '') -> dict:
        """Register a new user. Returns dict with success/error."""
        salt = secrets.token_hex(16)
        pw_hash = _hash_password(password, salt)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO users (username, password_hash, salt, role, full_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username.strip().lower(), pw_hash, salt, role,
                      full_name.strip(), datetime.now().isoformat()))
                user_id = cursor.lastrowid

                if role == 'patient':
                    conn.execute("""
                        INSERT INTO patient_profiles (user_id, updated_at)
                        VALUES (?, ?)
                    """, (user_id, datetime.now().isoformat()))

                conn.commit()
            return {'success': True, 'user_id': user_id}
        except sqlite3.IntegrityError:
            return {'success': False, 'error': 'Username already exists'}

    def login(self, username: str, password: str) -> dict | None:
        """Verify credentials. Returns user dict or None."""
        user = self.get_user_by_username(username)
        if not user:
            return None
        pw_hash = _hash_password(password, user['salt'])
        if pw_hash == user['password_hash']:
            return {k: v for k, v in user.items()
                    if k not in ('password_hash', 'salt')}
        return None

    def get_user_by_username(self, username: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username.strip().lower(),)
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

    def get_user_by_id(self, user_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

    def get_profile(self, user_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM patient_profiles WHERE user_id = ?", (user_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

    def update_profile(self, user_id: int, age: int, gender: str,
                       respiratory_condition: bool, smoking: bool,
                       notes: str = '') -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE patient_profiles
                SET age=?, gender=?, respiratory_condition=?,
                    smoking=?, notes=?, updated_at=?
                WHERE user_id=?
            """, (age, gender, int(respiratory_condition), int(smoking),
                  notes, datetime.now().isoformat(), user_id))
            conn.commit()
        return True

    def get_all_patients(self) -> list:
        """Get all patient accounts with their profiles (for doctor view)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT u.id, u.username, u.full_name, u.created_at,
                       p.age, p.gender, p.respiratory_condition, p.smoking, p.notes
                FROM users u
                LEFT JOIN patient_profiles p ON u.id = p.user_id
                WHERE u.role = 'patient'
                ORDER BY u.full_name
            """)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
