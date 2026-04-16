"""
server.py — Flask backend for RespiTriage AI
Replaces Streamlit. All ML/DB code untouched.

Run:  python server.py
      OR: flask --app server run --port 5000
"""

import os, sys, json, tempfile, base64
from functools import wraps
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_from_directory)
from flask.json.provider import DefaultJSONProvider
from utils.symptom_validator import validate_symptoms


class NumpyJSONProvider(DefaultJSONProvider):
    """Serialize numpy scalars/arrays so jsonify never crashes."""
    def dumps(self, obj, **kw):
        return json.dumps(obj, default=self._convert, **kw)

    def loads(self, s, **kw):
        return json.loads(s, **kw)

    @staticmethod
    def _convert(o):
        if isinstance(o, (np.integer,)):          return int(o)
        if isinstance(o, (np.floating,)):         return float(o)
        if isinstance(o, (np.bool_,)):            return bool(o)
        if isinstance(o, np.ndarray):             return o.tolist()
        raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')

sys.path.insert(0, os.path.dirname(__file__))

from database.auth_store    import AuthStore
from database.session_store import SessionStore
from pipeline.longitudinal  import interpret_score

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.secret_key = os.environ.get('SECRET_KEY', 'respitriage-secret-2026')
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

auth_store    = AuthStore()
session_store = SessionStore()

# Pre-load ML pipeline at startup so first request doesn't trigger reloader
print('[server] Pre-loading ML pipeline...')
from pipeline.triage_graph import run_triage as _preload_triage  # noqa: F401
print('[server] ML pipeline ready.')


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('index'))
        if session['user'].get('role') != 'doctor':
            return redirect(url_for('patient_portal'))
        return f(*args, **kwargs)
    return decorated


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user' in session:
        if session['user'].get('role') == 'doctor':
            return redirect(url_for('doctor_portal'))
        return redirect(url_for('patient_portal'))
    return render_template('login.html')


@app.route('/patient')
@login_required
def patient_portal():
    user = session['user']
    patient_id = f"patient_{user['id']}"
    profile = auth_store.get_profile(user['id']) or {}
    history = session_store.get_sessions(patient_id, n=30)
    alerts  = session_store.check_deterioration(patient_id) or []
    baseline = session_store.get_baseline(patient_id) or {}
    return render_template('patient.html',
                           user=user,
                           patient_id=patient_id,
                           profile=profile,
                           history=history,
                           alerts=alerts,
                           baseline=baseline)


@app.route('/doctor')
@doctor_required
def doctor_portal():
    user = session['user']
    patients = auth_store.get_all_patients()
    rows = []
    for p in patients:
        pid = f"patient_{p['id']}"
        latest = session_store.get_latest_session(pid)
        total  = len(session_store.get_sessions(pid, n=100))
        det    = session_store.check_deterioration(pid) or []
        rows.append({
            'id':         p['id'],
            'patient_id': pid,
            'name':       p['full_name'] or p['username'],
            'username':   p['username'],
            'age':        p.get('age') or '—',
            'gender':     (p.get('gender') or '—').capitalize(),
            'sessions':   total,
            'severity':   latest['severity'] if latest else '—',
            'risk_score': f"{latest.get('longitudinal_score',0):.0%}" if latest else '—',
            'copd':       f"{latest['copd_confidence']:.0%}" if latest else '—',
            'pneu':       f"{latest['pneu_confidence']:.0%}" if latest else '—',
            'voice':      f"{latest.get('voice_index',0):.0%}" if latest else '—',
            'last_visit': latest['timestamp'][:10] if latest else 'Never',
            'alert':      bool(det),
        })
    return render_template('doctor.html', user=user, patients=rows)


@app.route('/doctor/patient/<int:patient_db_id>')
@doctor_required
def doctor_patient(patient_db_id):
    user        = session['user']
    sel_user    = auth_store.get_user_by_id(patient_db_id) or {}
    sel_profile = auth_store.get_profile(patient_db_id) or {}
    sel_pid     = f"patient_{patient_db_id}"
    sel_name    = sel_user.get('full_name') or sel_user.get('username', '—')
    history     = session_store.get_sessions(sel_pid, n=50)
    alerts      = session_store.check_deterioration(sel_pid) or []
    baseline    = session_store.get_baseline(sel_pid) or {}
    latest      = history[0] if history else {}

    # Build chart data (chronological)
    chart_data = []
    for i, s in enumerate(reversed(history)):
        chart_data.append({
            'session':            i + 1,
            'longitudinal_score': round(s.get('longitudinal_score', 0), 3),
            'symptom_index':      round(s.get('symptom_index', 0), 3),
            'voice_index':        round(s.get('voice_index', 0), 3),
            'copd_confidence':    round(s.get('copd_confidence', 0), 3),
            'pneu_confidence':    round(s.get('pneu_confidence', 0), 3),
            'tier':               s.get('tier', 1),
        })

    long_score = latest.get('longitudinal_score', 0.0)
    interp     = interpret_score(long_score)

    return render_template('doctor_patient.html',
                           user=user,
                           sel_user=sel_user,
                           sel_profile=sel_profile,
                           sel_pid=sel_pid,
                           sel_name=sel_name,
                           history=history,
                           alerts=alerts,
                           baseline=baseline,
                           latest=latest,
                           chart_data=chart_data,
                           interp=interp,
                           long_score=long_score,
                           patient_db_id=patient_db_id)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    user = auth_store.login(data.get('username',''), data.get('password',''))
    if user:
        session['user'] = user
        return jsonify({'ok': True, 'role': user['role']})
    return jsonify({'ok': False, 'error': 'Invalid username or password'}), 401


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    result = auth_store.register_user(
        data.get('username',''), data.get('password',''),
        'patient', data.get('full_name',''))
    if result['success']:
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': result.get('error','Registration failed')}), 400


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'ok': True})


@app.route('/api/validate-symptoms', methods=['POST'])
@login_required
def api_validate_symptoms():
    """LLM-validate free-text symptoms entered by patient."""
    data     = request.get_json()
    raw_text = data.get('symptoms_text', '').strip()
    if not raw_text:
        return jsonify({'ok': True, 'valid': [], 'invalid': [], 'boost': 0.0, 'summary': ''})
    result = validate_symptoms(raw_text)
    return jsonify({'ok': True, **result})


@app.route('/api/profile/save', methods=['POST'])
@login_required
def api_profile_save():
    user = session['user']
    data = request.get_json()
    auth_store.update_profile(
        user['id'],
        int(data.get('age', 25)),
        data.get('gender', 'male'),
        bool(data.get('respiratory_condition', False)),
        bool(data.get('smoking', False)),
        data.get('notes', ''))
    return jsonify({'ok': True})


@app.route('/api/screen', methods=['POST'])
@login_required
def api_screen():
    """Tier 1 patient self-screen."""
    user       = session['user']
    patient_id = f"patient_{user['id']}"

    patient_info = {
        'age':                   int(request.form.get('age', 25)),
        'gender':                request.form.get('gender', 'male'),
        'symptoms':              json.loads(request.form.get('symptoms', '[]')),
        'fever_muscle_pain':     request.form.get('fever') == 'true',
        'respiratory_condition': request.form.get('resp_cond') == 'true',
        'cough_detected':        float(request.form.get('cough_sev', 0.3)),
        'cough_severity':        float(request.form.get('cough_sev', 0.3)) * 10,
        'dyspnea':               int(request.form.get('dyspnea_level', 0)) >= 2,
        'dyspnea_level':         int(request.form.get('dyspnea_level', 0)),
        'wheezing':              request.form.get('wheezing') == 'true',
        'congestion':            request.form.get('congestion') == 'true',
        'chest_tightness':       int(request.form.get('chest_tightness', 0)),
        'sleep_quality':         int(request.form.get('sleep_quality', 0)),
        'energy_level':          int(request.form.get('energy_level', 0)),
        'sputum':                int(request.form.get('sputum', 0)),
        'extra_symptom_boost':   float(request.form.get('extra_symptom_boost', 0.0)),
    }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            vowel_path = ''
            cough_path = ''

            vowel_file = request.files.get('vowel_file')
            if vowel_file and vowel_file.filename:
                ext = vowel_file.filename.rsplit('.', 1)[-1]
                vowel_path = os.path.join(tmpdir, f'vowel.{ext}')
                vowel_file.save(vowel_path)

            cough_file = request.files.get('cough_file')
            if cough_file and cough_file.filename:
                ext = cough_file.filename.rsplit('.', 1)[-1]
                cough_path = os.path.join(tmpdir, f'cough.{ext}')
                cough_file.save(cough_path)

            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info,
                                cough_audio_path=cough_path,
                                lung_audio_path='',
                                vowel_audio_path=vowel_path,
                                patient_id=patient_id)

        decision   = result.get('triage_decision', {})
        long_score = result.get('longitudinal_score', 0.0)
        interp     = interpret_score(long_score)

        # Apply extra symptom boost from LLM-validated free-text symptoms
        boost        = patient_info.get('extra_symptom_boost', 0.0)
        symptom_idx  = min(result.get('symptom_index', 0) + boost, 1.0)

        return jsonify({
            'ok':          True,
            'severity':    decision.get('severity', 'UNKNOWN'),
            'diagnosis':   decision.get('diagnosis', 'N/A'),
            'referral':    decision.get('referral_urgency', 'none').upper(),
            'action':      decision.get('recommended_action', '—'),
            'reasoning':   decision.get('reasoning', '—'),
            'long_score':  long_score,
            'interp':      interp,
            'symptom_index': symptom_idx,
            'voice_index':   result.get('voice_index', 0),
            'drift_score':   result.get('drift_score', 0),
            'voice_features': result.get('voice_result', {}).get('features', {}),
            'is_baseline':   result.get('voice_result', {}).get('is_baseline', False),
            'alerts':        (result.get('session_result', {}) or {}).get('deterioration_alerts') or [],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'error': f'Pipeline error: {str(e)}'}), 500


@app.route('/api/tier2', methods=['POST'])
@doctor_required
def api_tier2():
    """Tier 2 doctor assessment."""
    patient_db_id = int(request.form.get('patient_db_id'))
    sel_pid       = f"patient_{patient_db_id}"
    sel_profile   = auth_store.get_profile(patient_db_id) or {}

    patient_info = {
        'age':                   int(sel_profile.get('age') or 45),
        'gender':                sel_profile.get('gender', 'male'),
        'symptoms':              json.loads(request.form.get('symptoms', '[]')),
        'fever_muscle_pain':     request.form.get('fever') == 'true',
        'respiratory_condition': request.form.get('resp_cond') == 'true',
        'cough_detected':        float(request.form.get('cough_sev', 0.3)),
        'cough_severity':        float(request.form.get('cough_sev', 0.3)) * 10,
        'dyspnea':               request.form.get('dyspnea') == 'true',
        'dyspnea_level':         int(request.form.get('dyspnea_level', 0)),
        'wheezing':              request.form.get('wheezing') == 'true',
        'congestion':            request.form.get('congestion') == 'true',
        'chest_tightness': 0, 'sleep_quality': 0,
        'energy_level': 0, 'sputum': 0,
        'extra_symptom_boost': float(request.form.get('extra_symptom_boost', 0.0)),
    }

    lung_file = request.files.get('lung_file')
    if not lung_file or not lung_file.filename:
        return jsonify({'ok': False, 'error': 'No lung audio uploaded'}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = lung_file.filename.rsplit('.', 1)[-1]
            lung_path = os.path.join(tmpdir, f'lung.{ext}')
            lung_file.save(lung_path)

            from pipeline.triage_graph import run_triage
            result = run_triage(patient_info,
                                cough_audio_path=lung_path,
                                lung_audio_path=lung_path,
                                vowel_audio_path='',
                                patient_id=sel_pid)

        decision    = result.get('triage_decision', {})
        long_score  = result.get('longitudinal_score', 0.0)
        interp      = interpret_score(long_score)
        copd_r      = result.get('copd_result', {})
        pneu_r      = result.get('pneumonia_result', {})
        snd_r       = result.get('sound_result', {})
        boost       = patient_info.get('extra_symptom_boost', 0.0)
        symptom_idx = min(result.get('symptom_index', 0) + boost, 1.0)

        return jsonify({
            'ok':           True,
            'severity':     decision.get('severity', 'UNKNOWN'),
            'diagnosis':    decision.get('diagnosis', 'N/A'),
            'confidence':   decision.get('confidence', 0),
            'referral':     decision.get('referral_urgency', 'none').upper(),
            'action':       decision.get('recommended_action', '—'),
            'reasoning':    decision.get('reasoning', '—'),
            'long_score':   long_score,
            'interp':       interp,
            'symptom_index': symptom_idx,
            'copd':         copd_r,
            'pneu':         pneu_r,
            'sound':        snd_r,
            'alerts':       (result.get('session_result', {}) or {}).get('deterioration_alerts') or [],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'error': f'Pipeline error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
