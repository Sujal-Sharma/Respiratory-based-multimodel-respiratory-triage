"""
pipeline/triage_graph.py — LangGraph StateGraph for respiratory triage.

Tier 1 (patient self-screen): symptoms + optional vowel + optional cough
    → SymptomAgent + VoiceAgent + CoughDrift → LongitudinalScore → RuleEngine
Tier 2 (clinician): stethoscope audio
    → COPDAgent + PneumoniaAgent + SoundAgent → RuleEngine

Graph:
  START → analyze_symptoms → run_voice_agent → run_cough_drift
        → [run_copd + run_pneumonia + analyze_lung if Tier 2]
        → apply_rules → record_session → END
"""

import os
import numpy as np
from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.symptom_agent  import SymptomAgent
from agents.voice_agent    import VoiceAgent
from agents.copd_agent     import COPDAgent
from agents.pneumonia_agent import PneumoniaAgent
from agents.sound_agent    import SoundAgent
from agents.session_agent  import SessionAgent
from pipeline.rule_engine  import RespiratoryRuleEngine
from pipeline.longitudinal import (compute_longitudinal_score,
                                    compute_cough_drift, interpret_score)
from database.session_store import SessionStore

# ── Singletons ────────────────────────────────────────────────────────────────
_symptom_agent   = None
_voice_agent     = None
_copd_agent      = None
_pneumonia_agent = None
_sound_agent     = None
_session_agent   = None
_session_store   = None
_rule_engine     = RespiratoryRuleEngine()


def _get_symptom_agent():
    global _symptom_agent
    if _symptom_agent is None:
        _symptom_agent = SymptomAgent()
    return _symptom_agent

def _get_voice_agent():
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgent()
    return _voice_agent

def _get_copd_agent():
    global _copd_agent
    if _copd_agent is None:
        _copd_agent = COPDAgent()
    return _copd_agent

def _get_pneumonia_agent():
    global _pneumonia_agent
    if _pneumonia_agent is None:
        _pneumonia_agent = PneumoniaAgent()
    return _pneumonia_agent

def _get_sound_agent():
    global _sound_agent
    if _sound_agent is None:
        _sound_agent = SoundAgent()
    return _sound_agent

def _get_session_agent():
    global _session_agent
    if _session_agent is None:
        _session_agent = SessionAgent()
    return _session_agent

def _get_session_store():
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


# ── State schema ──────────────────────────────────────────────────────────────

class TriageState(TypedDict):
    # Inputs
    patient_info:     dict
    patient_id:       str
    cough_audio_path: str   # optional cough recording (Tier 1)
    lung_audio_path:  str   # stethoscope recording (Tier 2, empty = Tier 1)
    vowel_audio_path: str   # sustained vowel recording (Tier 1)

    # Agent outputs
    symptom_result:   dict
    voice_result:     dict
    copd_result:      dict
    pneumonia_result: dict
    sound_result:     dict

    # Longitudinal scores
    symptom_index:      float
    voice_index:        float
    drift_score:        float
    longitudinal_score: float

    # Final
    triage_decision: dict
    session_result:  dict
    tier:            int


# ── Skip defaults ─────────────────────────────────────────────────────────────

_COPD_SKIP = {
    'agent': 'COPD Agent', 'disease': 'COPD',
    'detected': False, 'confidence': 0.0,
    'probability': 0.0, 'severity_hint': 'LOW',
    'threshold_used': 0.5, 'error': 'Tier 1 — no stethoscope audio',
}
_PNEU_SKIP = {
    'agent': 'Pneumonia Agent', 'disease': 'Pneumonia',
    'detected': False, 'confidence': 0.0,
    'probability': 0.0, 'severity_hint': 'LOW',
    'threshold_used': 0.64, 'error': 'Tier 1 — no stethoscope audio',
}


# ── Graph nodes ───────────────────────────────────────────────────────────────

def analyze_symptoms(state: TriageState) -> dict:
    print("[triage] Analyzing symptoms (CAT-style) ...")
    info = state["patient_info"]
    try:
        result = _get_symptom_agent().predict(
            age              = info.get("age", 30),
            gender           = info.get("gender", "unknown"),
            fever_muscle_pain= info.get("fever_muscle_pain", False),
            dyspnea          = info.get("dyspnea", False),
            wheezing         = info.get("wheezing", False),
            congestion       = info.get("congestion", False),
            resp_condition   = info.get("respiratory_condition", False),
            cough_severity   = info.get("cough_severity", 0),
            dyspnea_level    = info.get("dyspnea_level", -1),
            chest_tightness  = info.get("chest_tightness", 0),
            sleep_quality    = info.get("sleep_quality", 0),
            energy_level     = info.get("energy_level", 0),
            sputum           = info.get("sputum", 0),
        )
    except Exception as e:
        print(f"[triage] SymptomAgent error: {e}")
        result = {
            'agent': 'Symptom Agent', 'symptom_index': 0.0,
            'symptomatic_probability': 0.0,
            'copd_probability_hint': 0.0, 'pneumonia_probability_hint': 0.0,
            'detected': False, 'confidence': 0.0, 'error': str(e),
        }
    symptom_index = result.get('symptom_index', result.get('symptomatic_probability', 0.0))
    return {"symptom_result": result, "symptom_index": float(symptom_index)}


def run_voice_agent(state: TriageState) -> dict:
    """Run VoiceAgent on sustained vowel recording (Tier 1 only)."""
    vowel_path = state.get("vowel_audio_path", "")
    if not vowel_path or state.get("lung_audio_path", ""):
        return {"voice_result": {}, "voice_index": 0.0}

    print("[triage] Analyzing voice biomarkers ...")
    patient_id = state.get("patient_id", "anonymous")

    # Load baseline if exists
    store    = _get_session_store()
    baseline = store.get_baseline(patient_id)
    baseline_features = baseline.get('voice_features') if baseline else None

    try:
        result = _get_voice_agent().predict(vowel_path, baseline_features)
        voice_index = result.get('voice_index', 0.0)

        # Save as baseline if first session
        if result.get('is_baseline') and result.get('features'):
            store.save_baseline(patient_id,
                                voice_features=result['features'],
                                cough_embedding=None)
            print(f"[triage] Voice baseline saved for {patient_id}")

    except Exception as e:
        print(f"[triage] VoiceAgent error: {e}")
        result = {'agent': 'Voice Agent', 'features': {},
                  'voice_index': 0.0, 'is_baseline': False, 'error': str(e)}
        voice_index = 0.0

    return {"voice_result": result, "voice_index": float(voice_index)}


def run_cough_drift(state: TriageState) -> dict:
    """Compute OPERA-CT cough embedding drift from personal baseline (Tier 1)."""
    cough_path = state.get("cough_audio_path", "")
    lung_path  = state.get("lung_audio_path", "")
    if not cough_path or lung_path:
        return {"drift_score": 0.0}

    print("[triage] Computing cough drift ...")
    patient_id = state.get("patient_id", "anonymous")
    store      = _get_session_store()
    baseline   = store.get_baseline(patient_id)

    try:
        from models.opera_encoder import OPERAEncoder
        enc = OPERAEncoder()
        current_emb = enc.encode(cough_path)

        # Check if cough baseline exists (may have voice features but no cough yet)
        has_cough_baseline = (baseline is not None and
                              baseline.get('cough_embedding') is not None and
                              len(baseline['cough_embedding']) > 0)

        if has_cough_baseline:
            drift = compute_cough_drift(current_emb, baseline['cough_embedding'])
            print(f"[triage] Cough drift: {drift:.4f}")
        else:
            drift = 0.0
            # Preserve existing voice features when saving cough baseline
            existing_vf = (baseline.get('voice_features') or {}) if baseline else {}
            store.save_baseline(patient_id,
                                voice_features=existing_vf,
                                cough_embedding=current_emb)
            print(f"[triage] Cough baseline saved for {patient_id}")

    except Exception as e:
        print(f"[triage] CoughDrift error: {e}")
        drift = 0.0

    return {"drift_score": float(drift)}


def run_copd_agent(state: TriageState) -> dict:
    lung_path = state.get("lung_audio_path", "")
    if not lung_path:
        return {"copd_result": _COPD_SKIP}
    print("[triage] Running COPD agent ...")
    try:
        result = _get_copd_agent().predict(lung_path)
    except Exception as e:
        print(f"[triage] COPDAgent error: {e}")
        result = {**_COPD_SKIP, 'error': str(e)}
    return {"copd_result": result}


def run_pneumonia_agent(state: TriageState) -> dict:
    lung_path = state.get("lung_audio_path", "")
    if not lung_path:
        return {"pneumonia_result": _PNEU_SKIP}
    print("[triage] Running Pneumonia agent ...")
    try:
        result = _get_pneumonia_agent().predict(lung_path)
    except Exception as e:
        print(f"[triage] PneumoniaAgent error: {e}")
        result = {**_PNEU_SKIP, 'error': str(e)}
    return {"pneumonia_result": result}


def route_tier(state: TriageState) -> str:
    lung_path = state.get("lung_audio_path", "")
    return "analyze_lung" if (lung_path and lung_path.strip()) else "apply_rules"


def analyze_lung(state: TriageState) -> dict:
    print("[triage] Analyzing lung sounds (Tier 2) ...")
    try:
        result = _get_sound_agent().predict(state["lung_audio_path"])
    except Exception as e:
        print(f"[triage] SoundAgent error: {e}")
        result = {'agent': 'Sound Agent', 'sound_type': 'Normal',
                  'confidence': 0.0, 'all_probabilities': {}, 'error': str(e)}
    return {"sound_result": result}


def compute_longitudinal(state: TriageState) -> dict:
    """Fuse symptom + voice + drift into longitudinal score."""
    symptom_idx = state.get("symptom_index", 0.0)
    voice_idx   = state.get("voice_index",   0.0)
    drift       = state.get("drift_score",   0.0)

    long_score  = compute_longitudinal_score(symptom_idx, voice_idx, drift)
    print(f"[triage] Longitudinal score: {long_score:.3f} "
          f"(sym={symptom_idx:.3f}, voice={voice_idx:.3f}, drift={drift:.3f})")
    return {"longitudinal_score": float(long_score)}


def apply_rules(state: TriageState) -> dict:
    print("[triage] Applying clinical rules ...")
    lung_path = state.get("lung_audio_path", "")
    tier = 2 if (lung_path and lung_path.strip()) else 1

    decision = _rule_engine.evaluate(
        patient_info     = state["patient_info"],
        copd_result      = state.get("copd_result", {}),
        pneumonia_result = state.get("pneumonia_result", {}),
        symptom_result   = state.get("symptom_result", {}),
        sound_result     = state.get("sound_result") if tier == 2 else None,
        longitudinal_score = state.get("longitudinal_score", 0.0),
    )
    decision["tier"] = tier

    print(f"[triage] Decision: {decision.get('diagnosis')} | "
          f"Severity: {decision.get('severity')}")
    return {"triage_decision": decision, "tier": tier}


def record_session(state: TriageState) -> dict:
    print("[triage] Recording session ...")
    try:
        tier      = state.get("tier", 1)
        copd_conf = state.get("copd_result", {}).get("probability", 0.0)
        pneu_conf = state.get("pneumonia_result", {}).get("probability", 0.0)
        sym       = state.get("symptom_result", {})

        # Tier 1: use symptom hints as proxy confidence for trend tracking
        if copd_conf == 0.0:
            copd_conf = sym.get("copd_probability_hint", 0.0)
        if pneu_conf == 0.0:
            pneu_conf = sym.get("pneumonia_probability_hint", 0.0)

        sound_type  = state.get("sound_result", {}).get("sound_type", "Normal") \
                      if state.get("sound_result") else "Normal"
        cough_sev   = float(state["patient_info"].get("cough_severity", 0))
        sym_idx     = state.get("symptom_index",      0.0)
        voice_idx   = state.get("voice_index",        0.0)
        drift       = state.get("drift_score",        0.0)
        long_score  = state.get("longitudinal_score", 0.0)

        store = _get_session_store()
        store.save_session(
            patient_id         = state.get("patient_id", "anonymous"),
            triage_result      = state.get("triage_decision", {}),
            copd_conf          = copd_conf,
            pneu_conf          = pneu_conf,
            tier               = tier,
            sound_type         = sound_type,
            cough_severity     = cough_sev,
            symptom_index      = sym_idx,
            voice_index        = voice_idx,
            drift_score        = drift,
            longitudinal_score = long_score,
        )

        alerts  = store.check_deterioration(state.get("patient_id", "anonymous"))
        history = store.get_sessions(state.get("patient_id", "anonymous"), n=10)

        if alerts:
            for a in alerts:
                print(f"[triage] *** {a['message']}")

        session_result = {
            'agent':                'Session Memory Agent',
            'session_saved':        True,
            'deterioration_alerts': alerts,
            'session_history':      history,
            'total_sessions':       len(history),
        }
    except Exception as e:
        print(f"[triage] SessionAgent error: {e}")
        session_result = {
            'agent': 'Session Memory Agent', 'session_saved': False,
            'deterioration_alerts': None, 'session_history': [],
            'total_sessions': 0, 'error': str(e),
        }
    return {"session_result": session_result}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_triage_graph() -> StateGraph:
    graph = StateGraph(TriageState)

    graph.add_node("analyze_symptoms",    analyze_symptoms)
    graph.add_node("run_voice_agent",     run_voice_agent)
    graph.add_node("run_cough_drift",     run_cough_drift)
    graph.add_node("run_copd_agent",      run_copd_agent)
    graph.add_node("run_pneumonia_agent", run_pneumonia_agent)
    graph.add_node("analyze_lung",        analyze_lung)
    graph.add_node("compute_longitudinal",compute_longitudinal)
    graph.add_node("apply_rules",         apply_rules)
    graph.add_node("record_session",      record_session)

    graph.set_entry_point("analyze_symptoms")

    # Tier 1 chain: symptoms → voice → cough drift → longitudinal → rules
    graph.add_edge("analyze_symptoms",    "run_voice_agent")
    graph.add_edge("run_voice_agent",     "run_cough_drift")
    graph.add_edge("run_cough_drift",     "run_copd_agent")
    graph.add_edge("run_copd_agent",      "run_pneumonia_agent")

    # Tier 2 routes to lung analysis
    graph.add_conditional_edges(
        "run_pneumonia_agent",
        route_tier,
        {"analyze_lung": "analyze_lung", "apply_rules": "compute_longitudinal"},
    )

    graph.add_edge("analyze_lung",        "compute_longitudinal")
    graph.add_edge("compute_longitudinal","apply_rules")
    graph.add_edge("apply_rules",         "record_session")
    graph.add_edge("record_session",      END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_triage(patient_info: dict,
               cough_audio_path: str = "",
               lung_audio_path: str  = "",
               vowel_audio_path: str = "",
               patient_id: str       = "anonymous") -> dict:
    graph = build_triage_graph()
    initial: TriageState = {
        "patient_info":      patient_info,
        "patient_id":        patient_id,
        "cough_audio_path":  cough_audio_path,
        "lung_audio_path":   lung_audio_path,
        "vowel_audio_path":  vowel_audio_path,
        "symptom_result":    {},
        "voice_result":      {},
        "copd_result":       {},
        "pneumonia_result":  {},
        "sound_result":      {},
        "symptom_index":     0.0,
        "voice_index":       0.0,
        "drift_score":       0.0,
        "longitudinal_score":0.0,
        "triage_decision":   {},
        "session_result":    {},
        "tier":              1,
    }
    return graph.invoke(initial)
