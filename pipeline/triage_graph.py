"""
pipeline/triage_graph.py — LangGraph StateGraph for respiratory triage.

Updated architecture:
  Tier 1 (patient self-screen): cough audio + symptoms
      → COPDAgent + PneumoniaAgent + SymptomAgent → RuleEngine → SessionAgent
  Tier 2 (clinician): above + stethoscope lung audio
      → + SoundAgent → RuleEngine → SessionAgent

Graph structure:
  START → analyze_symptoms → run_copd_agent → run_pneumonia_agent
        → [analyze_lung if Tier 2] → apply_rules → record_session → END
"""

import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from agents.symptom_agent import SymptomAgent
from agents.copd_agent import COPDAgent
from agents.pneumonia_agent import PneumoniaAgent
from agents.sound_agent import SoundAgent
from agents.session_agent import SessionAgent
from pipeline.rule_engine import RespiratoryRuleEngine

# ── Lazy-loaded singleton agents ────────────────────────────────────────────
# Loaded once on first use to avoid re-loading models per request

_symptom_agent  = None
_copd_agent     = None
_pneumonia_agent = None
_sound_agent    = None
_session_agent  = None
_rule_engine    = RespiratoryRuleEngine()


def _get_symptom_agent():
    global _symptom_agent
    if _symptom_agent is None:
        _symptom_agent = SymptomAgent()
    return _symptom_agent


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


# ── State schema ─────────────────────────────────────────────────────────────

class TriageState(TypedDict):
    # Inputs
    patient_info:     dict    # age, gender, symptoms, dyspnea, wheezing, ...
    patient_id:       str     # unique ID for session tracking
    cough_audio_path: str     # path to cough recording
    lung_audio_path:  str     # path to stethoscope recording ('' = Tier 1)

    # Agent outputs (filled by nodes)
    symptom_result:   dict
    copd_result:      dict
    pneumonia_result: dict
    sound_result:     dict

    # Final outputs
    triage_decision:  dict
    session_result:   dict
    tier:             int     # 1 or 2


# ── Graph nodes ──────────────────────────────────────────────────────────────

def analyze_symptoms(state: TriageState) -> dict:
    """Run SymptomAgent on patient metadata."""
    print("[triage] Analyzing symptoms ...")
    info = state["patient_info"]
    try:
        result = _get_symptom_agent().predict(
            age=info.get("age", 30),
            gender=info.get("gender", "unknown"),
            fever_muscle_pain=info.get("fever_muscle_pain", False),
            dyspnea=info.get("dyspnea", False),
            wheezing=info.get("wheezing", False),
            congestion=info.get("congestion", False),
            resp_condition=info.get("respiratory_condition", False),
            cough_severity=info.get("cough_severity", 5),
        )
    except Exception as e:
        print(f"[triage] SymptomAgent error: {e}")
        result = {
            'agent': 'Symptom Agent', 'symptomatic_probability': 0.0,
            'copd_probability_hint': 0.0, 'pneumonia_probability_hint': 0.0,
            'detected': False, 'confidence': 0.0, 'error': str(e),
        }
    return {"symptom_result": result}


_COPD_SKIP = {
    'agent': 'COPD Agent', 'disease': 'COPD',
    'detected': False, 'confidence': 0.0,
    'probability': 0.0, 'severity_hint': 'LOW',
    'threshold_used': 0.5, 'error': 'No audio — Tier 1 symptoms-only mode',
}
_PNEU_SKIP = {
    'agent': 'Pneumonia Agent', 'disease': 'Pneumonia',
    'detected': False, 'confidence': 0.0,
    'probability': 0.0, 'severity_hint': 'LOW',
    'threshold_used': 0.64, 'error': 'No audio — Tier 1 symptoms-only mode',
}

def run_copd_agent(state: TriageState) -> dict:
    """Run COPDAgent on lung audio (Tier 2 only). Skip in Tier 1."""
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
    """Run PneumoniaAgent on lung audio (Tier 2 only). Skip in Tier 1."""
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
    """Route to lung analysis (Tier 2) or directly to rules (Tier 1)."""
    lung_path = state.get("lung_audio_path", "")
    if lung_path and lung_path.strip():
        return "analyze_lung"
    return "apply_rules"


def analyze_lung(state: TriageState) -> dict:
    """Run SoundAgent on stethoscope recording (Tier 2 only)."""
    print("[triage] Analyzing lung sounds (Tier 2) ...")
    try:
        result = _get_sound_agent().predict(state["lung_audio_path"])
    except Exception as e:
        print(f"[triage] SoundAgent error: {e}")
        result = {
            'agent': 'Sound Agent', 'sound_type': 'Normal',
            'confidence': 0.0, 'all_probabilities': {},
            'error': str(e),
        }
    return {"sound_result": result}


def apply_rules(state: TriageState) -> dict:
    """Apply deterministic rule engine to all agent outputs."""
    print("[triage] Applying clinical rules ...")
    lung_path = state.get("lung_audio_path", "")
    tier = 2 if (lung_path and lung_path.strip()) else 1

    decision = _rule_engine.evaluate(
        patient_info=state["patient_info"],
        copd_result=state.get("copd_result", {}),
        pneumonia_result=state.get("pneumonia_result", {}),
        symptom_result=state.get("symptom_result", {}),
        sound_result=state.get("sound_result") if tier == 2 else None,
    )
    decision["tier"] = tier

    print(f"[triage] Decision: {decision.get('diagnosis', 'N/A')} | "
          f"Severity: {decision.get('severity', 'N/A')}")

    return {"triage_decision": decision, "tier": tier}


def record_session(state: TriageState) -> dict:
    """Record session and check for deterioration trend."""
    print("[triage] Recording session ...")
    try:
        copd_conf = state.get("copd_result", {}).get("probability", 0.0)
        pneu_conf = state.get("pneumonia_result", {}).get("probability", 0.0)
        sound_type = state.get("sound_result", {}).get("sound_type", "Normal") \
            if state.get("sound_result") else "Normal"
        cough_sev = state["patient_info"].get("cough_severity", 0)

        session_result = _get_session_agent().record_and_check(
            patient_id=state.get("patient_id", "anonymous"),
            triage_result=state.get("triage_decision", {}),
            copd_conf=copd_conf,
            pneu_conf=pneu_conf,
            tier=state.get("tier", 1),
            sound_type=sound_type,
            cough_severity=int(cough_sev),
        )
    except Exception as e:
        print(f"[triage] SessionAgent error: {e}")
        session_result = {
            'agent': 'Session Memory Agent', 'session_saved': False,
            'deterioration_alerts': None, 'session_history': [],
            'total_sessions': 0, 'error': str(e),
        }
    return {"session_result": session_result}


# ── Build the graph ───────────────────────────────────────────────────────────

def build_triage_graph() -> StateGraph:
    """
    Build and compile the LangGraph triage pipeline.

    Invoke with:
        result = graph.invoke({
            "patient_info":     {...},
            "patient_id":       "P001",
            "cough_audio_path": "path/to/cough.wav",
            "lung_audio_path":  "",   # empty = Tier 1
        })
    """
    graph = StateGraph(TriageState)

    graph.add_node("analyze_symptoms",   analyze_symptoms)
    graph.add_node("run_copd_agent",     run_copd_agent)
    graph.add_node("run_pneumonia_agent", run_pneumonia_agent)
    graph.add_node("analyze_lung",       analyze_lung)
    graph.add_node("apply_rules",        apply_rules)
    graph.add_node("record_session",     record_session)

    # Entry: symptoms first (fastest, no GPU needed)
    graph.set_entry_point("analyze_symptoms")

    # Then run both audio disease agents in sequence
    graph.add_edge("analyze_symptoms",   "run_copd_agent")
    graph.add_edge("run_copd_agent",     "run_pneumonia_agent")

    # Route: Tier 2 adds lung sound analysis, Tier 1 skips
    graph.add_conditional_edges(
        "run_pneumonia_agent",
        route_tier,
        {
            "analyze_lung": "analyze_lung",
            "apply_rules":  "apply_rules",
        },
    )

    graph.add_edge("analyze_lung",   "apply_rules")
    graph.add_edge("apply_rules",    "record_session")
    graph.add_edge("record_session", END)

    return graph.compile()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_triage(patient_info: dict,
               cough_audio_path: str,
               lung_audio_path: str = "",
               patient_id: str = "anonymous") -> dict:
    """
    Run the full triage pipeline and return the complete state.

    Parameters
    ----------
    patient_info     : dict with age, gender, fever_muscle_pain,
                       respiratory_condition, dyspnea, wheezing,
                       congestion, cough_severity (0-10)
    cough_audio_path : path to cough recording
    lung_audio_path  : path to stethoscope recording ('' = Tier 1)
    patient_id       : unique ID for longitudinal session tracking
    """
    graph = build_triage_graph()

    initial_state: TriageState = {
        "patient_info":     patient_info,
        "patient_id":       patient_id,
        "cough_audio_path": cough_audio_path,
        "lung_audio_path":  lung_audio_path,
        "symptom_result":   {},
        "copd_result":      {},
        "pneumonia_result": {},
        "sound_result":     {},
        "triage_decision":  {},
        "session_result":   {},
        "tier":             1,
    }

    return graph.invoke(initial_state)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pipeline/triage_graph.py <cough_audio> [lung_audio]")
        sys.exit(1)

    cough_path = sys.argv[1]
    lung_path  = sys.argv[2] if len(sys.argv) > 2 else ""

    test_patient = {
        "age": 55, "gender": "male",
        "fever_muscle_pain": False, "respiratory_condition": True,
        "dyspnea": True, "wheezing": True, "congestion": False,
        "cough_severity": 7,
    }

    result = run_triage(test_patient, cough_path, lung_path, patient_id="TEST001")

    print("\n" + "=" * 60)
    print("TRIAGE RESULT")
    print("=" * 60)
    print(json.dumps(result["triage_decision"], indent=2))

    if result["session_result"].get("deterioration_alerts"):
        print("\n*** DETERIORATION ALERTS ***")
        for alert in result["session_result"]["deterioration_alerts"]:
            print(f"  {alert['message']}")
