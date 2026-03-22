"""
pipeline/triage_graph.py — LangGraph StateGraph for respiratory triage.

Two-tier agentic pipeline:
  Tier 1 (patient self-screening): cough audio + symptoms → LLM triage
  Tier 2 (clinician confirmation): cough + symptoms + lung audio → LLM triage

Graph structure:
  START → [analyze_cough, analyze_symptoms] → route_tier
    Tier 1: → llm_triage → END
    Tier 2: → analyze_lung → llm_triage → END
"""

import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

from agents.cough_agent import predict_cough
from agents.symptom_agent import predict_symptom
from agents.lung_agent import predict_from_audio as predict_lung
from pipeline.llm_provider import build_triage_prompt, call_llm


# ── State schema ────────────────────────────────────────────────────────────

class TriageState(TypedDict):
    # Inputs
    patient_info: dict           # age, gender, symptoms list
    cough_audio_path: str        # path to cough .wav/.mp3
    lung_audio_path: str         # path to lung .wav (empty string = Tier 1)

    # Agent outputs (filled by nodes)
    cough_result: dict
    symptom_result: dict
    lung_result: dict

    # Final output
    triage_decision: dict
    tier: int                    # 1 or 2


# ── Node functions ──────────────────────────────────────────────────────────

def analyze_cough(state: TriageState) -> dict:
    """Run cough agent on the uploaded audio."""
    print("[triage] Analyzing cough audio …")
    try:
        result = predict_cough(state["cough_audio_path"])
    except Exception as e:
        print(f"[triage] Cough agent error: {e}")
        result = {"label": "Error", "confidence": 0.0, "probabilities": {},
                  "agent": "cough_agent", "error": str(e)}
    return {"cough_result": result}


def analyze_symptoms(state: TriageState) -> dict:
    """Run symptom agent on patient-reported metadata."""
    print("[triage] Analyzing patient symptoms …")
    info = state["patient_info"]
    try:
        result = predict_symptom(
            age=info.get("age", 30),
            gender=info.get("gender", "unknown"),
            fever_muscle_pain=info.get("fever_muscle_pain", False),
            respiratory_condition=info.get("respiratory_condition", False),
            cough_detected=info.get("cough_detected", 0.5),
            dyspnea=info.get("dyspnea", False),
            wheezing=info.get("wheezing", False),
            congestion=info.get("congestion", False),
        )
    except Exception as e:
        print(f"[triage] Symptom agent error: {e}")
        result = {"label": "Error", "confidence": 0.0, "probabilities": {},
                  "agent": "symptom_agent", "error": str(e)}
    return {"symptom_result": result}


def analyze_lung(state: TriageState) -> dict:
    """Run lung agent on stethoscope recording (Tier 2 only)."""
    print("[triage] Analyzing lung sounds …")
    try:
        result = predict_lung(state["lung_audio_path"])
    except Exception as e:
        print(f"[triage] Lung agent error: {e}")
        result = {"disease": {"label": "Error", "confidence": 0.0, "probabilities": {}},
                  "sound": {"label": "Error", "confidence": 0.0, "probabilities": {}},
                  "agent": "lung_agent", "error": str(e)}
    return {"lung_result": result}


def route_tier(state: TriageState) -> str:
    """Route to lung analysis (Tier 2) or straight to LLM (Tier 1)."""
    lung_path = state.get("lung_audio_path", "")
    if lung_path and lung_path.strip():
        return "analyze_lung"
    return "llm_triage"


def llm_triage(state: TriageState) -> dict:
    """Send all agent results to LLM for final triage decision."""
    print("[triage] Sending to LLM for clinical reasoning …")

    lung_result = state.get("lung_result")
    tier = 2 if (lung_result and "disease" in lung_result
                 and lung_result["disease"]["label"] != "Error") else 1

    prompt = build_triage_prompt(
        patient_info=state["patient_info"],
        cough_result=state.get("cough_result", {}),
        symptom_result=state.get("symptom_result", {}),
        lung_result=lung_result if tier == 2 else None,
    )

    decision = call_llm(prompt)
    decision["tier"] = tier

    print(f"[triage] Decision: {decision.get('diagnosis', 'N/A')} | "
          f"Severity: {decision.get('severity', 'N/A')} | "
          f"Provider: {decision.get('llm_provider', 'N/A')}")

    return {"triage_decision": decision, "tier": tier}


# ── Build the graph ─────────────────────────────────────────────────────────

def build_triage_graph() -> StateGraph:
    """
    Build and compile the LangGraph triage pipeline.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({
            "patient_info": {...},
            "cough_audio_path": "path/to/cough.wav",
            "lung_audio_path": "",  # empty = Tier 1
        })
    """
    graph = StateGraph(TriageState)

    # Add nodes
    graph.add_node("analyze_cough", analyze_cough)
    graph.add_node("analyze_symptoms", analyze_symptoms)
    graph.add_node("analyze_lung", analyze_lung)
    graph.add_node("llm_triage", llm_triage)

    # START → cough + symptoms in parallel
    graph.set_entry_point("analyze_cough")
    graph.add_edge("analyze_cough", "analyze_symptoms")

    # After symptoms → route based on tier
    graph.add_conditional_edges(
        "analyze_symptoms",
        route_tier,
        {
            "analyze_lung": "analyze_lung",
            "llm_triage": "llm_triage",
        }
    )

    # Lung → LLM (Tier 2 path)
    graph.add_edge("analyze_lung", "llm_triage")

    # LLM → END
    graph.add_edge("llm_triage", END)

    return graph.compile()


# ── Convenience function ────────────────────────────────────────────────────

def run_triage(patient_info: dict,
               cough_audio_path: str,
               lung_audio_path: str = "") -> dict:
    """
    Run the full triage pipeline.

    Parameters
    ----------
    patient_info : dict with keys:
        age (int), gender (str), symptoms (list[str]),
        fever_muscle_pain (bool), respiratory_condition (bool),
        cough_detected (float 0-1), dyspnea (bool),
        wheezing (bool), congestion (bool)
    cough_audio_path : path to cough recording (.wav/.mp3)
    lung_audio_path  : path to lung recording (.wav) — empty string for Tier 1

    Returns
    -------
    dict : full state including all agent results + triage_decision
    """
    graph = build_triage_graph()

    initial_state = {
        "patient_info": patient_info,
        "cough_audio_path": cough_audio_path,
        "lung_audio_path": lung_audio_path,
        "cough_result": {},
        "symptom_result": {},
        "lung_result": {},
        "triage_decision": {},
        "tier": 1,
    }

    result = graph.invoke(initial_state)
    return result


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pipeline/triage_graph.py <cough_audio> [lung_audio]")
        print("\nExample (Tier 1 — no lung audio):")
        print('  python pipeline/triage_graph.py test_cough.wav')
        print("\nExample (Tier 2 — with lung audio):")
        print('  python pipeline/triage_graph.py test_cough.wav test_lung.wav')
        sys.exit(1)

    cough_path = sys.argv[1]
    lung_path  = sys.argv[2] if len(sys.argv) > 2 else ""

    test_patient = {
        "age": 45,
        "gender": "male",
        "symptoms": ["persistent cough", "fever", "difficulty breathing"],
        "fever_muscle_pain": True,
        "respiratory_condition": False,
        "cough_detected": 0.95,
        "dyspnea": True,
        "wheezing": False,
        "congestion": False,
    }

    print("=" * 60)
    print("RESPIRATORY TRIAGE PIPELINE")
    print(f"Tier: {'2 (with lung audio)' if lung_path else '1 (cough + symptoms only)'}")
    print("=" * 60)

    result = run_triage(test_patient, cough_path, lung_path)

    print("\n" + "=" * 60)
    print("TRIAGE RESULT")
    print("=" * 60)
    print(json.dumps(result["triage_decision"], indent=2))
