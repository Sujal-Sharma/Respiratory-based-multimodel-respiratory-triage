"""
pipeline/llm_provider.py — Dual-provider LLM for clinical triage reasoning.

Providers (both free):
  Primary  : Groq API  — Llama 3.3 70B (30 RPM, 14,400 req/day)
  Fallback : Gemini API — Gemini 2.5 Flash (10 RPM, 250 req/day)

Returns structured JSON with diagnosis, severity, and recommended action.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Clinical prompt template ────────────────────────────────────────────────

TRIAGE_PROMPT = """You are a clinical respiratory triage AI assistant.

Analyse the following patient data from multiple diagnostic agents and provide a structured triage decision.

## Patient Information
- Age: {age}
- Gender: {gender}
- Reported Symptoms: {symptoms}

## Agent 1 — Cough Analysis (LightCoughCNN)
- Prediction: {cough_label}
- Confidence: {cough_confidence}
- Probabilities: {cough_probabilities}

## Agent 2 — Symptom Analysis (XGBoost)
- Prediction: {symptom_label}
- Confidence: {symptom_confidence}
- Probabilities: {symptom_probabilities}

## Agent 3 — Lung Sound Analysis (MultiTaskEfficientNet)
{lung_section}

## Instructions
Based on ALL available agent results, provide your triage decision as JSON with exactly these keys:
- "diagnosis": most likely condition (string)
- "severity": one of "LOW", "MODERATE", "HIGH", "CRITICAL"
- "confidence": your overall confidence 0.0-1.0 (float)
- "reasoning": 2-3 sentence clinical reasoning (string)
- "recommended_action": specific next step for the patient (string)
- "referral_urgency": one of "routine", "soon", "urgent", "emergency"
- "agents_agreement": whether agents broadly agree or conflict (string)

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

LUNG_AVAILABLE_TEMPLATE = """- Disease Prediction: {disease_label} (confidence: {disease_confidence})
- Disease Probabilities: {disease_probabilities}
- Sound Prediction: {sound_label} (confidence: {sound_confidence})
- Sound Probabilities: {sound_probabilities}"""

LUNG_UNAVAILABLE = "- Not available (Tier 1 screening — no stethoscope recording provided)"


def build_triage_prompt(patient_info: dict,
                        cough_result: dict,
                        symptom_result: dict,
                        lung_result: dict | None = None) -> str:
    """Build the clinical prompt from agent results."""

    if lung_result and 'disease' in lung_result:
        lung_section = LUNG_AVAILABLE_TEMPLATE.format(
            disease_label=lung_result['disease']['label'],
            disease_confidence=lung_result['disease']['confidence'],
            disease_probabilities=lung_result['disease']['probabilities'],
            sound_label=lung_result['sound']['label'],
            sound_confidence=lung_result['sound']['confidence'],
            sound_probabilities=lung_result['sound']['probabilities'],
        )
    else:
        lung_section = LUNG_UNAVAILABLE

    symptoms_str = ", ".join(patient_info.get('symptoms', [])) or "None reported"

    return TRIAGE_PROMPT.format(
        age=patient_info.get('age', 'Unknown'),
        gender=patient_info.get('gender', 'Unknown'),
        symptoms=symptoms_str,
        cough_label=cough_result.get('label', 'N/A'),
        cough_confidence=cough_result.get('confidence', 'N/A'),
        cough_probabilities=cough_result.get('probabilities', {}),
        symptom_label=symptom_result.get('label', 'N/A'),
        symptom_confidence=symptom_result.get('confidence', 'N/A'),
        symptom_probabilities=symptom_result.get('probabilities', {}),
        lung_section=lung_section,
    )


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


# ── Groq provider ───────────────────────────────────────────────────────────

def call_groq(prompt: str) -> dict | None:
    """Call Groq API with Llama 3.3 70B. Returns parsed JSON or None on failure."""
    if not GROQ_API_KEY:
        print("[llm_provider] Groq API key not set — skipping")
        return None

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a clinical triage AI. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return _parse_json_response(response.choices[0].message.content)

    except Exception as e:
        print(f"[llm_provider] Groq failed: {e}")
        return None


# ── Gemini provider ─────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> dict | None:
    """Call Gemini 2.5 Flash API. Returns parsed JSON or None on failure."""
    if not GEMINI_API_KEY:
        print("[llm_provider] Gemini API key not set — skipping")
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3,
                "max_output_tokens": 1024,
            },
        )
        response = model.generate_content(prompt)
        return _parse_json_response(response.text)

    except Exception as e:
        print(f"[llm_provider] Gemini failed: {e}")
        return None


# ── Unified call with fallback ──────────────────────────────────────────────

def call_llm(prompt: str) -> dict:
    """
    Call LLM with automatic fallback: Groq → Gemini → error dict.

    Returns
    -------
    dict : parsed JSON triage decision
    """
    # Try Groq first (faster, higher limits)
    result = call_groq(prompt)
    if result:
        result['llm_provider'] = 'groq'
        return result

    # Fallback to Gemini
    print("[llm_provider] Falling back to Gemini …")
    result = call_gemini(prompt)
    if result:
        result['llm_provider'] = 'gemini'
        return result

    # Both failed
    print("[llm_provider] All providers failed — returning fallback response")
    return {
        "diagnosis": "Unable to determine",
        "severity": "HIGH",
        "confidence": 0.0,
        "reasoning": "AI triage system unavailable. All LLM providers failed.",
        "recommended_action": "Please consult a healthcare professional directly.",
        "referral_urgency": "urgent",
        "agents_agreement": "N/A",
        "llm_provider": "none",
    }
