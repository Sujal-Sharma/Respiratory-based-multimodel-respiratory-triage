"""
utils/symptom_validator.py — LLM-based free-text symptom validator.

Uses Groq (llama-3.3-70b-versatile) to:
  1. Validate each symptom is a real medical/respiratory symptom
  2. Reject non-medical text (e.g. "love failure", "stressed", "bad day")
  3. Map valid symptoms to a numeric boost score (0.0–1.0) that gets
     added to the symptom_index in the pipeline

Returns:
  {
    'valid':    [list of accepted symptom strings],
    'invalid':  [list of rejected symptom strings with reasons],
    'boost':    float  — extra score to add to symptom_index (0.0–0.25 max)
    'summary':  str    — short human-readable summary
  }
"""

import os
import json
import re

# Try to load .env manually (no python-dotenv required)
def _load_env_key():
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # Walk up from this file to find .env
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


_SYSTEM_PROMPT = """You are a strict medical symptom validator for a respiratory triage AI system.

Your job:
1. Evaluate each symptom the user entered.
2. Accept ONLY real medical/physical symptoms that could be relevant to respiratory or general health assessment.
3. Reject anything that is NOT a medical symptom — emotions, life events, relationship problems, vague non-medical phrases, nonsense, etc.
4. For each VALID symptom, assign a respiratory_relevance score (0.0 to 1.0):
   - 1.0 = directly respiratory (e.g. "shortness of breath", "wheezing", "chest pain")
   - 0.5 = general medical but relevant (e.g. "fever", "fatigue", "headache")
   - 0.2 = mildly relevant (e.g. "nausea", "loss of appetite")
5. Return ONLY a JSON object, no extra text.

JSON format:
{
  "results": [
    {
      "symptom": "<original text>",
      "valid": true or false,
      "reason": "<brief reason if invalid, or accepted symptom name if valid>",
      "respiratory_relevance": 0.0
    }
  ],
  "boost": <float 0.0-0.25, total extra score from all valid symptoms>,
  "summary": "<one sentence summary>"
}

Rules:
- "love failure", "heartbreak", "stress", "bad day", "boredom" → INVALID
- "chest pain", "cough", "breathlessness", "wheezing", "fever", "sore throat", "runny nose" → VALID
- "tired", "fatigue", "weakness" → VALID (respiratory_relevance: 0.3)
- The boost value = sum of (respiratory_relevance * 0.1) for all valid symptoms, capped at 0.25
- Be strict. When in doubt, reject it."""


def validate_symptoms(raw_text: str) -> dict:
    """
    Validate free-text symptoms using Groq LLM.

    Parameters
    ----------
    raw_text : str
        Comma or newline separated symptom text from patient input.

    Returns
    -------
    dict with keys: valid, invalid, boost, summary, raw_results
    """
    # Default safe return
    default = {
        'valid': [], 'invalid': [], 'boost': 0.0,
        'summary': 'No additional symptoms provided.', 'raw_results': []
    }

    if not raw_text or not raw_text.strip():
        return default

    api_key = _load_env_key()
    if not api_key:
        print("[symptom_validator] No GROQ_API_KEY found — skipping LLM validation")
        return default

    # Split input into individual symptoms
    symptoms = [s.strip() for s in re.split(r'[,\n;]+', raw_text) if s.strip()]
    if not symptoms:
        return default

    # Cap at 10 symptoms
    symptoms = symptoms[:10]
    symptom_list = "\n".join(f"- {s}" for s in symptoms)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Validate these symptoms:\n{symptom_list}"}
            ],
            temperature=0.1,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            print(f"[symptom_validator] Could not parse JSON from: {content[:200]}")
            return default

        data = json.loads(json_match.group())
        results = data.get('results', [])

        valid   = [r['reason'] for r in results if r.get('valid')]
        invalid = [
            {'symptom': r['symptom'], 'reason': r.get('reason', 'Not a medical symptom')}
            for r in results if not r.get('valid')
        ]
        boost   = float(data.get('boost', 0.0))
        boost   = max(0.0, min(0.25, boost))   # hard cap
        summary = data.get('summary', '')

        print(f"[symptom_validator] Valid: {valid} | Invalid: {[i['symptom'] for i in invalid]} | Boost: {boost:.3f}")

        return {
            'valid':       valid,
            'invalid':     invalid,
            'boost':       boost,
            'summary':     summary,
            'raw_results': results,
        }

    except Exception as e:
        print(f"[symptom_validator] Error: {e}")
        return default
