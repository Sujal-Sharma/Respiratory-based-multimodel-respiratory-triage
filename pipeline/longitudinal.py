"""
pipeline/longitudinal.py — Longitudinal score fusion for Tier 1 monitoring.

Combines 3 signals into a single Longitudinal Risk Score (0-1):
  1. Symptom Severity Index  (50%) — CAT-style validated composite
  2. Voice Health Index       (35%) — within-person acoustic deviation
  3. Cough Drift Score        (15%) — OPERA-CT embedding drift from baseline

MCID (Minimum Clinically Important Difference) = 0.05
A change >= 0.05 per session is considered clinically meaningful.

Citations:
  - Kon et al. (2014) Thorax — CAT MCID
  - GOLD 2024 — longitudinal COPD monitoring
"""

import numpy as np

# Fusion weights
W_SYMPTOM = 0.50
W_VOICE   = 0.35
W_DRIFT   = 0.15

# MCID threshold for flagging meaningful change
MCID = 0.05


def compute_longitudinal_score(symptom_index: float,
                                voice_index: float,
                                drift_score: float) -> float:
    """
    Fuse 3 signals into a single longitudinal risk score (0-1).

    Parameters
    ----------
    symptom_index : 0-1, CAT-style symptom severity
    voice_index   : 0-1, within-person voice deviation (0 if no voice recording)
    drift_score   : 0-1, OPERA cough embedding drift (0 if no cough recording)

    Returns
    -------
    float in [0, 1]
    """
    # Adjust weights if voice/drift are missing (0 = not recorded this session)
    if voice_index == 0.0 and drift_score == 0.0:
        # Only symptoms available
        return round(float(symptom_index), 4)

    if voice_index == 0.0:
        # Symptoms + drift only — redistribute voice weight to symptom
        score = (W_SYMPTOM + W_VOICE) * symptom_index + W_DRIFT * drift_score
    elif drift_score == 0.0:
        # Symptoms + voice only — redistribute drift weight to symptom
        score = (W_SYMPTOM + W_DRIFT) * symptom_index + W_VOICE * voice_index
    else:
        # Full composite
        score = (W_SYMPTOM * symptom_index +
                 W_VOICE   * voice_index   +
                 W_DRIFT   * drift_score)

    return round(min(float(score), 1.0), 4)


def compute_cough_drift(current_embedding: np.ndarray,
                        baseline_embedding: np.ndarray) -> float:
    """
    Compute cosine drift between current and baseline OPERA-CT embeddings.

    Returns float in [0, 1]:
      0.0 = identical to baseline
      1.0 = maximally different
    """
    if current_embedding is None or baseline_embedding is None:
        return 0.0
    cur = current_embedding.astype(np.float32)
    bas = baseline_embedding.astype(np.float32)
    norm_cur = np.linalg.norm(cur)
    norm_bas = np.linalg.norm(bas)
    if norm_cur == 0 or norm_bas == 0:
        return 0.0
    cos_sim = float(np.dot(cur, bas) / (norm_cur * norm_bas))
    drift = 1.0 - cos_sim
    return round(float(np.clip(drift, 0.0, 1.0)), 4)


def interpret_score(score: float) -> dict:
    """
    Interpret a longitudinal score into a clinical label.

    Returns dict with: label, color, description
    """
    if score < 0.20:
        return {
            'label':       'Stable',
            'color':       'green',
            'description': 'No significant respiratory concern.',
        }
    elif score < 0.40:
        return {
            'label':       'Mild Concern',
            'color':       'yellow',
            'description': 'Mild symptoms. Monitor closely.',
        }
    elif score < 0.60:
        return {
            'label':       'Moderate Concern',
            'color':       'orange',
            'description': 'Moderate respiratory symptoms. GP review recommended.',
        }
    elif score < 0.80:
        return {
            'label':       'High Concern',
            'color':       'red',
            'description': 'Significant symptoms. GP appointment this week.',
        }
    else:
        return {
            'label':       'Critical Concern',
            'color':       'darkred',
            'description': 'Severe symptoms. Seek medical attention today.',
        }
