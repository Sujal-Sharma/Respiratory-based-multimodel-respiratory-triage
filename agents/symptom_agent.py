"""
agents/symptom_agent.py — XGBoost symptom + metadata agent.

Updated from original: now outputs disease-specific probability hints
(copd_probability_hint, pneumonia_probability_hint) in addition to
the binary healthy/symptomatic prediction.

Keeps the same XGBoost model format (loaded from xgboost_coughvid.pkl)
and the same 8-feature metadata vector. The disease hints are derived
from clinical heuristics applied to the symptom pattern.
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_DEFAULT_MODEL_PATH = './saved_models/xgboost_coughvid.pkl'

_model        = None
_selected_idx = None
_threshold    = 0.5


def _load_model(model_path: str = _DEFAULT_MODEL_PATH):
    global _model, _selected_idx, _threshold
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}. "
                "Run XGBoost training first."
            )
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        _model        = data['model']
        _selected_idx = data.get('selected_idx', None)
        _threshold    = data.get('threshold', 0.5)
        n_sel = len(_selected_idx) if _selected_idx is not None else '?'
        print(f"[SymptomAgent] Loaded {model_path} | "
              f"{n_sel} selected features | threshold={_threshold:.3f}")
    return _model


class SymptomAgent:
    """
    Symptom-based risk agent using XGBoost on 8 metadata features.

    Outputs binary healthy/symptomatic prediction plus disease-specific
    probability hints based on clinical feature patterns.
    """

    AGENT_NAME = 'Symptom Agent'

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model        = None
        self._selected_idx = None
        self._threshold    = 0.5
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"XGBoost model not found at {self.model_path}."
            )
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self._model        = data['model']
        self._selected_idx = data.get('selected_idx', None)
        self._threshold    = data.get('threshold', 0.5)
        print(f"[SymptomAgent] Loaded | threshold={self._threshold:.3f}")

    def build_features(self,
                       age: float,
                       gender: str,
                       fever_muscle_pain: bool,
                       dyspnea: bool,
                       wheezing: bool,
                       congestion: bool,
                       resp_condition: bool,
                       cough_severity: float) -> np.ndarray:
        """Build 8-feature metadata vector (same order as training)."""
        gender_enc = {'male': 0.0, 'female': 1.0}.get(
            str(gender).strip().lower(), 0.5
        )
        return np.array([
            float(age) / 100.0,
            gender_enc,
            1.0 if fever_muscle_pain else 0.0,
            1.0 if dyspnea           else 0.0,
            1.0 if wheezing          else 0.0,
            1.0 if congestion        else 0.0,
            1.0 if resp_condition    else 0.0,
            float(cough_severity) / 10.0,
        ], dtype=np.float32)

    def predict(self,
                age: float,
                gender: str,
                fever_muscle_pain: bool,
                dyspnea: bool,
                wheezing: bool,
                congestion: bool,
                resp_condition: bool,
                cough_severity: float) -> dict:
        """
        Predict symptomatic risk from patient metadata.

        Returns
        -------
        dict with keys:
            agent, symptomatic_probability, copd_probability_hint,
            pneumonia_probability_hint, detected, confidence, error
        """
        try:
            features = self.build_features(
                age, gender, fever_muscle_pain, dyspnea,
                wheezing, congestion, resp_condition, cough_severity
            )

            if self._selected_idx is not None:
                features = features[self._selected_idx]

            proba            = self._model.predict_proba(features.reshape(1, -1))[0]
            symptomatic_prob = float(proba[1])

            # Clinical heuristics: derive disease-specific hints from symptom pattern
            # COPD indicators: prior respiratory condition + dyspnea (chronic obstruction)
            copd_factor = 0.7 if (resp_condition and dyspnea) else 0.3
            # Pneumonia indicators: fever + dyspnea (acute infection pattern)
            pneu_factor = 0.6 if (fever_muscle_pain and dyspnea) else 0.2

            copd_hint = min(symptomatic_prob * copd_factor, 1.0)
            pneu_hint = min(symptomatic_prob * pneu_factor, 1.0)

            return {
                'agent':                    self.AGENT_NAME,
                'symptomatic_probability':  round(symptomatic_prob, 4),
                'copd_probability_hint':    round(copd_hint, 4),
                'pneumonia_probability_hint': round(pneu_hint, 4),
                'detected':                 symptomatic_prob >= self._threshold,
                'confidence':               round(symptomatic_prob, 4),
                'error':                    None,
            }

        except Exception as e:
            return {
                'agent':                    self.AGENT_NAME,
                'symptomatic_probability':  0.0,
                'copd_probability_hint':    0.0,
                'pneumonia_probability_hint': 0.0,
                'detected':                 False,
                'confidence':               0.0,
                'error':                    str(e),
            }


# ── Module-level convenience function (preserves backward compatibility) ─────

def predict_symptom(age, gender, fever_muscle_pain, respiratory_condition,
                    cough_detected=0.5, dyspnea=False, wheezing=False,
                    congestion=False, audio_path=None) -> dict:
    """
    Backward-compatible function for triage_graph.py and app.py.
    Delegates to SymptomAgent.predict().
    """
    agent = SymptomAgent()
    return agent.predict(
        age=age,
        gender=gender,
        fever_muscle_pain=fever_muscle_pain,
        dyspnea=dyspnea,
        wheezing=wheezing,
        congestion=congestion,
        resp_condition=respiratory_condition,
        cough_severity=float(cough_detected) * 10,
    )


if __name__ == '__main__':
    agent  = SymptomAgent()
    result = agent.predict(
        age=55, gender='male',
        fever_muscle_pain=False, dyspnea=True,
        wheezing=True, congestion=False,
        resp_condition=True, cough_severity=7,
    )
    print("[SymptomAgent] Example prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
