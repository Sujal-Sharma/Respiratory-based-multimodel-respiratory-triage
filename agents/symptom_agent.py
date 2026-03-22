"""
agents/symptom_agent.py — XGBoost inference wrapper for COUGHVID.

This agent receives patient metadata and returns:
  - predicted class label
  - confidence probabilities
  - structured result dict for the LangGraph pipeline (Phase 3)
"""

import os
import pickle
import numpy as np
from config import SAVED_MODELS_DIR, COUGHVID_CLASSES

_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "xgboost_coughvid.pkl")
_model      = None   # lazy-loaded singleton


def _load_model():
    """Load XGBoost model from disk (once)."""
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"XGBoost model not found at {_MODEL_PATH}. "
                "Run training.train_xgboost() first."
            )
        with open(_MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        print(f"[symptom_agent] Model loaded <- {_MODEL_PATH}")
    return _model


def predict_symptom(
    age: float,
    gender: str,
    fever_muscle_pain: bool,
    respiratory_condition: bool,
    cough_detected: float,
    dyspnea: bool = False,
    wheezing: bool = False,
    congestion: bool = False
) -> dict:
    """
    Predict COVID-19 / healthy / symptomatic from patient metadata.

    Parameters
    ----------
    age                   : patient age (years)
    gender                : 'male' | 'female' | 'unknown'
    fever_muscle_pain     : True/False
    respiratory_condition : True/False
    cough_detected        : cough detection confidence score 0–1
    dyspnea               : difficulty breathing True/False
    wheezing              : True/False
    congestion            : True/False

    Returns
    -------
    dict with:
      'label'       : predicted class string (e.g. 'COVID-19')
      'label_int'   : integer label
      'confidence'  : confidence of top prediction (0–1)
      'probabilities': dict {class: prob}
      'agent'       : 'symptom_agent'
    """
    model = _load_model()

    # Encode features (same as preprocessing.py)
    age_norm  = min(float(age), 100.0) / 100.0

    gender_map = {'male': 0.0, 'female': 1.0}
    gender_enc = gender_map.get(str(gender).strip().lower(), 0.5)

    def _bool(v):
        return 1.0 if v else 0.0

    meta = [
        age_norm,
        gender_enc,
        _bool(fever_muscle_pain),
        _bool(respiratory_condition),
        float(cough_detected),
        _bool(dyspnea),
        _bool(wheezing),
        _bool(congestion),
    ]
    # XGBoost was trained on 168 features (8 meta + 160 MFCC).
    # When called without audio, pad MFCC columns with zeros.
    mfcc_pad = [0.0] * 160
    features = np.array([meta + mfcc_pad], dtype=np.float32)

    proba     = model.predict_proba(features)[0]
    label_int = int(np.argmax(proba))
    label     = COUGHVID_CLASSES[label_int]
    confidence = float(proba[label_int])

    return {
        'label':         label,
        'label_int':     label_int,
        'confidence':    round(confidence, 4),
        'probabilities': {cls: round(float(p), 4)
                          for cls, p in zip(COUGHVID_CLASSES, proba)},
        'agent':         'symptom_agent',
    }


def predict_from_array(features: np.ndarray) -> dict:
    """
    Predict from a pre-encoded feature array of shape (8,) or (1, 8).
    Useful for batch inference or evaluation replay.
    """
    model = _load_model()

    if features.ndim == 1:
        features = features.reshape(1, -1)

    proba     = model.predict_proba(features.astype(np.float32))[0]
    label_int = int(np.argmax(proba))
    label     = COUGHVID_CLASSES[label_int]

    return {
        'label':         label,
        'label_int':     label_int,
        'confidence':    round(float(proba[label_int]), 4),
        'probabilities': {cls: round(float(p), 4)
                          for cls, p in zip(COUGHVID_CLASSES, proba)},
        'agent':         'symptom_agent',
    }


if __name__ == "__main__":
    # Example inference
    result = predict_symptom(
        age=45,
        gender='male',
        fever_muscle_pain=True,
        respiratory_condition=False,
        cough_detected=0.95,
        dyspnea=True,
        wheezing=False,
        congestion=False
    )
    print("\n[symptom_agent] Example prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
