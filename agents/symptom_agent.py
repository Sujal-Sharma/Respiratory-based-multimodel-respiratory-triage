"""
agents/symptom_agent.py — XGBoost inference wrapper for COUGHVID.

This agent receives patient metadata and returns:
  - predicted class label
  - confidence probabilities
  - structured result dict for the LangGraph pipeline

Model format: dict with model + feature selection indices + threshold.
"""

import os
import pickle
import numpy as np
from config import SAVED_MODELS_DIR, COUGHVID_CLASSES, XGB_N_AUDIO_FEATURES

_MODEL_PATH   = os.path.join(SAVED_MODELS_DIR, "xgboost_coughvid.pkl")
_model        = None   # lazy-loaded singleton
_selected_idx = None   # feature selection indices
_threshold    = 0.5    # classification threshold (tuned during training)


def _load_model():
    """Load XGBoost model from disk (once)."""
    global _model, _selected_idx, _threshold
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"XGBoost model not found at {_MODEL_PATH}. "
                "Run training.train_xgboost() first."
            )
        with open(_MODEL_PATH, 'rb') as f:
            data = pickle.load(f)

        _model        = data['model']
        _selected_idx = data.get('selected_idx', None)
        _threshold    = data.get('threshold', 0.5)
        n_sel = len(_selected_idx) if _selected_idx is not None else '?'
        print(f"[symptom_agent] model loaded <- {_MODEL_PATH}")
        print(f"  {n_sel} selected features, threshold={_threshold:.3f}")
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
    Predict Healthy / Symptomatic from patient metadata.

    When called from the pipeline (without audio), audio features are
    zero-padded. Metadata alone gives ~51% confidence -- acts as
    supplementary signal, not primary classifier.

    Returns
    -------
    dict with label, label_int, confidence, probabilities, agent
    """
    model = _load_model()

    # Encode metadata (same order as preprocessing.py)
    age_norm   = min(float(age), 100.0) / 100.0
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

    # Zero-pad audio features (no audio available in pipeline mode)
    audio_pad = [0.0] * XGB_N_AUDIO_FEATURES  # 256

    full_features = np.array([meta + audio_pad], dtype=np.float32)

    # Apply feature selection
    if _selected_idx is not None:
        full_features = full_features[:, _selected_idx]

    proba = model.predict_proba(full_features)[0]

    # Apply optimised threshold (tuned to maximise Macro F1)
    # proba[1] = P(Symptomatic)
    if proba[1] >= _threshold:
        label_int = 1
    else:
        label_int = 0

    label      = COUGHVID_CLASSES[label_int]
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
    Predict from a pre-encoded feature array (full 264-feature vector).
    Feature selection is applied automatically.
    """
    model = _load_model()

    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = features.astype(np.float32)

    # Apply feature selection
    if _selected_idx is not None:
        features = features[:, _selected_idx]

    proba = model.predict_proba(features)[0]

    if proba[1] >= _threshold:
        label_int = 1
    else:
        label_int = 0

    label = COUGHVID_CLASSES[label_int]

    return {
        'label':         label,
        'label_int':     label_int,
        'confidence':    round(float(proba[label_int]), 4),
        'probabilities': {cls: round(float(p), 4)
                          for cls, p in zip(COUGHVID_CLASSES, proba)},
        'agent':         'symptom_agent',
    }


if __name__ == "__main__":
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
