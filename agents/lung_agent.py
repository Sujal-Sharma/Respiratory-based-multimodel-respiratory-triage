"""
agents/lung_agent.py — MultiTaskEfficientNet inference wrapper.

Returns both disease and sound predictions in a single forward pass.
Used by the LangGraph pipeline (Phase 3).
"""

import os
import numpy as np
import torch
from torch.cuda.amp import autocast

from config import (
    DEVICE, SAVED_MODELS_DIR,
    LUNG_DISEASE_CLASSES, LUNG_SOUND_CLASSES,
)
from models.cnn_model import build_multitask_efficientnet
from preprocessing import extract_melspectrogram
from utils import load_checkpoint

_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "multitask_efficientnet.pt")
_model      = None   # lazy-loaded singleton


def _load_model():
    """Load MultiTaskEfficientNet from disk (once, singleton)."""
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"MultiTask model not found at {_MODEL_PATH}. "
                "Run training.train_multitask_efficientnet() first."
            )
        _model = build_multitask_efficientnet(pretrained=False)
        _model = load_checkpoint(_model, _MODEL_PATH)
        _model.eval()
        print(f"[lung_agent] Model loaded <- {_MODEL_PATH}")
    return _model


def _infer(mel: np.ndarray) -> dict:
    """
    Run inference on a mel spectrogram array.

    Parameters
    ----------
    mel : np.ndarray shape (N_MELS, T) float16 or float32

    Returns
    -------
    dict with 'disease' and 'sound' sub-dicts, each containing:
        label, label_int, confidence, probabilities
    """
    model = _load_model()

    mel_tensor = (
        torch.tensor(mel.astype(np.float32), dtype=torch.float32)
        .unsqueeze(0).unsqueeze(0)   # (1, 1, N_MELS, T)
        .to(DEVICE)
    )

    with torch.no_grad():
        with autocast():
            dis_out, snd_out = model(mel_tensor)
        dis_proba = torch.softmax(dis_out.float(), dim=1).cpu().numpy()[0]
        snd_proba = torch.softmax(snd_out.float(), dim=1).cpu().numpy()[0]

    dis_int  = int(np.argmax(dis_proba))
    snd_int  = int(np.argmax(snd_proba))

    return {
        'disease': {
            'label':         LUNG_DISEASE_CLASSES[dis_int],
            'label_int':     dis_int,
            'confidence':    round(float(dis_proba[dis_int]), 4),
            'probabilities': {
                cls: round(float(p), 4)
                for cls, p in zip(LUNG_DISEASE_CLASSES, dis_proba)
            },
        },
        'sound': {
            'label':         LUNG_SOUND_CLASSES[snd_int],
            'label_int':     snd_int,
            'confidence':    round(float(snd_proba[snd_int]), 4),
            'probabilities': {
                cls: round(float(p), 4)
                for cls, p in zip(LUNG_SOUND_CLASSES, snd_proba)
            },
        },
        'agent': 'lung_agent',
    }


def predict_from_audio(audio_path: str) -> dict:
    """
    Predict disease and sound class from a raw audio file.

    Parameters
    ----------
    audio_path : path to .wav / .mp3 / .flac audio file

    Returns
    -------
    dict:
        {
          'disease': {'label': 'COPD', 'confidence': 0.84, 'probabilities': {...}},
          'sound':   {'label': 'Crackle', 'confidence': 0.79, 'probabilities': {...}},
          'agent':   'lung_agent'
        }
    """
    mel = extract_melspectrogram(audio_path)
    return _infer(mel)


def predict_from_spectrogram(spec: np.ndarray) -> dict:
    """
    Predict from a pre-extracted mel spectrogram array.

    Parameters
    ----------
    spec : np.ndarray shape (N_MELS, T), values in [0, 1]

    Returns
    -------
    Same dict structure as predict_from_audio().
    """
    return _infer(spec)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agents/lung_agent.py <audio_file>")
    else:
        result = predict_from_audio(sys.argv[1])
        print("\n[lung_agent] Prediction:")
        print(f"  Disease : {result['disease']['label']} "
              f"(confidence={result['disease']['confidence']})")
        print(f"  Sound   : {result['sound']['label']} "
              f"(confidence={result['sound']['confidence']})")
        print(f"  Disease probs: {result['disease']['probabilities']}")
        print(f"  Sound probs  : {result['sound']['probabilities']}")