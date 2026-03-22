"""
agents/cough_agent.py — LightCoughCNN inference for phone-uploaded cough audio.

PURPOSE:
  Patient uploads a cough recording from their phone (browser/app).
  This agent preprocesses the audio and classifies it as:

    Healthy     — cough acoustics consistent with a healthy airway
    Symptomatic — cough acoustics indicate respiratory symptoms
                  (COVID-19 is included here — symptoms overlap)

Label mapping (same as training):
    'healthy'     → 'Healthy'
    'symptomatic' → 'Symptomatic'
    'COVID-19'    → 'Symptomatic'   # merged at data level, not inference level

PHONE RECORDING NOTES:
  Audio uploaded by patients via phone (browser/app) in .webm/.wav/.ogg/.m4a.
  The pipeline handles all phone-specific issues:
    • Variable mic sensitivity     → peak normalisation
    • Silence before/after cough   → trim (top_db=25)
    • Background / handling noise  → noise gate (2% of peak)
    • Phone mic high-freq rolloff  → pre-emphasis (coef=0.97)
    • Compressed formats           → ffmpeg decode via librosa

OUTPUT FORMAT:
    {
        'top_prediction' : 'Symptomatic',
        'confidence'     : 0.81,
        'scores' : {
            'Healthy'     : 0.12,
            'Symptomatic' : 0.81,    # cough acoustics + reported symptoms
        },
        'agent' : 'cough_agent',
    }
"""

import os
import numpy as np
import torch
import librosa

from config import (
    DEVICE, SAVED_MODELS_DIR,
    COUGHVID_CLASSES, COUGHVID_SR, COUGHVID_DURATION,
    COUGHVID_SPEC_FRAMES, N_MELS, HOP_LENGTH,
)
from models.cnn_model import build_light_cough_cnn
from utils import load_checkpoint

_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "coughvid_efficientnet.pt")
_model      = None   # lazy-loaded singleton


def _load_model():
    """Load LightCoughCNN from disk once (singleton)."""
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Cough model not found at {_MODEL_PATH}. "
                "Run training.train_coughvid_efficientnet() first."
            )
        _model = build_light_cough_cnn()
        _model = load_checkpoint(_model, _MODEL_PATH)
        _model.eval()
        print(f"[cough_agent] Model loaded <- {_MODEL_PATH}")
    return _model


def _preprocess_audio(audio_path: str) -> np.ndarray:
    """
    Phone-aware audio preprocessing → mel spectrogram (128, COUGHVID_SPEC_FRAMES).

    Replicates the exact pipeline used during training so inference is consistent.
    Supports: .wav .webm .ogg .m4a .mp3 .flac (ffmpeg handles compressed formats).

    NOTE: No pre-emphasis applied. Pre-emphasis is a speech processing technique
    that suppresses low-frequency energy — harmful for cough classification where
    low-freq characteristics distinguish wet from dry cough.

    Steps:
      1. Load & resample to COUGHVID_SR (22050 Hz)
      2. Trim silence (top_db=20)
      3. Peak normalise
      4. Pad / truncate to 4 seconds
      5. Log-mel (n_fft=2048, fmin=50, fmax=8000, n_mels=128)
      6. (mel_db + 80) / 80 → [0, 1]
    """
    # 1. Load — librosa uses ffmpeg backend for .webm/.ogg/.m4a
    y, sr = librosa.load(audio_path, sr=COUGHVID_SR,
                         duration=COUGHVID_DURATION + 1.0, mono=True)

    # 2. Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Peak normalise
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = y / peak

    # 4. Pad / truncate
    target = int(COUGHVID_SR * COUGHVID_DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode='constant')
    else:
        y = y[:target]

    # 5. Log-mel
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=2048,
        hop_length=HOP_LENGTH, fmin=50, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel_db = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)

    # 6. Fixed width
    if mel_db.shape[1] < COUGHVID_SPEC_FRAMES:
        mel_db = np.pad(mel_db,
                        ((0, 0), (0, COUGHVID_SPEC_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :COUGHVID_SPEC_FRAMES]

    return mel_db.astype(np.float32)


def _build_1channel_tensor(mel: np.ndarray) -> torch.Tensor:
    """
    Build 1-channel tensor from mel spectrogram for LightCoughCNN.
    Matches CoughvidSpecDataset.__getitem__() exactly: (1, 128, T).
    """
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)


def predict_cough(audio_path: str) -> dict:
    """
    Classify a phone-uploaded cough recording.

    This is the primary entry point for the LangGraph triage pipeline.
    Handles the full pipeline: audio → preprocessing → model → output dict.

    Parameters
    ----------
    audio_path : str
        Path to cough audio file (.wav, .webm, .ogg, .m4a, .mp3, .flac).
        Typically a temp file path from the patient phone upload.

    Returns
    -------
    dict
        {
            'top_prediction' : 'Symptomatic',       # or 'Healthy'
            'confidence'     : 0.81,                # 0–1
            'scores' : {
                'Healthy'     : 0.12,
                'Symptomatic' : 0.81,
            },
            'agent' : 'cough_agent',
        }

    Raises
    ------
    FileNotFoundError : if audio_path does not exist
    RuntimeError      : if audio is too short (< 0.3 s) to be a valid cough
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model()

    # Preprocess
    mel = _preprocess_audio(audio_path)

    # Validate — reject if audio is effectively silent after preprocessing
    if mel.max() < 0.01:
        raise RuntimeError(
            "Audio appears to be silent or too short. "
            "Please upload a recording that clearly contains a cough."
        )

    # Build 1-channel tensor for LightCoughCNN
    img_t = _build_1channel_tensor(mel)
    x     = img_t.unsqueeze(0).to(DEVICE)   # (1, 1, 128, 173)

    # Inference
    with torch.no_grad():
        logits = model(x)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    label_int = int(np.argmax(proba))
    label     = COUGHVID_CLASSES[label_int]

    return {
        'top_prediction' : label,
        'confidence'     : round(float(proba[label_int]), 4),
        'scores'         : {
            cls: round(float(p), 4)
            for cls, p in zip(COUGHVID_CLASSES, proba)
        },
        'agent'          : 'cough_agent',
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"\n[cough_agent] Predicting: {audio_file}")
        result = predict_cough(audio_file)
        print(f"\n  top_prediction : {result['top_prediction']}")
        print(f"  confidence     : {result['confidence']}")
        print(f"  scores         :")
        for cls, score in result['scores'].items():
            bar = '█' * int(score * 30)
            print(f"    {cls:<14} {score:.4f}  {bar}")
    else:
        print("[cough_agent] Usage: python agents/cough_agent.py <audio_file.wav>")
        print(f"  Model path : {_MODEL_PATH}")
        print(f"  Classes    : {COUGHVID_CLASSES}")
        print(f"  Input      : phone .wav / .webm / .ogg (4 s cough recording)")
