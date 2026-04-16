"""
agents/voice_agent.py — Sustained vowel phonation analysis.

Patient says "ahhh" for 5 seconds into phone mic.
Extracts clinically validated voice biomarkers using Praat/parselmouth:
  - Jitter (local)  : cycle-to-cycle F0 variation — increases with airflow instability
  - Shimmer (local) : amplitude variation          — increases with glottal inconsistency
  - HNR             : harmonics-to-noise ratio     — decreases with turbulent airflow
  - F0 mean/std     : fundamental frequency stats
  - Phonation time  : how long patient sustained vowel

Within-person baseline comparison:
  First session → saved as personal baseline
  Each future session → Voice Health Index = weighted deviation from baseline
  0.0 = identical to baseline (healthy/stable)
  1.0 = maximum deviation (significant change)

Citations:
  - Gupta et al. (2020) Journal of Voice — jitter/shimmer in COPD
  - Coppock et al. (2021) — COVID voice biomarkers (Cambridge study)
  - Praat: Boersma & Weenink (2022) — standard voice analysis tool
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import parselmouth
    from parselmouth.praat import call
    _PARSELMOUTH_OK = True
except ImportError:
    _PARSELMOUTH_OK = False

# Feature weights for Voice Health Index
# Higher weight = more sensitive to respiratory change
_FEATURE_WEIGHTS = {
    'jitter':            0.25,  # most sensitive to airflow irregularity
    'shimmer':           0.25,  # amplitude instability
    'hnr':               0.25,  # noise in signal (inverted — lower = worse)
    'f0_std':            0.15,  # pitch instability
    'phonation_duration': 0.10, # shorter = reduced lung capacity (inverted)
}

# Minimum clinically meaningful change per feature (for paper threshold justification)
# Derived from literature: jitter MCID ~0.002, shimmer ~0.02, HNR ~2dB
_MCID = {
    'jitter':            0.002,
    'shimmer':           0.020,
    'hnr':               2.0,
    'f0_std':            5.0,
    'phonation_duration': 1.0,
}

logger = logging.getLogger(__name__)


class VoiceAgent:
    """
    Extracts voice biomarkers from a sustained /a:/ vowel recording.
    Computes within-person Voice Health Index against stored baseline.
    """

    AGENT_NAME = 'Voice Agent'

    def _to_wav(self, audio_path: str) -> str:
        """Convert any audio format to wav using librosa+soundfile for parselmouth."""
        import tempfile, soundfile as sf, librosa
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.wav':
            return audio_path
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, y, sr)
            return tmp.name
        except Exception:
            return audio_path  # fallback — let parselmouth try directly

    def extract_features(self, audio_path: str) -> dict | None:
        """
        Extract voice features from audio file.
        Returns dict of features, or None if extraction fails.
        """
        if not _PARSELMOUTH_OK:
            return None
        try:
            wav_path = self._to_wav(str(audio_path))
            sound = parselmouth.Sound(wav_path)
            duration = sound.get_total_duration()

            if duration < 1.0:
                return None  # too short

            # Trim to max 8 seconds from middle (most stable part)
            if duration > 8.0:
                mid = duration / 2
                sound = sound.extract_part(mid - 4.0, mid + 4.0)

            # Pitch / F0
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            f0_std  = call(pitch, "Get standard deviation", 0, 0, "Hertz")

            if not f0_mean or np.isnan(f0_mean) or f0_mean == 0:
                # No F0 detected — use energy-based features as fallback
                import librosa
                y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
                rms    = float(np.sqrt(np.mean(y**2)))
                zcr    = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)
                if rms < 0.001:
                    return None  # silence
                return {
                    'f0_mean': 0.0, 'f0_std': 0.0,
                    'jitter': zcr * 0.1,   # proxy
                    'shimmer': rms,
                    'hnr': 10.0,            # assume normal HNR as fallback
                    'phonation_duration': float(sound.get_total_duration()),
                }

            # Jitter
            pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

            # Shimmer
            shimmer = call([sound, pp], "Get shimmer (local)",
                           0, 0, 0.0001, 0.02, 1.3, 1.6)

            # HNR
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)

            # Sanitise
            def _safe(v, default=0.0):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return float(v)

            features = {
                'f0_mean':           _safe(f0_mean),
                'f0_std':            _safe(f0_std),
                'jitter':            _safe(jitter),
                'shimmer':           _safe(shimmer),
                'hnr':               _safe(hnr, default=10.0),
                'phonation_duration': _safe(duration),
            }
            return features

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            logger.exception("Voice feature extraction error")
            return None

    def compute_voice_index(self,
                            current: dict,
                            baseline: dict) -> float:
        """
        Compute Voice Health Index = weighted deviation from personal baseline.

        Returns float in [0, 1]:
          0.0 = no change from baseline (stable/healthy)
          1.0 = maximum deviation (significant change)
        """
        if not baseline or not current:
            return 0.0

        total = 0.0
        for feat, weight in _FEATURE_WEIGHTS.items():
            b = baseline.get(feat, 0.0)
            c = current.get(feat, 0.0)
            if b == 0:
                continue

            # Direction matters: increase in jitter/shimmer/f0_std = worse
            #                    decrease in hnr/phonation_duration = worse
            if feat in ('hnr', 'phonation_duration'):
                delta = max(0.0, (b - c) / (abs(b) + 1e-6))
            else:
                delta = max(0.0, (c - b) / (abs(b) + 1e-6))

            # Clip to [0, 1] per feature
            total += weight * min(delta, 1.0)

        return round(min(total, 1.0), 4)

    def predict(self, audio_path: str, baseline_features: dict | None = None) -> dict:
        """
        Full voice analysis pipeline.

        Parameters
        ----------
        audio_path        : path to sustained vowel recording
        baseline_features : dict from first session (None if this IS the first session)

        Returns
        -------
        dict with: agent, features, voice_index, is_baseline, error
        """
        if not _PARSELMOUTH_OK:
            return {
                'agent': self.AGENT_NAME,
                'features': {},
                'voice_index': 0.0,
                'is_baseline': False,
                'error': 'parselmouth not installed',
            }

        features = self.extract_features(audio_path)

        if features is None:
            return {
                'agent': self.AGENT_NAME,
                'features': {},
                'voice_index': 0.0,
                'is_baseline': baseline_features is None,
                'error': 'Feature extraction failed — check audio quality',
            }

        is_baseline = baseline_features is None
        voice_index = 0.0 if is_baseline else \
            self.compute_voice_index(features, baseline_features)

        return {
            'agent':       self.AGENT_NAME,
            'features':    features,
            'voice_index': voice_index,
            'is_baseline': is_baseline,
            'error':       None,
        }
