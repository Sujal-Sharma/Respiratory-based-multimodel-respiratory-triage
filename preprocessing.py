"""
preprocessing.py — Feature extraction for all datasets.

Functions:
  preprocess_coughvid()      → tabular features for XGBoost (COUGHVID)
  preprocess_icbhi()         → mel spectrograms from breath cycles (ICBHI)
  preprocess_kauh()          → mel spectrograms from recordings (KAUH)
  preprocess_hf_lung()       → mel spectrograms from HF Lung V1 manifest
  build_multitask_dataset()  → combines ICBHI + KAUH + HF Lung → multitask_labels.csv
  extract_melspectrogram()   → shared mel extractor (used by all audio functions)
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from config import (
    COUGHVID_METADATA, COUGHVID_AUDIO_DIR, COUGHVID_LABELS_CSV,
    COUGHVID_SPEC_LABELS_CSV, COUGHVID_SPEC_DIR,
    COUGHVID_SAMPLES_PER_CLASS, COUGHVID_CLASSES, COUGHVID_LABEL_REMAP,
    COUGHVID_SR, COUGHVID_DURATION, COUGHVID_SPEC_FRAMES,
    ICBHI_AUDIO_DIR, ICBHI_DIAGNOSIS, ICBHI_DISEASE_MAP,
    KAUH_AUDIO_DIR, KAUH_DISEASE_MAP, KAUH_SOUND_MAP,
    HF_LUNG_MANIFEST,
    SPEC_DIR, MULTITASK_LABELS_CSV,
    LUNG_DISEASE_CLASSES, LUNG_SOUND_CLASSES,
    SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, N_FFT, SPEC_TIME_FRAMES,
    N_MFCC,
)

os.makedirs(COUGHVID_SPEC_DIR, exist_ok=True)

warnings.filterwarnings('ignore')
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs('./data', exist_ok=True)

# Integer maps for disease and sound
_DISEASE_INT = {cls: i for i, cls in enumerate(LUNG_DISEASE_CLASSES)}
_SOUND_INT   = {cls: i for i, cls in enumerate(LUNG_SOUND_CLASSES)}


# ══════════════════════════════════════════════════════════════════════════════
# Shared mel spectrogram extractor
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass_filter(y: np.ndarray, sr: int,
                     low_hz: float = 200.0, high_hz: float = 1800.0,
                     order: int = 3) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to keep only lung-sound frequencies.

    200–1800 Hz is the clinically relevant range for respiratory sounds.
    Removing DC offset and high-frequency noise improves mel-spectrogram
    quality for pretrained CNNs.
    """
    nyq  = sr / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq
    sos  = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def extract_melspectrogram(audio_path: str, offset: float = 0.0,
                           duration: float = None) -> np.ndarray:
    """
    Load audio and extract a mel spectrogram.

    Pipeline (matches top ICBHI papers):
      1. Resample to SAMPLE_RATE (16 kHz)
      2. Bandpass 200–1800 Hz  — removes DC + out-of-band noise
      3. Pad / truncate to DURATION seconds
      4. Log-mel spectrogram  (n_fft=1024, hop=512, n_mels=128)
      5. power_to_db with top_db=80  → range [−80, 0]
      6. Shift to [0, 1]: (mel_db + 80) / 80  — fixed, not per-sample

    Returns
    -------
    mel : np.ndarray shape (N_MELS, SPEC_TIME_FRAMES) dtype float16
    """
    dur = duration if duration is not None else DURATION
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, offset=offset,
                         duration=dur, mono=True)

    # 1. Bandpass filter — keeps 200–1800 Hz (lung sound range)
    y = _bandpass_filter(y, sr)

    # 2. Pad / truncate to fixed length
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # 3. Log-mel spectrogram (n_fft=1024 for better temporal resolution)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=200, fmax=1800
    )

    # 4. Convert to dB with fixed 80 dB dynamic range
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

    # 5. Normalise to [0, 1] using fixed range — preserves relative amplitude
    #    mel_db is in [−80, 0]; (mel_db + 80) / 80 → [0, 1]
    mel_db = (mel_db + 80.0) / 80.0
    mel_db = np.clip(mel_db, 0.0, 1.0)

    # 6. Ensure fixed width
    if mel_db.shape[1] < SPEC_TIME_FRAMES:
        pad_w = SPEC_TIME_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_w)), mode='constant')
    else:
        mel_db = mel_db[:, :SPEC_TIME_FRAMES]

    return mel_db.astype(np.float16)


# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID — Metadata preprocessing for XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def _build_coughvid_metadata() -> pd.DataFrame:
    """
    Compile individual COUGHVID JSON files into a single metadata CSV.

    Each recording has one JSON file with fields:
      cough_detected, age, gender, respiratory_condition,
      fever_muscle_pain, status

    Saves result to COUGHVID_METADATA path and returns DataFrame.
    """
    import json, glob
    print(f"  Building metadata from JSON files in {COUGHVID_AUDIO_DIR} …")

    json_files = glob.glob(os.path.join(COUGHVID_AUDIO_DIR, "*.json"))
    print(f"  Found {len(json_files):,} JSON files")

    records = []
    for jf in tqdm(json_files, desc="Reading JSONs"):
        uuid = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            data['uuid'] = uuid
            records.append(data)
        except Exception:
            continue

    df = pd.DataFrame(records)
    df.to_csv(COUGHVID_METADATA, index=False)
    print(f"  Compiled {len(df):,} records → {COUGHVID_METADATA}")
    return df


def _encode_gender(val) -> float:
    if isinstance(val, str):
        v = val.strip().lower()
        if v == 'male':   return 0.0
        if v == 'female': return 1.0
    return 0.5


def _encode_bool(val) -> float:
    if isinstance(val, bool):   return 1.0 if val else 0.0
    if isinstance(val, str):
        v = val.strip().lower()
        if v == 'true':  return 1.0
        if v == 'false': return 0.0
    return 0.0


def _find_coughvid_audio(uuid: str) -> str:
    """Find a COUGHVID audio file by UUID, trying common extensions."""
    for ext in ['.wav', '.webm', '.ogg', '.mp3', '.flac']:
        path = os.path.join(COUGHVID_AUDIO_DIR, f"{uuid}{ext}")
        if os.path.exists(path):
            return path
    return None


def _extract_mfcc_features(audio_path: str, n_mfcc: int = N_MFCC,
                            sr: int = 16000, duration: float = 3.0) -> np.ndarray:
    """
    Extract MFCC statistical features from a cough audio file.

    Pipeline (matches Orlandic/Chaudhari COUGHVID papers):
      1. Load & resample to 16 kHz
      2. Trim leading/trailing silence (top_db=20) — important for crowd-sourced audio
      3. Peak-normalize (handles variable mic levels across devices)
      4. Pad/truncate to 3 seconds (covers ~95% of valid coughs)
      5. Compute MFCCs + delta coefficients
      6. Aggregate: mean + std per coefficient

    Returns
    -------
    features : np.ndarray shape (4 * n_mfcc,)  — 160 features if n_mfcc=40
    All-zeros on load failure.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration + 1.0, mono=True)

        # Trim silence — critical for COUGHVID crowd-sourced recordings
        y, _ = librosa.effects.trim(y, top_db=20)

        # Peak normalise — handles variable recording levels across devices
        peak = np.max(np.abs(y))
        if peak > 1e-9:
            y = y / peak

        # Pad / truncate to fixed duration (3s covers ~95% of coughs)
        target = int(sr * duration)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)), mode='constant')
        else:
            y = y[:target]

        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                      n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfcc)

        features = np.concatenate([
            mfcc.mean(axis=1),  mfcc.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
        ])
        return features.astype(np.float32)
    except Exception:
        return np.zeros(4 * n_mfcc, dtype=np.float32)


def preprocess_coughvid():
    """
    Load COUGHVID, extract metadata + MFCC audio features, balance to 2 classes.

    Label remapping:
      healthy     → Healthy
      symptomatic → Symptomatic
      COVID-19    → Symptomatic  (symptoms overlap; binary is more robust)

    Features per sample (total 168):
      Metadata (8):  age_norm, gender_enc, fever_muscle_pain_enc,
                     resp_cond_enc, cough_score, dyspnea_enc,
                     wheezing_enc, congestion_enc
      MFCC (160):    40 MFCC mean + 40 MFCC std + 40 delta mean + 40 delta std

    Samples: 1000 per class × 2 classes = 2000 total (fast to process).

    Returns
    -------
    X : np.ndarray  shape (N, 168)
    y : np.ndarray  shape (N,)
    label_map : dict {class_name: int}
    """
    print("\n" + "═" * 60)
    print("COUGHVID PREPROCESSING  (metadata + MFCC audio features)")
    print("═" * 60)

    # Build metadata CSV from JSONs if it doesn't exist yet
    if not os.path.exists(COUGHVID_METADATA):
        _build_coughvid_metadata()

    df = pd.read_csv(COUGHVID_METADATA, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Remap labels: COVID-19 → Symptomatic, healthy → Healthy
    df['status_orig'] = df['status']
    df['status']      = df['status'].map(COUGHVID_LABEL_REMAP)
    df = df[df['status'].isin(COUGHVID_CLASSES)].copy()

    # Keep only high-confidence cough recordings
    df = df[df['cough_detected'] >= 0.8].copy()
    print(f"  After filters: {len(df):,}")
    print(f"  Distribution:\n{df['status'].value_counts()}")

    # ── Metadata features ──────────────────────────────────────────
    df['age_norm']              = (df['age'].clip(0, 100).fillna(50)) / 100.0
    df['gender_enc']            = df['gender'].apply(_encode_gender)
    df['fever_muscle_pain_enc'] = df['fever_muscle_pain'].apply(_encode_bool)
    df['resp_cond_enc']         = df['respiratory_condition'].apply(_encode_bool)
    df['cough_score']           = df['cough_detected'].astype(float)
    df['dyspnea_enc']           = df['dyspnea_1'].apply(_encode_bool) if 'dyspnea_1' in df.columns else 0.0
    df['wheezing_enc']          = df['wheezing_1'].apply(_encode_bool) if 'wheezing_1' in df.columns else 0.0
    df['congestion_enc']        = df['congestion_1'].apply(_encode_bool) if 'congestion_1' in df.columns else 0.0

    META_COLS = [
        'age_norm', 'gender_enc', 'fever_muscle_pain_enc',
        'resp_cond_enc', 'cough_score',
        'dyspnea_enc', 'wheezing_enc', 'congestion_enc'
    ]
    df = df.dropna(subset=['age_norm']).reset_index(drop=True)

    # ── Balance: 1000 per class ────────────────────────────────────
    balanced_dfs = []
    for cls in COUGHVID_CLASSES:
        cls_df = df[df['status'] == cls]
        n      = min(len(cls_df), COUGHVID_SAMPLES_PER_CLASS)
        balanced_dfs.append(cls_df.sample(n=n, random_state=42))
        print(f"  '{cls}': {n:,} samples selected")
    df_bal = pd.concat(balanced_dfs).reset_index(drop=True)

    # ── Extract MFCC features from audio ──────────────────────────
    n_mfcc     = N_MFCC
    mfcc_dim   = 4 * n_mfcc  # mean+std for mfcc and delta = 4 × 40 = 160
    mfcc_names = (
        [f'mfcc_mean_{i}' for i in range(n_mfcc)] +
        [f'mfcc_std_{i}'  for i in range(n_mfcc)] +
        [f'delta_mean_{i}' for i in range(n_mfcc)] +
        [f'delta_std_{i}'  for i in range(n_mfcc)]
    )

    print(f"\n  Extracting MFCC features from {len(df_bal):,} audio files …")
    mfcc_matrix  = np.zeros((len(df_bal), mfcc_dim), dtype=np.float32)
    audio_found  = 0

    for i, row in tqdm(df_bal.iterrows(), total=len(df_bal),
                        desc="COUGHVID MFCC"):
        uuid       = str(row.get('uuid', ''))
        audio_path = _find_coughvid_audio(uuid)
        if audio_path:
            mfcc_matrix[i] = _extract_mfcc_features(audio_path)
            audio_found    += 1

    print(f"  Audio files found: {audio_found:,} / {len(df_bal):,} "
          f"({100*audio_found/len(df_bal):.1f}%)")
    if audio_found < len(df_bal) * 0.5:
        print("  WARNING: < 50% audio found — XGBoost will rely mostly on metadata")

    # ── Combine metadata + MFCC into one DataFrame ────────────────
    label_map      = {cls: i for i, cls in enumerate(COUGHVID_CLASSES)}
    df_bal['label'] = df_bal['status'].map(label_map)

    mfcc_df = pd.DataFrame(mfcc_matrix, columns=mfcc_names)
    df_out  = pd.concat([df_bal[META_COLS + ['status', 'label', 'uuid']].reset_index(drop=True),
                         mfcc_df], axis=1)
    df_out.to_csv(COUGHVID_LABELS_CSV, index=False)
    print(f"  Saved → {COUGHVID_LABELS_CSV}")

    X = df_out[META_COLS + mfcc_names].values.astype(np.float32)
    y = df_out['label'].values.astype(np.int32)
    print(f"  Feature shape: {X.shape}  (8 metadata + {mfcc_dim} MFCC)")
    print("═" * 60)
    return X, y, label_map


# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID — Mel spectrogram extraction for EfficientNet binary classifier
# ══════════════════════════════════════════════════════════════════════════════

def extract_cough_spectrogram(audio_path: str) -> np.ndarray:
    """
    Mel spectrogram for cough audio recorded on a patient's phone.

    ── PHONE RECORDING CONTEXT ────────────────────────────────────────────────
    Audio is submitted by patients via phone (browser/app) — this creates:
      • Variable mic sensitivity across Android/iOS devices  → peak normalise
      • Silence before/after cough (patient hesitation)      → silence trim
      • Background noise (room, handling)                    → soft noise gate
      • Compressed formats: .webm / .ogg / .m4a              → ffmpeg decode

    NOTE: No pre-emphasis applied. Pre-emphasis is a speech processing technique
    that suppresses low-frequency energy. Cough classification relies on
    low-frequency characteristics (wet cough = low-freq resonance from mucus,
    dry cough = broadband). Pre-emphasis destroys this discriminative signal.
    ───────────────────────────────────────────────────────────────────────────

    Pipeline:
      1. Load & resample to COUGHVID_SR (22050 Hz)
      2. Trim silence (top_db=20)
      3. Peak-normalise
      4. Pad / truncate to COUGHVID_DURATION (4 s)
      5. Log-mel (n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=8000)
      6. (mel_db + 80) / 80 → [0, 1]

    Returns
    -------
    mel : np.ndarray shape (128, COUGHVID_SPEC_FRAMES) dtype float32
    """
    # 1. Load — librosa uses ffmpeg backend for .webm/.ogg/.m4a (phone formats)
    y, sr = librosa.load(audio_path, sr=COUGHVID_SR,
                         duration=COUGHVID_DURATION + 1.0, mono=True)

    # 2. Trim leading/trailing silence — patients often pause before coughing
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Peak-normalise — equalises sensitivity differences across phone models
    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = y / peak

    # 4. Pad / truncate to fixed length
    target = int(COUGHVID_SR * COUGHVID_DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode='constant')
    else:
        y = y[:target]

    # 5. Log-mel spectrogram
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=2048,
        hop_length=HOP_LENGTH, fmin=50, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel_db = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)

    # 6. Ensure fixed width
    if mel_db.shape[1] < COUGHVID_SPEC_FRAMES:
        mel_db = np.pad(mel_db,
                        ((0, 0), (0, COUGHVID_SPEC_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :COUGHVID_SPEC_FRAMES]

    return mel_db.astype(np.float32)


def _extract_pitch_shifted_spectrogram(audio_path: str, n_steps: int = -4) -> np.ndarray:
    """
    Extract mel spectrogram from pitch-shifted audio (augmentation).

    Pitch-shifting by n_steps semitones changes the spectral content without
    altering duration — the single biggest augmentation win for cough
    classification (+3% accuracy, CovidCoughNet 2023).

    Uses the same pipeline as extract_cough_spectrogram() but with pitch shift.
    """
    y, sr = librosa.load(audio_path, sr=COUGHVID_SR,
                         duration=COUGHVID_DURATION + 1.0, mono=True)

    # Pitch shift BEFORE other processing (modifies spectral content)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Same pipeline as extract_cough_spectrogram()
    y, _ = librosa.effects.trim(y, top_db=20)

    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = y / peak

    target = int(COUGHVID_SR * COUGHVID_DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode='constant')
    else:
        y = y[:target]

    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=2048,
        hop_length=HOP_LENGTH, fmin=50, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel_db = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)

    if mel_db.shape[1] < COUGHVID_SPEC_FRAMES:
        mel_db = np.pad(mel_db,
                        ((0, 0), (0, COUGHVID_SPEC_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :COUGHVID_SPEC_FRAMES]

    return mel_db.astype(np.float32)


def preprocess_coughvid_spectrograms() -> pd.DataFrame:
    """
    Extract mel spectrograms from COUGHVID audio for CNN binary classifier.

    Uses ALL quality-filtered samples (cough_detected >= 0.8).
    For minority class (Symptomatic), also creates pitch-shifted augmented copies
    to balance the dataset (~5600 Healthy vs ~1700 Symptomatic).

    Pitch-shifting: n_steps=-4 semitones (CovidCoughNet 2023, +3% accuracy).

    Returns
    -------
    df : pd.DataFrame with columns [spec_path, label, status]
    """
    print("\n" + "═" * 60)
    print("COUGHVID SPECTROGRAM PREPROCESSING  (all data + pitch-shift aug)")
    print("═" * 60)

    df = pd.read_csv(COUGHVID_METADATA, low_memory=False)
    df['status'] = df['status'].map(COUGHVID_LABEL_REMAP)
    df = df[df['status'].isin(COUGHVID_CLASSES)].copy()
    df = df[df['cough_detected'] >= 0.8].copy()

    # Use ALL samples (no cap) or cap if configured
    class_dfs = []
    for cls in COUGHVID_CLASSES:
        cls_df = df[df['status'] == cls]
        if COUGHVID_SAMPLES_PER_CLASS > 0:
            n = min(len(cls_df), COUGHVID_SAMPLES_PER_CLASS)
            cls_df = cls_df.sample(n=n, random_state=42)
        print(f"  '{cls}': {len(cls_df):,} samples")
        class_dfs.append(cls_df)
    df_all = pd.concat(class_dfs).reset_index(drop=True)

    label_map       = {cls: i for i, cls in enumerate(COUGHVID_CLASSES)}
    df_all['label'] = df_all['status'].map(label_map)

    records     = []
    err_count   = 0
    found_count = 0
    aug_count   = 0

    # Determine if we need pitch-shift augmentation for minority class
    class_counts = df_all['status'].value_counts()
    max_count    = class_counts.max()
    minority_cls = class_counts.idxmin()
    minority_gap = max_count - class_counts[minority_cls]
    print(f"  Majority: {class_counts.idxmax()} ({max_count:,})")
    print(f"  Minority: {minority_cls} ({class_counts[minority_cls]:,})")
    print(f"  Gap: {minority_gap:,} → will augment minority with pitch-shifting")

    for i, row in tqdm(df_all.iterrows(), total=len(df_all),
                       desc="COUGHVID mel-spec"):
        uuid       = str(row.get('uuid', ''))
        audio_path = _find_coughvid_audio(uuid)

        spec_name = f"coughvid_{uuid}.npy"
        spec_path = os.path.join(COUGHVID_SPEC_DIR, spec_name)

        if not os.path.exists(spec_path):
            if audio_path is None:
                err_count += 1
                continue
            try:
                mel = extract_cough_spectrogram(audio_path)
                np.save(spec_path, mel)
                found_count += 1
            except Exception:
                err_count += 1
                continue
        else:
            found_count += 1

        records.append({
            'spec_path': spec_path,
            'label':     int(row['label']),
            'status':    row['status'],
        })

    # ── Pitch-shift augmentation for minority class ────────────────
    # Create augmented copies to balance dataset
    minority_df = df_all[df_all['status'] == minority_cls].reset_index(drop=True)
    n_aug_needed = min(minority_gap, len(minority_df))  # augment up to balance

    if n_aug_needed > 0:
        aug_subset = minority_df.sample(n=n_aug_needed, random_state=42, replace=True)
        for i, row in tqdm(aug_subset.iterrows(), total=len(aug_subset),
                           desc="Pitch-shift augmentation"):
            uuid       = str(row.get('uuid', ''))
            audio_path = _find_coughvid_audio(uuid)

            spec_name = f"coughvid_{uuid}_ps.npy"  # _ps = pitch-shifted
            spec_path = os.path.join(COUGHVID_SPEC_DIR, spec_name)

            if not os.path.exists(spec_path):
                if audio_path is None:
                    continue
                try:
                    mel = _extract_pitch_shifted_spectrogram(audio_path, n_steps=-4)
                    np.save(spec_path, mel)
                    aug_count += 1
                except Exception:
                    continue
            else:
                aug_count += 1

            records.append({
                'spec_path': spec_path,
                'label':     int(row['label']),
                'status':    row['status'],
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(COUGHVID_SPEC_LABELS_CSV, index=False)

    print(f"\n  Original spectrograms: {found_count:,} | errors: {err_count:,}")
    print(f"  Pitch-shifted augmented: {aug_count:,}")
    print(f"  Total: {len(df_out):,}")
    print(f"  Label distribution: {df_out['status'].value_counts().to_dict()}")
    print(f"  Saved → {COUGHVID_SPEC_LABELS_CSV}")
    print("═" * 60)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# ICBHI — Breath cycle extraction
# ══════════════════════════════════════════════════════════════════════════════

def _parse_icbhi_sound(crackle: int, wheeze: int) -> int:
    """Convert crackle/wheeze bits to sound class index."""
    if crackle == 1 and wheeze == 1:
        return _SOUND_INT['Both']
    if crackle == 1:
        return _SOUND_INT['Crackle']
    if wheeze == 1:
        return _SOUND_INT['Wheeze']
    return _SOUND_INT['Normal']


def preprocess_icbhi() -> pd.DataFrame:
    """
    Extract breath cycles from ICBHI recordings.

    For each .txt annotation file:
      - Read breath cycles (start, end, crackle, wheeze)
      - Load the corresponding .wav, extract each cycle segment
      - Assign disease_label from patient_diagnosis.csv
        (disease_label = -1 if disease not in LUNG_DISEASE_CLASSES)
      - Assign sound_label from crackle/wheeze bits
      - Save mel spectrogram as .npy

    Returns
    -------
    df : pd.DataFrame with columns [spec_path, disease_label, sound_label, source]
    """
    print("\n" + "═" * 60)
    print("ICBHI PREPROCESSING — Breath cycle extraction")
    print("═" * 60)

    # Load patient diagnosis
    diag_df = pd.read_csv(ICBHI_DIAGNOSIS, header=None, names=['patient_id', 'disease'])
    diag_map = dict(zip(diag_df['patient_id'].astype(str), diag_df['disease']))
    print(f"  Loaded {len(diag_map)} patient diagnoses")
    print(f"  Disease distribution: {pd.Series(diag_map.values()).value_counts().to_dict()}")

    records    = []
    skip_count = 0
    err_count  = 0

    txt_files = [f for f in os.listdir(ICBHI_AUDIO_DIR) if f.endswith('.txt')]
    print(f"  Found {len(txt_files)} annotation files")

    for txt_file in tqdm(txt_files, desc="Extracting ICBHI cycles"):
        base      = txt_file.replace('.txt', '')
        wav_path  = os.path.join(ICBHI_AUDIO_DIR, base + '.wav')
        txt_path  = os.path.join(ICBHI_AUDIO_DIR, txt_file)

        if not os.path.exists(wav_path):
            skip_count += 1
            continue

        # Patient ID = first token before first underscore
        patient_id = base.split('_')[0]
        disease_raw = diag_map.get(patient_id, None)

        # Map to our disease classes (-1 = masked, sound head still trains)
        if disease_raw and disease_raw in ICBHI_DISEASE_MAP:
            disease_label = _DISEASE_INT[ICBHI_DISEASE_MAP[disease_raw]]
        else:
            disease_label = -1  # URTI/LRTI/Bronchiectasis → sound only

        # Parse annotation file
        try:
            cycles = []
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    start, end   = float(parts[0]), float(parts[1])
                    crackle, wheeze = int(parts[2]), int(parts[3])
                    cycles.append((start, end, crackle, wheeze))
        except Exception:
            err_count += 1
            continue

        if not cycles:
            continue

        # Extract spectrogram for each breath cycle
        for i, (start, end, crackle, wheeze) in enumerate(cycles):
            spec_name = f"icbhi_{base}_c{i:03d}.npy"
            spec_path = os.path.join(SPEC_DIR, spec_name)

            if not os.path.exists(spec_path):
                try:
                    seg_dur = max(end - start, 0.1)
                    mel = extract_melspectrogram(wav_path, offset=start, duration=seg_dur)
                    np.save(spec_path, mel)
                except Exception:
                    err_count += 1
                    continue

            sound_label = _parse_icbhi_sound(crackle, wheeze)
            records.append({
                'spec_path':     spec_path,
                'disease_label': disease_label,
                'sound_label':   sound_label,
                'source':        'icbhi',
            })

    df = pd.DataFrame(records)
    print(f"  Extracted {len(df):,} breath cycles | skipped={skip_count} errors={err_count}")
    print(f"  Disease distribution: {pd.Series(df['disease_label']).value_counts().to_dict()}")
    print(f"  Sound distribution  : {pd.Series(df['sound_label']).value_counts().to_dict()}")
    print("═" * 60)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# KAUH — Filename-based label extraction
# ══════════════════════════════════════════════════════════════════════════════

def _parse_kauh_filename(fname: str):
    """
    Parse disease and sound labels from KAUH filename.

    Format: BP108_COPD,E W,P R L ,63,M.wav
    Returns (disease_label_int, sound_label_int) or (None, None) to skip.
    """
    name = os.path.splitext(fname)[0]  # strip .wav

    # Split on first underscore to separate ID from rest
    parts = name.split('_', 1)
    if len(parts) < 2:
        return None, None

    rest   = parts[1]              # e.g. "COPD,E W,P R L ,63,M"
    tokens = rest.split(',')       # ['COPD', 'E W', 'P R L ', '63', 'M']

    if len(tokens) < 2:
        return None, None

    disease_raw = tokens[0].strip()
    sound_raw   = tokens[1].strip()

    # Disease mapping
    if disease_raw in KAUH_DISEASE_MAP:
        disease_label = _DISEASE_INT[KAUH_DISEASE_MAP[disease_raw]]
    else:
        disease_label = -1  # unknown disease → sound head only

    # Sound mapping — handle combined codes with '+'
    if '+' in sound_raw or (
        any(w in sound_raw for w in ['W', 'w']) and
        any(c in sound_raw for c in ['C', 'Crep'])
    ):
        sound_label = _SOUND_INT['Both']
    elif sound_raw in KAUH_SOUND_MAP:
        sound_label = _SOUND_INT[KAUH_SOUND_MAP[sound_raw]]
    elif any(w in sound_raw for w in ['W', 'Wheez', 'wheez']):
        sound_label = _SOUND_INT['Wheeze']
    elif any(c in sound_raw for c in ['C', 'Crep', 'crep']):
        sound_label = _SOUND_INT['Crackle']
    elif sound_raw == 'N' or sound_raw == '':
        sound_label = _SOUND_INT['Normal']
    else:
        return disease_label, None  # unknown sound → skip sound label

    return disease_label, sound_label


def preprocess_kauh() -> pd.DataFrame:
    """
    Load KAUH recordings, parse labels from filenames, extract mel spectrograms.

    Returns
    -------
    df : pd.DataFrame with columns [spec_path, disease_label, sound_label, source]
    """
    print("\n" + "═" * 60)
    print("KAUH PREPROCESSING — Filename label parsing")
    print("═" * 60)

    wav_files = [f for f in os.listdir(KAUH_AUDIO_DIR) if f.endswith('.wav')]
    print(f"  Found {len(wav_files)} .wav files")

    records   = []
    skip_count = 0
    err_count  = 0

    for fname in tqdm(wav_files, desc="Extracting KAUH spectrograms"):
        disease_label, sound_label = _parse_kauh_filename(fname)

        if sound_label is None:
            skip_count += 1
            continue

        spec_name = f"kauh_{os.path.splitext(fname)[0].replace(' ', '_')}.npy"
        spec_path = os.path.join(SPEC_DIR, spec_name)

        if not os.path.exists(spec_path):
            wav_path = os.path.join(KAUH_AUDIO_DIR, fname)
            try:
                mel = extract_melspectrogram(wav_path)
                np.save(spec_path, mel)
            except Exception as e:
                err_count += 1
                continue

        records.append({
            'spec_path':     spec_path,
            'disease_label': disease_label,
            'sound_label':   sound_label,
            'source':        'kauh',
        })

    df = pd.DataFrame(records)
    print(f"  Processed {len(df):,} files | skipped={skip_count} errors={err_count}")
    print(f"  Disease distribution: {pd.Series(df['disease_label']).value_counts().to_dict()}")
    print(f"  Sound distribution  : {pd.Series(df['sound_label']).value_counts().to_dict()}")
    print("═" * 60)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HF Lung V1 — Load from manifest
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_hf_lung() -> pd.DataFrame:
    """
    Load HF Lung V1 files from the pre-built manifest and extract mel spectrograms.
    disease_label = -1 (sound head only — no disease annotation in HF Lung V1).

    Returns
    -------
    df : pd.DataFrame with columns [spec_path, disease_label, sound_label, source]
    """
    print("\n" + "═" * 60)
    print("HF LUNG V1 PREPROCESSING — Load from manifest")
    print("═" * 60)

    manifest = pd.read_csv(HF_LUNG_MANIFEST)
    print(f"  Manifest rows: {len(manifest):,}")
    print(f"  Sound distribution: {manifest['label'].value_counts().to_dict()}")

    records   = []
    err_count = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="Extracting HF Lung spectrograms"):
        raw_label = row['label']
        if raw_label not in _SOUND_INT:
            continue

        sound_label = _SOUND_INT[raw_label]

        # audio_path in manifest may use Windows backslashes
        audio_path = str(row['audio_path']).replace('\\', '/')

        # Build spectrogram filename from audio basename
        base      = os.path.splitext(os.path.basename(audio_path))[0]
        spec_name = f"hflung_{base}.npy"
        spec_path = os.path.join(SPEC_DIR, spec_name)

        if not os.path.exists(spec_path):
            if not os.path.exists(audio_path):
                err_count += 1
                continue
            try:
                mel = extract_melspectrogram(audio_path)
                np.save(spec_path, mel)
            except Exception:
                err_count += 1
                continue

        records.append({
            'spec_path':     spec_path,
            'disease_label': -1,          # masked — no disease label
            'sound_label':   sound_label,
            'source':        'hflung',
        })

    df = pd.DataFrame(records)
    print(f"  Processed {len(df):,} files | errors={err_count}")
    print(f"  Sound distribution: {pd.Series(df['sound_label']).value_counts().to_dict()}")
    print("═" * 60)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Build combined multi-task dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_multitask_dataset() -> pd.DataFrame:
    """
    Run all three audio preprocessors and combine into one CSV.

    Saves to MULTITASK_LABELS_CSV.
    Returns combined DataFrame.
    """
    print("\n" + "═" * 60)
    print("BUILDING COMBINED MULTI-TASK DATASET")
    print("═" * 60)

    df_icbhi = preprocess_icbhi()
    df_kauh  = preprocess_kauh()
    df_hf    = preprocess_hf_lung()

    df = pd.concat([df_icbhi, df_kauh, df_hf], ignore_index=True)

    # Drop rows where spectrogram file doesn't exist
    df = df[df['spec_path'].apply(os.path.exists)].reset_index(drop=True)

    # Summary
    print("\n" + "─" * 50)
    print("COMBINED DATASET SUMMARY")
    print("─" * 50)
    print(f"Total samples      : {len(df):,}")
    print(f"  ICBHI            : {(df['source']=='icbhi').sum():,}")
    print(f"  KAUH             : {(df['source']=='kauh').sum():,}")
    print(f"  HF Lung          : {(df['source']=='hflung').sum():,}")
    print("\nDisease head (disease_label >= 0):")
    df_dis = df[df['disease_label'] >= 0]
    print(f"  Labelled samples : {len(df_dis):,}")
    for i, cls in enumerate(LUNG_DISEASE_CLASSES):
        n = (df_dis['disease_label'] == i).sum()
        print(f"    [{i}] {cls:<16}: {n:,}")
    print("\nSound head (all samples):")
    for i, cls in enumerate(LUNG_SOUND_CLASSES):
        n = (df['sound_label'] == i).sum()
        print(f"    [{i}] {cls:<10}: {n:,}")
    print("─" * 50)

    df.to_csv(MULTITASK_LABELS_CSV, index=False)
    print(f"\nSaved → {MULTITASK_LABELS_CSV}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # COUGHVID — already preprocessed, just verify
    if os.path.exists(COUGHVID_LABELS_CSV):
        print(f"[preprocessing] COUGHVID labels already exist at {COUGHVID_LABELS_CSV}")
    else:
        X, y, label_map = preprocess_coughvid()
        print(f"COUGHVID ready: X={X.shape}, classes={label_map}")

    # Build multi-task dataset (ICBHI + KAUH + HF Lung)
    df = build_multitask_dataset()
    print(f"\nMulti-task dataset ready: {len(df):,} samples → {MULTITASK_LABELS_CSV}")