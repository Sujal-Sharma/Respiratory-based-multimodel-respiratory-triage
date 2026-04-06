"""
scripts/build_label_csvs.py — Build the three label CSVs from all datasets.

Run after fix_kauh_parser.py. Creates:
  data/copd_binary_labels.csv       — COPD vs Normal (binary)
  data/pneumonia_binary_labels.csv  — Pneumonia vs Normal (binary)
  data/sound_labels.csv             — Normal/Crackle/Wheeze/Both (4-class)

Sources:
  COPD labels    : ICBHI + KAUH
  Pneumonia labels: ICBHI + KAUH
  Normal (negatives): ICBHI Normal + KAUH Normal + COUGHVID Healthy
  Sound labels   : ICBHI + KAUH + HF Lung V1
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    ICBHI_AUDIO_DIR, ICBHI_DIAGNOSIS,
    KAUH_AUDIO_DIR,
    COUGHVID_AUDIO_DIR, COUGHVID_METADATA,
    HF_LUNG_MANIFEST,
)

os.makedirs('data', exist_ok=True)

SOUND_INT = {'Normal': 0, 'Crackle': 1, 'Wheeze': 2, 'Both': 3}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ICBHI 2017
# ══════════════════════════════════════════════════════════════════════════════

def load_icbhi():
    """
    Parse ICBHI dataset: patient_diagnosis.csv + per-cycle .txt annotations.

    Returns DataFrame with columns:
        file_path, disease, sound_type, patient_id, source
    """
    records = []

    # Load patient-level diagnosis (CSV: patient_id,diagnosis — no header)
    diag_df = pd.read_csv(ICBHI_DIAGNOSIS, header=None,
                          names=['patient_id', 'diagnosis'])
    diag_map = dict(zip(diag_df['patient_id'].astype(str),
                        diag_df['diagnosis']))

    for fname in os.listdir(ICBHI_AUDIO_DIR):
        if not fname.endswith('.wav'):
            continue

        patient_id = fname.split('_')[0]
        disease_raw = diag_map.get(patient_id, 'Unknown')

        # Map to our 3-class disease set
        if disease_raw == 'COPD':
            disease = 'COPD'
        elif disease_raw == 'Pneumonia':
            disease = 'Pneumonia'
        elif disease_raw == 'Healthy':
            disease = 'Normal'
        else:
            continue  # skip URTI, Bronchiectasis, etc.

        # Try to read sound annotation from companion .txt
        txt_path = os.path.join(ICBHI_AUDIO_DIR,
                                fname.replace('.wav', '.txt'))
        sound_type = 'Normal'
        if os.path.exists(txt_path):
            try:
                ann = pd.read_csv(txt_path, sep='\t', header=None)
                # Columns: start, end, crackle_flag, wheeze_flag
                crackles = ann.iloc[:, 2].sum() > 0
                wheezes  = ann.iloc[:, 3].sum() > 0
                if crackles and wheezes:
                    sound_type = 'Both'
                elif crackles:
                    sound_type = 'Crackle'
                elif wheezes:
                    sound_type = 'Wheeze'
            except Exception:
                pass

        records.append({
            'file_path':  os.path.abspath(os.path.join(ICBHI_AUDIO_DIR, fname)),
            'disease':    disease,
            'sound_type': sound_type,
            'patient_id': patient_id,
            'source':     'icbhi',
        })

    df = pd.DataFrame(records)
    print(f"[ICBHI] Loaded {len(df)} files")
    if len(df) == 0:
        raise RuntimeError(
            f"ICBHI loaded 0 files. Check ICBHI_AUDIO_DIR in config.py.\n"
            f"  AUDIO_DIR = {ICBHI_AUDIO_DIR}\n"
            f"  DIAGNOSIS  = {ICBHI_DIAGNOSIS}"
        )
    print(f"  Disease counts:\n{df['disease'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. KAUH (requires data/kauh_parsed.csv from fix_kauh_parser.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_kauh():
    """Load KAUH from pre-parsed CSV (run fix_kauh_parser.py first)."""
    kauh_csv = 'data/kauh_parsed.csv'
    if not os.path.exists(kauh_csv):
        raise FileNotFoundError(
            "data/kauh_parsed.csv not found. "
            "Run: python scripts/fix_kauh_parser.py"
        )
    df = pd.read_csv(kauh_csv)

    # Keep only COPD, Pneumonia, Normal
    df = df[df['disease'].isin(['COPD', 'Pneumonia', 'Normal'])].copy()
    df = df.rename(columns={'sound_type': 'sound_type'})
    df['source'] = 'kauh'

    print(f"[KAUH] Loaded {len(df)} files (COPD/Pneumonia/Normal only)")
    print(f"  Disease counts:\n{df['disease'].value_counts().to_string()}")
    return df[['file_path', 'disease', 'sound_type', 'patient_id', 'source']]


# ══════════════════════════════════════════════════════════════════════════════
# 3. COUGHVID (Healthy samples only — used as negatives)
# ══════════════════════════════════════════════════════════════════════════════

def load_coughvid_healthy(max_samples: int = 1000):
    """Load healthy COUGHVID samples as Normal class negatives."""
    if not os.path.exists(COUGHVID_METADATA):
        print("[COUGHVID] Metadata not found — skipping")
        return pd.DataFrame()

    meta = pd.read_csv(COUGHVID_METADATA)
    # Keep only confirmed healthy
    healthy = meta[meta['status'].str.lower() == 'healthy'].copy()
    if len(healthy) > max_samples:
        healthy = healthy.sample(max_samples, random_state=42)

    records = []
    for _, row in healthy.iterrows():
        # Try .webm first, then .wav
        for ext in ('.webm', '.wav', '.mp3'):
            fpath = os.path.join(COUGHVID_AUDIO_DIR, row['uuid'] + ext)
            if os.path.exists(fpath):
                records.append({
                    'file_path':  os.path.abspath(fpath),
                    'disease':    'Normal',
                    'sound_type': 'Normal',
                    'patient_id': str(row['uuid']),
                    'source':     'coughvid',
                })
                break

    df = pd.DataFrame(records)
    print(f"[COUGHVID] Loaded {len(df)} healthy samples")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. HF Lung V1 (sound labels only — no disease label)
# ══════════════════════════════════════════════════════════════════════════════

def load_hf_lung():
    """Load HF Lung V1 sound labels (for sound_labels.csv only).

    Manifest columns: audio_path, label, split
    label values: Crackle, Wheeze, Normal, Artifact (skip Artifact)
    """
    if not os.path.exists(HF_LUNG_MANIFEST):
        print("[HF Lung] Manifest not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(HF_LUNG_MANIFEST)

    # Rename to standard column names
    df = df.rename(columns={'audio_path': 'file_path', 'label': 'sound_type'})

    # Skip Artifact rows — not a real respiratory sound class
    df = df[df['sound_type'].isin(['Normal', 'Crackle', 'Wheeze'])].copy()

    # Make file paths absolute (they use Windows backslash relative paths)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df['file_path'] = df['file_path'].apply(
        lambda p: os.path.abspath(os.path.join(project_root, p.replace('\\', os.sep)))
    )

    df['disease']    = 'Unknown'   # no disease label in HF Lung
    df['source']     = 'hf_lung'
    df['patient_id'] = 'hf_' + df.index.astype(str)

    print(f"[HF Lung] Loaded {len(df)} files")
    print(f"  Sound counts:\n{df['sound_type'].value_counts().to_string()}")
    return df[['file_path', 'disease', 'sound_type', 'patient_id', 'source']]


# ══════════════════════════════════════════════════════════════════════════════
# Build and save the three CSVs
# ══════════════════════════════════════════════════════════════════════════════

def main():
    icbhi   = load_icbhi()
    kauh    = load_kauh()
    coughvid = load_coughvid_healthy()
    hf_lung = load_hf_lung()

    all_labeled = pd.concat([icbhi, kauh], ignore_index=True)

    # ── copd_binary_labels.csv ────────────────────────────────────────────────
    copd_pos = all_labeled[all_labeled['disease'] == 'COPD'].copy()
    copd_neg = pd.concat([
        all_labeled[all_labeled['disease'] == 'Normal'],
        coughvid,
    ], ignore_index=True)

    copd_pos['label'] = 1
    copd_neg['label'] = 0

    df_copd = pd.concat([copd_pos, copd_neg], ignore_index=True)
    df_copd = df_copd[['file_path', 'label', 'source']].drop_duplicates()
    df_copd.to_csv('data/copd_binary_labels.csv', index=False)
    print(f"\n[OUT] data/copd_binary_labels.csv — {len(df_copd)} rows")
    print(f"  Labels: {df_copd['label'].value_counts().to_dict()}")

    # ── pneumonia_binary_labels.csv ───────────────────────────────────────────
    pneu_pos = all_labeled[all_labeled['disease'] == 'Pneumonia'].copy()
    pneu_neg = pd.concat([
        all_labeled[all_labeled['disease'] == 'Normal'],
        coughvid,
    ], ignore_index=True)

    pneu_pos['label'] = 1
    pneu_neg['label'] = 0

    df_pneu = pd.concat([pneu_pos, pneu_neg], ignore_index=True)
    df_pneu = df_pneu[['file_path', 'label', 'source']].drop_duplicates()
    df_pneu.to_csv('data/pneumonia_binary_labels.csv', index=False)
    print(f"\n[OUT] data/pneumonia_binary_labels.csv — {len(df_pneu)} rows")
    print(f"  Labels: {df_pneu['label'].value_counts().to_dict()}")

    # ── sound_labels.csv ──────────────────────────────────────────────────────
    sound_sources = pd.concat([all_labeled, hf_lung], ignore_index=True)
    sound_sources = sound_sources[sound_sources['sound_type'].isin(
        SOUND_INT.keys()
    )].copy()
    sound_sources['sound_label'] = sound_sources['sound_type'].map(SOUND_INT)

    df_sound = sound_sources[['file_path', 'sound_label', 'source']].drop_duplicates()
    df_sound.to_csv('data/sound_labels.csv', index=False)
    print(f"\n[OUT] data/sound_labels.csv — {len(df_sound)} rows")
    print(f"  Sound labels: {df_sound['sound_label'].value_counts().to_dict()}")

    print("\nAll label CSVs created. Next: run extract_opera_embeddings.py")


if __name__ == '__main__':
    main()
