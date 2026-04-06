"""
scripts/fix_kauh_parser.py — Parse KAUH dataset filenames into a clean CSV.

KAUH filenames encode: patient_id_DISEASE,SOUND,REGION,AGE,GENDER.wav
Example: BP108_COPD,E W,P R L,63,M.wav

Run this FIRST before any preprocessing or embedding extraction.
Output: data/kauh_parsed.csv
"""

import os
import pandas as pd

KAUH_DIR = "./DATASET/KAUH_DATASET/Audio Files/"

def map_disease(raw: str) -> str:
    """Normalise disease name to canonical form (case-insensitive)."""
    r = raw.strip().lower()
    if r in ('copd', 'copd '):
        return 'COPD'
    if r in ('pneumonia', 'pneumonia '):
        return 'Pneumonia'
    if r in ('n', 'normal', 'healthy'):
        return 'Normal'
    if r in ('asthma',):
        return 'Asthma'
    if r in ('heart failure', 'heart_failure'):
        return 'Heart_Failure'
    # Comorbidities containing COPD — still useful as COPD samples
    if 'copd' in r and 'heart' not in r:
        return 'COPD'
    return 'OTHER'


SOUND_MAP = {
    'N': 'Normal',
    'E W': 'Wheeze', 'I E W': 'Wheeze', 'I W': 'Wheeze',
    'B W': 'Wheeze', 'W': 'Wheeze',
    'C': 'Crackle', 'Crep': 'Crackle', 'E Crep': 'Crackle',
}

records = []
for fname in os.listdir(KAUH_DIR):
    if not fname.endswith('.wav'):
        continue
    try:
        # Remove .wav, split on FIRST underscore only
        base = fname.replace('.wav', '')
        underscore_idx = base.index('_')
        patient_id = base[:underscore_idx]
        rest = base[underscore_idx + 1:]

        # Split rest on comma
        fields = [f.strip() for f in rest.split(',')]

        disease_raw = fields[0] if len(fields) > 0 else 'Unknown'
        sound_raw   = fields[1] if len(fields) > 1 else 'N'
        age         = int(fields[3]) if len(fields) > 3 and fields[3].strip().isdigit() else -1
        gender      = fields[4].strip() if len(fields) > 4 else 'Unknown'

        disease = map_disease(disease_raw)
        sound   = SOUND_MAP.get(sound_raw.strip(), 'Normal')

        records.append({
            'file_path':  os.path.abspath(os.path.join(KAUH_DIR, fname)),
            'patient_id': patient_id,
            'disease':    disease,
            'sound_type': sound,
            'age':        age,
            'gender':     gender,
            'source':     'kauh',
        })
    except Exception as e:
        print(f"Failed to parse {fname}: {e}")
        continue

df = pd.DataFrame(records)
print("KAUH per-disease counts:")
print(df['disease'].value_counts())
print("\nKAUH per-sound counts:")
print(df['sound_type'].value_counts())
print(f"\nTotal files parsed: {len(df)}")

# Sanity checks — KAUH is a small dataset (336 files total)
# COPD: expect ~24-30, Pneumonia: expect ~5-15
n_copd = df[df['disease'] == 'COPD'].shape[0]
n_pneu = df[df['disease'] == 'Pneumonia'].shape[0]
print(f"\nCOPD samples: {n_copd}")
print(f"Pneumonia samples: {n_pneu}")
assert n_copd > 10, f"COPD count {n_copd} too low — parser broken"
assert n_pneu > 3,  f"Pneumonia count {n_pneu} too low — check filenames"

os.makedirs('data', exist_ok=True)
df.to_csv('data/kauh_parsed.csv', index=False)
print("\nSaved to data/kauh_parsed.csv")
