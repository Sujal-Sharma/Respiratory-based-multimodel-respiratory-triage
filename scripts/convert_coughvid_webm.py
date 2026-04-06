"""
scripts/convert_coughvid_webm.py — Convert COUGHVID .webm files to .wav

OPERA's get_entire_signal_librosa() strips the file extension and appends
.wav, so .webm files always fail. Convert them in-place (keeping originals).

Run this ONCE before extract_opera_embeddings.py.
Also updates data/copd_binary_labels.csv and data/pneumonia_binary_labels.csv
to point to the new .wav paths.

Requirements: ffmpeg must be on PATH.
"""

import os
import sys
import subprocess
import pandas as pd
from tqdm import tqdm

COUGHVID_DIR = './DATASET/COUGHVID_DATASET/'
LABEL_CSVS = [
    'data/copd_binary_labels.csv',
    'data/pneumonia_binary_labels.csv',
]


def convert_webm_to_wav(webm_path: str) -> str:
    """Convert a single .webm file to .wav using ffmpeg. Returns .wav path."""
    wav_path = webm_path.replace('.webm', '.wav')
    if os.path.exists(wav_path):
        return wav_path  # already converted

    result = subprocess.run(
        ['ffmpeg', '-y', '-i', webm_path,
         '-ar', '16000',   # 16 kHz — what OPERA expects
         '-ac', '1',       # mono
         '-f', 'wav',
         wav_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {webm_path}:\n{result.stderr[-300:]}")
    return wav_path


def main():
    # Find all .webm files
    webm_files = [
        os.path.join(COUGHVID_DIR, f)
        for f in os.listdir(COUGHVID_DIR)
        if f.endswith('.webm')
    ]
    print(f"Found {len(webm_files)} .webm files to convert")

    failed = []
    wav_map = {}  # webm_path -> wav_path

    for webm_path in tqdm(webm_files, desc="Converting .webm → .wav"):
        try:
            wav_path = convert_webm_to_wav(webm_path)
            wav_map[os.path.abspath(webm_path)] = os.path.abspath(wav_path)
        except Exception as e:
            print(f"\nFailed: {webm_path}\n  {e}")
            failed.append(webm_path)

    print(f"\nConverted: {len(wav_map)} | Failed: {len(failed)}")

    # Update label CSVs to use .wav paths
    for csv_path in LABEL_CSVS:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_updated = 0
        for idx, row in df.iterrows():
            fp = str(row['file_path'])
            if fp in wav_map:
                df.at[idx, 'file_path'] = wav_map[fp]
                n_updated += 1
        df.to_csv(csv_path, index=False)
        print(f"Updated {n_updated} paths in {csv_path}")

    print("\nDone. Now run: python scripts/extract_opera_embeddings.py")


if __name__ == '__main__':
    main()
