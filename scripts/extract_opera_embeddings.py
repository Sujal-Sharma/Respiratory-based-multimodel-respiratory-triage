"""
scripts/extract_opera_embeddings.py — One-time OPERA-CT embedding extraction.

Runs OPERA-CT on every audio file in the three label CSVs and saves
embeddings as .npy files. After this, OPERA never runs again — training
loads only from .npy files.

OPERA-CT checkpoint auto-downloads from HuggingFace on first run.

Prerequisites:
  - data/copd_binary_labels.csv      (from scripts/build_label_csvs.py)
  - data/pneumonia_binary_labels.csv
  - data/sound_labels.csv
  - ./OPERA/ cloned from github.com/evelyn0414/OPERA

Output:
  - data/opera_embeddings/<source>/<filename>.npy
  - data/copd_binary_labels_with_embeddings.csv
  - data/pneumonia_binary_labels_with_embeddings.csv
  - data/sound_labels_with_embeddings.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.opera_encoder import OPERAEncoder

OUTPUT_DIR   = './data/opera_embeddings'
CHUNK_SIZE   = 64    # files sent to encoder.encode_batch() at once
GPU_BATCH    = 16    # GPU forward pass batch size (safe for GTX 1650 4GB)
N_WORKERS    = 4     # CPU threads for parallel audio preprocessing
PRETRAIN     = 'operaCT'
INPUT_SEC    = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_and_save(csv_path: str, encoder: OPERAEncoder) -> pd.DataFrame:
    """
    Extract OPERA embeddings for all files in csv_path.
    Skips files whose .npy already exists (safe to re-run after interruption).

    Returns updated DataFrame with 'embedding_path' column filled.
    """
    df = pd.read_csv(csv_path)
    if 'embedding_path' not in df.columns:
        df['embedding_path'] = None

    # Determine which rows still need extraction
    todo_mask = df['embedding_path'].isna() | (df['embedding_path'] == '')
    # Also check if the .npy file actually exists for non-null paths
    for idx, row in df[~todo_mask].iterrows():
        if not os.path.exists(str(row['embedding_path'])):
            todo_mask.at[idx] = True

    todo_df = df[todo_mask].copy()
    print(f"  {len(todo_df)} files to extract ({len(df) - len(todo_df)} already done)")

    if len(todo_df) == 0:
        return df

    failed = []

    # Process in chunks
    for batch_start in tqdm(range(0, len(todo_df), CHUNK_SIZE),
                            desc=f"  Extracting {os.path.basename(csv_path)}"):
        batch = todo_df.iloc[batch_start: batch_start + CHUNK_SIZE]

        # Build output paths first
        out_paths = []
        valid_rows = []
        for _, row in batch.iterrows():
            file_path = row['file_path']
            if not os.path.exists(file_path):
                failed.append(file_path)
                continue

            fname    = os.path.basename(file_path)
            fname    = (fname.replace('.wav', '.npy')
                             .replace('.webm', '.npy')
                             .replace('.mp3', '.npy'))
            source   = str(row.get('source', 'unknown'))
            out_dir  = os.path.join(OUTPUT_DIR, source)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)

            out_paths.append(out_path)
            valid_rows.append((row.name, file_path, out_path))

        if not valid_rows:
            continue

        # Skip rows where .npy already exists
        to_encode    = [(idx, fp, op) for idx, fp, op in valid_rows
                        if not os.path.exists(op)]
        already_done = [(idx, fp, op) for idx, fp, op in valid_rows
                        if os.path.exists(op)]

        # Update already-done rows
        for idx, fp, op in already_done:
            df.at[idx, 'embedding_path'] = op

        if not to_encode:
            continue

        file_paths_batch = [fp for _, fp, _ in to_encode]
        out_paths_batch  = [op for _, _, op in to_encode]
        indices_batch    = [idx for idx, _, _ in to_encode]

        try:
            embeddings = encoder.encode_batch(file_paths_batch)
            for i, (idx, out_path) in enumerate(zip(indices_batch, out_paths_batch)):
                np.save(out_path, embeddings[i].astype(np.float32))
                df.at[idx, 'embedding_path'] = out_path
        except Exception as e:
            print(f"\n  Batch failed: {e}")
            # Fall back to one-by-one
            for idx, file_path, out_path in to_encode:
                try:
                    emb = encoder.encode_batch([file_path])
                    np.save(out_path, emb[0].astype(np.float32))
                    df.at[idx, 'embedding_path'] = out_path
                except Exception as e2:
                    print(f"\n  Failed: {file_path} — {e2}")
                    failed.append(file_path)

    n_done = df['embedding_path'].notna().sum()
    print(f"  Done: {n_done} extracted | Failed: {len(failed)}")
    if failed:
        print(f"  Failed files: {failed[:5]}{'...' if len(failed) > 5 else ''}")

    return df


def main():
    encoder = OPERAEncoder(pretrain=PRETRAIN, input_sec=INPUT_SEC,
                           batch_size=GPU_BATCH, n_workers=N_WORKERS)

    print("\n[1/3] COPD binary dataset")
    df_copd = extract_and_save('data/copd_binary_labels.csv', encoder)
    df_copd.to_csv('data/copd_binary_labels_with_embeddings.csv', index=False)
    print("  Saved: data/copd_binary_labels_with_embeddings.csv")

    print("\n[2/3] Pneumonia binary dataset")
    df_pneu = extract_and_save('data/pneumonia_binary_labels.csv', encoder)
    df_pneu.to_csv('data/pneumonia_binary_labels_with_embeddings.csv', index=False)
    print("  Saved: data/pneumonia_binary_labels_with_embeddings.csv")

    print("\n[3/3] Sound labels dataset")
    df_snd = extract_and_save('data/sound_labels.csv', encoder)
    df_snd.to_csv('data/sound_labels_with_embeddings.csv', index=False)
    print("  Saved: data/sound_labels_with_embeddings.csv")

    print("\nAll embeddings extracted. OPERA will not run again during training.")


if __name__ == '__main__':
    main()
