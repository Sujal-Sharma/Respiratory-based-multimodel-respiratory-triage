"""
scripts/extract_cough_embeddings.py — Extract OPERA-CT embeddings for COUGHVID.

Reads data/cough_labels.csv, extracts 768-dim OPERA embeddings,
saves each as .npy and writes data/cough_labels_with_embeddings.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.opera_encoder import OPERAEncoder

EMBED_DIR = 'data/opera_embeddings/cough'
os.makedirs(EMBED_DIR, exist_ok=True)

df = pd.read_csv('data/cough_labels.csv')
print(f"Total samples: {len(df)}")
print(df['label_str'].value_counts())

encoder = OPERAEncoder(pretrain='operaCT', batch_size=16, n_workers=4)

embedding_paths = []
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
    uid  = row['uuid']
    path = row['file_path']
    npy_path = os.path.join(EMBED_DIR, uid + '.npy')

    if os.path.exists(npy_path):
        embedding_paths.append(npy_path)
        continue

    emb = encoder.encode(str(path))
    if np.all(emb == 0):
        failed += 1
        embedding_paths.append(None)
        continue

    np.save(npy_path, emb)
    embedding_paths.append(npy_path)

df['embedding_path'] = embedding_paths
df.to_csv('data/cough_labels_with_embeddings.csv', index=False)

valid = df['embedding_path'].notna().sum()
print(f"\nDone. {valid}/{len(df)} embeddings extracted. {failed} failed.")
print("Saved: data/cough_labels_with_embeddings.csv")
