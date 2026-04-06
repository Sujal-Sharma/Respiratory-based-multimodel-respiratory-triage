"""
models/embedding_dataset.py — PyTorch Dataset for pre-computed OPERA embeddings.

Training is extremely fast because no audio processing happens at runtime —
embeddings are loaded directly from pre-saved .npy files.

Label columns:
  - Binary disease tasks: 'label' (0 or 1)
  - Sound classification:  'sound_label' (0-3)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Loads pre-computed OPERA embeddings from .npy files.

    Parameters
    ----------
    csv_path  : path to CSV with columns [embedding_path, <label_col>]
    label_col : column name for the class label
    augment   : whether to apply lightweight embedding-space augmentation
    """

    def __init__(self, csv_path: str,
                 label_col: str = 'label',
                 augment: bool = False):
        self.label_col = label_col
        self.augment   = augment

        df = pd.read_csv(csv_path)

        # Drop rows without a valid embedding path
        df = df.dropna(subset=['embedding_path'])
        df = df[df['embedding_path'].apply(
            lambda p: isinstance(p, str) and p.strip().endswith('.npy')
        )]
        self.df = df.reset_index(drop=True)

        print(f"[EmbeddingDataset] {csv_path}: {len(self.df)} samples")
        print(f"  Label distribution:\n{self.df[label_col].value_counts().to_string()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        embedding = np.load(row['embedding_path']).astype(np.float32)
        label     = int(row[self.label_col])

        if self.augment:
            # Gaussian noise in embedding space (very small — maintains semantics)
            if np.random.random() < 0.3:
                embedding += np.random.randn(*embedding.shape).astype(np.float32) * 0.01
            # Random amplitude scaling
            if np.random.random() < 0.2:
                embedding *= np.random.uniform(0.95, 1.05)
            # Re-normalise to unit sphere after augmentation
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
