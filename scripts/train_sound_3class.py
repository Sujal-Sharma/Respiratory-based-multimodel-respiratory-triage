"""
scripts/train_sound_3class.py — Train 3-class lung sound classifier.

Classes: 0=Normal, 1=Crackle, 2=Wheeze
"Both" class (original label 3) is merged into Crackle (label 1) because
crackle is the clinically dominant sound when both are present.

Input:  768-dim OPERA-CT embeddings
Output: saved_models/sound_opera_mlp_3class.pt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import SoundMLPClassifier, FocalLoss

# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH        = 'data/sound_labels_with_embeddings.csv'
MODEL_SAVE_PATH = 'saved_models/sound_opera_mlp_3class.pt'
RESULTS_PATH    = 'outputs/results_sound_3class.json'

INPUT_DIM    = 768
HIDDEN_DIMS  = [512, 256, 64]
DROPOUT      = 0.3
BATCH_SIZE   = 128
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42
SOUND_LABELS = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze'}
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[train_sound_3class] Device: {device}")


class SoundDataset3Class(Dataset):
    """Loads embeddings and maps label 3 (Both) → 1 (Crackle)."""

    def __init__(self, csv_path: str, augment: bool = False):
        df = pd.read_csv(csv_path).dropna(subset=['embedding_path'])
        # Merge Both → Crackle
        df['sound_label'] = df['sound_label'].replace({3: 1})
        self.paths  = df['embedding_path'].tolist()
        self.labels = df['sound_label'].tolist()
        self.augment = augment
        print(f"[SoundDataset3Class] {csv_path}: {len(df)} samples")
        print(f"  Label dist: {df['sound_label'].value_counts().sort_index().to_dict()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = np.load(self.paths[idx]).astype(np.float32)
        if self.augment:
            emb = emb + np.random.normal(0, 0.01, emb.shape).astype(np.float32)
            scale = np.random.uniform(0.95, 1.05)
            emb = emb * scale
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        return torch.tensor(emb), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Load and split ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna(subset=['embedding_path'])
df['sound_label'] = df['sound_label'].replace({3: 1})  # Both → Crackle

print(f"\n[train_sound_3class] After merge — Label distribution:")
print(df['sound_label'].value_counts().sort_index())

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['sound_label'], random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['sound_label'], random_state=RANDOM_STATE
)

train_df.to_csv('data/sound_train_3class.csv', index=False)
val_df.to_csv(  'data/sound_val_3class.csv',   index=False)
test_df.to_csv( 'data/sound_test_3class.csv',  index=False)

print(f"\n[train_sound_3class] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Datasets & loaders ────────────────────────────────────────────────────────
train_dataset = SoundDataset3Class('data/sound_train_3class.csv', augment=True)
val_dataset   = SoundDataset3Class('data/sound_val_3class.csv',   augment=False)
test_dataset  = SoundDataset3Class('data/sound_test_3class.csv',  augment=False)

labels_arr   = train_df['sound_label'].values
class_counts = np.bincount(labels_arr)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels_arr]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model — 3-class output ─────────────────────────────────────────────────
class SoundMLP3Class(torch.nn.Module):
    """Same architecture as SoundMLPClassifier but with 3 output classes."""
    def __init__(self, input_dim=768, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                torch.nn.Linear(prev, h),
                torch.nn.BatchNorm1d(h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model     = SoundMLP3Class(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_f1      = 0.0
patience_counter = 0
best_model_state = None

print()
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for emb, lbl in train_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = criterion(model(emb), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb, lbl in val_loader:
            preds = model(emb.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1      = val_f1
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    if patience_counter >= PATIENCE:
        print(f"[train_sound_3class] Early stopping at epoch {epoch}")
        break

# ── Test ─────────────────────────────────────────────────────────────────────
model.load_state_dict(best_model_state)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for emb, lbl in test_loader:
        preds = model(emb.to(device)).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbl.numpy())

print("\n[train_sound_3class] Test Results:")
print(classification_report(
    all_labels, all_preds,
    target_names=[SOUND_LABELS[i] for i in range(3)],
    zero_division=0
))

per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
test_results = {
    'task':        'sound_3class',
    'note':        'Both(3) merged into Crackle(1)',
    'accuracy':    float(accuracy_score(all_labels, all_preds)),
    'f1_macro':    float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
    'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
    'per_class_f1': {SOUND_LABELS[i]: float(v) for i, v in enumerate(per_class)},
}

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('saved_models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

torch.save({
    'model_state_dict': best_model_state,
    'hidden_dims':      HIDDEN_DIMS,
    'input_dim':        INPUT_DIM,
    'num_classes':      3,
    'sound_labels':     SOUND_LABELS,
    'test_results':     test_results,
    'note':             'Both merged into Crackle',
}, MODEL_SAVE_PATH)

with open(RESULTS_PATH, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"[train_sound_3class] Model saved to {MODEL_SAVE_PATH}")
print(f"[train_sound_3class] F1 Macro: {test_results['f1_macro']:.4f}")
