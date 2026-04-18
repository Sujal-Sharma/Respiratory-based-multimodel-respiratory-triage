"""
scripts/train_cough_agent.py — Train binary Cough classifier on COUGHVID.

Labels: Healthy (0) vs Symptomatic (1) [COVID-19 merged into Symptomatic]
Input:  OPERA-CT 768-dim embeddings
Model:  BinaryMLP 768 -> 256 -> 64 -> 2 with BatchNorm + Dropout

Output: saved_models/cough_opera_mlp.pt
        outputs/cough_confusion_matrix.png
        outputs/cough_roc_curve.png
        outputs/results_cough.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.threshold_optimizer import optimize_threshold, compute_threshold_metrics
os.makedirs('outputs', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

RANDOM_STATE = 42
EPOCHS       = 60
BATCH_SIZE   = 64
LR           = 1e-3
DROPOUT      = 0.3
THRESHOLD_OBJECTIVE = 'youden_j'
THRESHOLD_TRIALS    = 200
torch.manual_seed(RANDOM_STATE)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/cough_labels_with_embeddings.csv').dropna(subset=['embedding_path'])
print(f"Total samples: {len(df)}")
print(df['label_str'].value_counts())

def load_embedding(path):
    return np.load(path).astype(np.float32)

X = np.stack([load_embedding(p) for p in df['embedding_path']])
y = df['label'].values.astype(np.int64)

# Train / val / test split (70/15/15) — stratified
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=RANDOM_STATE)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Test positives (symptomatic): {y_test.sum()}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = EmbeddingDataset(X_train, y_train)
val_ds   = EmbeddingDataset(X_val,   y_val)
test_ds  = EmbeddingDataset(X_test,  y_test)

# Weighted sampler for class imbalance
counts  = np.bincount(y_train)
weights = 1.0 / counts[y_train]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class CoughMLP(nn.Module):
    def __init__(self, input_dim=768, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

model = CoughMLP(dropout=DROPOUT).to(device)

# Class-weighted loss
class_weights = torch.tensor(
    [1.0, len(y_train) / (2 * y_train.sum())], dtype=torch.float32
).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training ──────────────────────────────────────────────────────────────────
best_val_f1  = 0.0
best_state   = None
patience     = 12
no_improve   = 0

print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve  = 0
    else:
        no_improve += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Val F1: {val_f1:.4f} | Best: {best_val_f1:.4f}")

    if no_improve >= patience:
        print(f"  Early stop at epoch {epoch}")
        break

# ── Threshold tuning on validation set ───────────────────────────────────────
model.load_state_dict(best_state)
model.eval()

val_probs, val_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
        val_probs.extend(probs)
        val_labels.extend(yb.numpy())

val_probs  = np.array(val_probs)
val_labels = np.array(val_labels)

best_thresh, threshold_metrics, _ = optimize_threshold(
    val_labels,
    val_probs,
    objective=THRESHOLD_OBJECTIVE,
    n_trials=THRESHOLD_TRIALS,
    seed=RANDOM_STATE,
    low=0.01,
    high=0.99,
)

print(
    f"\nBest threshold ({THRESHOLD_OBJECTIVE}): {best_thresh:.3f} "
    f"(Val F1: {threshold_metrics['f1_macro']:.4f} | "
    f"Youden J: {threshold_metrics['youden_j']:.4f} | F_SS: {threshold_metrics['f_ss']:.4f})"
)

# ── Test evaluation ───────────────────────────────────────────────────────────
test_probs, test_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
        test_probs.extend(probs)
        test_labels.extend(yb.numpy())

test_probs  = np.array(test_probs)
test_labels = np.array(test_labels)
test_preds  = (test_probs >= best_thresh).astype(int)
test_threshold_metrics = compute_threshold_metrics(test_labels, test_probs, best_thresh)

acc  = accuracy_score(test_labels, test_preds)
f1   = f1_score(test_labels, test_preds, average='macro', zero_division=0)
rec  = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
prec = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
auc  = roc_auc_score(test_labels, test_probs)
cm   = confusion_matrix(test_labels, test_preds)

print(f"\nTest Results:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Macro F1 : {f1:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  AUROC    : {auc:.4f}")
print(f"  Confusion Matrix:\n{cm}")

# ── Save model ────────────────────────────────────────────────────────────────
torch.save({
    'state_dict': model.state_dict(),
    'threshold':  best_thresh,
    'threshold_objective': THRESHOLD_OBJECTIVE,
    'threshold_metrics': threshold_metrics,
    'test_threshold_metrics': test_threshold_metrics,
    'config': {'input_dim': 768, 'dropout': DROPOUT},
}, 'saved_models/cough_opera_mlp.pt')
print("\nSaved: saved_models/cough_opera_mlp.pt")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Symptomatic'],
            yticklabels=['Healthy', 'Symptomatic'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Cough Agent — Confusion Matrix\nMacro F1={f1:.3f} | AUROC={auc:.3f}')
plt.tight_layout()
fig.savefig('outputs/cough_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/cough_confusion_matrix.png")

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Cough Agent — ROC Curve')
ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig('outputs/cough_roc_curve.png', dpi=150)
plt.close()
print("Saved: outputs/cough_roc_curve.png")

# ── Save results JSON ─────────────────────────────────────────────────────────
results = {
    'model': 'CoughMLP (OPERA-CT)',
    'task':  'Cough Classification (Healthy vs Symptomatic)',
    'accuracy':  round(acc, 4),
    'f1_macro':  round(f1, 4),
    'recall':    round(rec, 4),
    'precision': round(prec, 4),
    'auroc':     round(auc, 4),
    'threshold': round(best_thresh, 4),
    'train_size': len(X_train),
    'val_size':   len(X_val),
    'test_size':  len(X_test),
}
with open('outputs/results_cough.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: outputs/results_cough.json")
