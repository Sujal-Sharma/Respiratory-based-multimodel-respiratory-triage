"""
scripts/train_binary_agent.py — Train COPD or Pneumonia binary MLP classifier.

Change DISEASE at the top to switch between agents.
Run twice:
    DISEASE = 'COPD'      → saved_models/copd_opera_mlp.pt
    DISEASE = 'Pneumonia' → saved_models/pneumonia_opera_mlp.pt

Requirements: run scripts/extract_opera_embeddings.py first.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import BinaryMLPClassifier, FocalLoss
from models.embedding_dataset import EmbeddingDataset
from utils.threshold_optimizer import optimize_threshold, compute_threshold_metrics

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — change DISEASE to 'Pneumonia' for the second agent
# ══════════════════════════════════════════════════════════════════════════════
DISEASE         = 'Pneumonia'
CSV_PATH        = f'data/{DISEASE.lower()}_binary_labels_with_embeddings.csv'
MODEL_SAVE_PATH = f'saved_models/{DISEASE.lower()}_opera_mlp.pt'
RESULTS_PATH    = f'outputs/results_{DISEASE.lower()}.json'

INPUT_DIM      = 768   # OPERA-CT HT-SAT output dimension
HIDDEN_DIMS    = [256, 64]
DROPOUT        = 0.3
BATCH_SIZE     = 64
MAX_EPOCHS     = 100
PATIENCE       = 15      # early stopping patience
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TARGET_RECALL  = 0.80    # clinical safety: must not miss 80%+ of cases
RANDOM_STATE   = 42
THRESHOLD_OBJECTIVE = 'youden_j'   # switch to 'f_ss' if you prefer symmetric sens/spec optimisation
THRESHOLD_TRIALS    = 200
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[train] Disease: {DISEASE} | Device: {device}")

# ── Load and split dataset ──────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna(subset=['embedding_path'])
print(f"[train] Total samples: {len(df)}")
print(f"[train] Label distribution:\n{df['label'].value_counts()}")

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label'], random_state=RANDOM_STATE
)

train_df.to_csv(f'data/{DISEASE.lower()}_train_split.csv', index=False)
val_df.to_csv(  f'data/{DISEASE.lower()}_val_split.csv',   index=False)
test_df.to_csv( f'data/{DISEASE.lower()}_test_split.csv',  index=False)

print(f"[train] Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Datasets ────────────────────────────────────────────────────────────────
train_dataset = EmbeddingDataset(f'data/{DISEASE.lower()}_train_split.csv', augment=True)
val_dataset   = EmbeddingDataset(f'data/{DISEASE.lower()}_val_split.csv',   augment=False)
test_dataset  = EmbeddingDataset(f'data/{DISEASE.lower()}_test_split.csv',  augment=False)

# WeightedRandomSampler — balanced batches for minority class
labels  = train_df['label'].values
n_pos   = (labels == 1).sum()
n_neg   = (labels == 0).sum()
weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model, loss, optimiser ──────────────────────────────────────────────────
model     = BinaryMLPClassifier(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# ── Training loop ────────────────────────────────────────────────────────────
best_val_f1    = 0.0
patience_counter = 0
best_model_state = None

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for embeddings, labels_batch in train_loader:
        embeddings   = embeddings.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validate
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for embeddings, labels_batch in val_loader:
            logits = model(embeddings.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = (probs > 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    val_f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1      = val_f1
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {train_loss/len(train_loader):.4f} "
              f"| Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}")

    if patience_counter >= PATIENCE:
        print(f"[train] Early stopping at epoch {epoch}")
        break

# ── Threshold tuning on validation set ─────────────────────────────────────
model.load_state_dict(best_model_state)
model.eval()

all_probs_val, all_labels_val = [], []
with torch.no_grad():
    for embeddings, labels_batch in val_loader:
        logits = model(embeddings.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs_val.extend(probs.cpu().numpy())
        all_labels_val.extend(labels_batch.numpy())

all_probs_val  = np.array(all_probs_val)
all_labels_val = np.array(all_labels_val)

print(f"\n[train] Optimizing threshold with Optuna ({THRESHOLD_OBJECTIVE})...")
best_threshold, threshold_metrics, threshold_study = optimize_threshold(
    all_labels_val,
    all_probs_val,
    objective=THRESHOLD_OBJECTIVE,
    n_trials=THRESHOLD_TRIALS,
    seed=RANDOM_STATE,
    low=0.01,
    high=0.99,
    min_recall=TARGET_RECALL,
)

alt_objective = 'f_ss' if THRESHOLD_OBJECTIVE == 'youden_j' else 'youden_j'
alt_threshold, alt_metrics, _ = optimize_threshold(
    all_labels_val,
    all_probs_val,
    objective=alt_objective,
    n_trials=THRESHOLD_TRIALS,
    seed=RANDOM_STATE,
    low=0.01,
    high=0.99,
    min_recall=TARGET_RECALL,
)

print(
    f"[train] Best threshold ({THRESHOLD_OBJECTIVE}): {best_threshold:.3f} | "
    f"Recall: {threshold_metrics['recall']:.4f} | F1: {threshold_metrics['f1_macro']:.4f} | "
    f"Youden J: {threshold_metrics['youden_j']:.4f} | F_SS: {threshold_metrics['f_ss']:.4f}"
)
print(
    f"[train] Alternate threshold ({alt_objective}): {alt_threshold:.3f} | "
    f"Recall: {alt_metrics['recall']:.4f} | F1: {alt_metrics['f1_macro']:.4f}"
)

# ── Final evaluation on held-out test set ──────────────────────────────────
print("\n[train] Evaluating on test set...")
all_probs_test, all_labels_test = [], []
with torch.no_grad():
    for embeddings, labels_batch in test_loader:
        logits = model(embeddings.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs_test.extend(probs.cpu().numpy())
        all_labels_test.extend(labels_batch.numpy())

all_probs_test  = np.array(all_probs_test)
all_labels_test = np.array(all_labels_test)
all_preds_test  = (all_probs_test >= best_threshold).astype(int)

test_threshold_metrics = compute_threshold_metrics(all_labels_test, all_probs_test, best_threshold)

test_results = {
    'disease':   DISEASE,
    'threshold': float(best_threshold),
    'threshold_objective': THRESHOLD_OBJECTIVE,
    'threshold_metrics': threshold_metrics,
    'alternate_thresholds': {
        alt_objective: {
            'threshold': float(alt_threshold),
            'metrics': alt_metrics,
        }
    },
    'accuracy':  float(accuracy_score(all_labels_test, all_preds_test)),
    'f1_macro':  float(f1_score(all_labels_test, all_preds_test, average='macro')),
    'recall':    float(recall_score(all_labels_test, all_preds_test, pos_label=1)),
    'precision': float(precision_score(all_labels_test, all_preds_test, pos_label=1, zero_division=0)),
    'auroc':     float(roc_auc_score(all_labels_test, all_probs_test)),
    'test_threshold_metrics': test_threshold_metrics,
}

print("\n[train] Test Results:")
for k, v in test_results.items():
    print(f"  {k}: {v}")

# ── Save model + metadata ───────────────────────────────────────────────────
os.makedirs('saved_models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

torch.save({
    'model_state_dict': best_model_state,
    'threshold':        best_threshold,
    'threshold_objective': THRESHOLD_OBJECTIVE,
    'threshold_metrics': threshold_metrics,
    'threshold_alternatives': {
        alt_objective: {
            'threshold': float(alt_threshold),
            'metrics': alt_metrics,
        }
    },
    'hidden_dims':      HIDDEN_DIMS,
    'input_dim':        INPUT_DIM,
    'test_results':     test_results,
    'disease':          DISEASE,
}, MODEL_SAVE_PATH)

with open(RESULTS_PATH, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n[train] Model saved to {MODEL_SAVE_PATH}")
print(f"[train] Results saved to {RESULTS_PATH}")
