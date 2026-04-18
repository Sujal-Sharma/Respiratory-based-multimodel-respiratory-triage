"""
scripts/train_pneumonia_cv.py — 5-fold stratified cross-validation for Pneumonia.

Only 52 positive samples exist, making a single test split unreliable (8 positives).
5-fold CV gives ~10 positives per fold, uses all data for evaluation.

Final model is retrained on full dataset and saved for inference.
Output: saved_models/pneumonia_opera_mlp.pt (overwrites previous)
        outputs/results_pneumonia.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import BinaryMLPClassifier, FocalLoss
from models.embedding_dataset import EmbeddingDataset
from utils.threshold_optimizer import optimize_threshold

# ══════════════════════════════════════════════════════════════════════════════
DISEASE         = 'Pneumonia'
CSV_PATH        = 'data/pneumonia_binary_labels_with_embeddings.csv'
MODEL_SAVE_PATH = 'saved_models/pneumonia_opera_mlp.pt'
RESULTS_PATH    = 'outputs/results_pneumonia.json'

INPUT_DIM      = 768
HIDDEN_DIMS    = [256, 64]
DROPOUT        = 0.3
BATCH_SIZE     = 64
MAX_EPOCHS     = 150
PATIENCE       = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
TARGET_RECALL  = 0.80
N_FOLDS        = 5
RANDOM_STATE   = 42
THRESHOLD_OBJECTIVE = 'youden_j'   # switch to 'f_ss' if preferred for symmetric sensitivity/specificity
THRESHOLD_TRIALS    = 200
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[pneumonia_cv] Device: {device}")

df = pd.read_csv(CSV_PATH).dropna(subset=['embedding_path']).reset_index(drop=True)
print(f"[pneumonia_cv] Total samples: {len(df)}")
print(f"[pneumonia_cv] Label distribution:\n{df['label'].value_counts()}")


class EmbeddingDatasetFromDF(Dataset):
    def __init__(self, sub_df, augment=False):
        self.paths  = sub_df['embedding_path'].tolist()
        self.labels = sub_df['label'].tolist()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = np.load(self.paths[idx]).astype(np.float32)
        if self.augment:
            emb = emb + np.random.normal(0, 0.01, emb.shape).astype(np.float32)
            emb = emb * np.random.uniform(0.95, 1.05)
            n = np.linalg.norm(emb)
            if n > 0: emb /= n
        return torch.tensor(emb), torch.tensor(self.labels[idx], dtype=torch.long)


def make_sampler(labels_arr):
    n_pos = (labels_arr == 1).sum()
    n_neg = (labels_arr == 0).sum()
    weights = np.where(labels_arr == 1, 1.0/n_pos, 1.0/n_neg).astype(np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_one_fold(train_df, val_df):
    train_ds = EmbeddingDatasetFromDF(train_df, augment=True)
    val_ds   = EmbeddingDatasetFromDF(val_df,   augment=False)

    sampler = make_sampler(train_df['label'].values)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = BinaryMLPClassifier(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_f1      = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for emb, lbl in train_loader:
            emb, lbl = emb.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_v, labels_v, probs_v = [], [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                logits = model(emb.to(device))
                probs  = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                preds_v.extend((probs >= 0.5).astype(int))
                probs_v.extend(probs)
                labels_v.extend(lbl.numpy())

        val_f1 = f1_score(labels_v, preds_v, average='macro', zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Threshold tuning on this fold's val set
    model.load_state_dict(best_state)
    model.eval()
    probs_v2, labels_v2 = [], []
    with torch.no_grad():
        for emb, lbl in val_loader:
            probs = torch.softmax(model(emb.to(device)), dim=1)[:,1].cpu().numpy()
            probs_v2.extend(probs)
            labels_v2.extend(lbl.numpy())

    probs_v2  = np.array(probs_v2)
    labels_v2 = np.array(labels_v2)

    best_thresh, best_metrics, _ = optimize_threshold(
        labels_v2,
        probs_v2,
        objective=THRESHOLD_OBJECTIVE,
        n_trials=THRESHOLD_TRIALS,
        seed=RANDOM_STATE,
        low=0.01,
        high=0.99,
        min_recall=TARGET_RECALL,
    )

    return best_state, best_thresh, np.array(probs_v2), np.array(labels_v2), best_metrics


# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
X = df.index.values
y = df['label'].values

fold_metrics   = []
all_probs_oof  = np.zeros(len(df))
all_labels_oof = np.zeros(len(df), dtype=int)
thresholds     = []
threshold_metrics_per_fold = []

print(f"\n[pneumonia_cv] Running {N_FOLDS}-fold CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold+1}/{N_FOLDS} | Train: {len(train_idx)} | Val: {len(val_idx)} "
          f"| Val Pos: {y[val_idx].sum()}")
    train_df_fold = df.iloc[train_idx]
    val_df_fold   = df.iloc[val_idx]

    state, thresh, probs, labels, fold_threshold_metrics = train_one_fold(train_df_fold, val_df_fold)
    all_probs_oof[val_idx]  = probs
    all_labels_oof[val_idx] = labels
    thresholds.append(thresh)
    threshold_metrics_per_fold.append(fold_threshold_metrics)

    preds = (probs >= thresh).astype(int)
    m = {
        'fold':      fold + 1,
        'threshold': float(thresh),
        'accuracy':  float(accuracy_score(labels, preds)),
        'f1_macro':  float(f1_score(labels, preds, average='macro', zero_division=0)),
        'recall':    float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        'precision': float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        'auroc':     float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.0,
    }
    fold_metrics.append(m)
    print(f"    F1={m['f1_macro']:.3f} | Recall={m['recall']:.3f} | AUROC={m['auroc']:.3f} | Thresh={thresh:.2f}")

# ── Aggregate OOF results ─────────────────────────────────────────────────────
final_threshold, final_threshold_metrics, final_threshold_study = optimize_threshold(
    all_labels_oof,
    all_probs_oof,
    objective=THRESHOLD_OBJECTIVE,
    n_trials=THRESHOLD_TRIALS,
    seed=RANDOM_STATE,
    low=0.01,
    high=0.99,
    min_recall=TARGET_RECALL,
)
oof_preds = (all_probs_oof >= final_threshold).astype(int)

print(f"\n[pneumonia_cv] OOF Results (threshold={final_threshold:.2f}):")
print(classification_report(all_labels_oof, oof_preds,
                             target_names=['Normal', 'Pneumonia'], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(all_labels_oof, oof_preds))

cv_results = {
    'disease':          DISEASE,
    'evaluation':       '5-fold stratified CV',
    'threshold':        final_threshold,
    'threshold_objective': THRESHOLD_OBJECTIVE,
    'threshold_metrics': final_threshold_metrics,
    'fold_thresholds':   [float(t) for t in thresholds],
    'fold_threshold_metrics': threshold_metrics_per_fold,
    'accuracy':         float(accuracy_score(all_labels_oof, oof_preds)),
    'f1_macro':         float(f1_score(all_labels_oof, oof_preds, average='macro', zero_division=0)),
    'recall':           float(recall_score(all_labels_oof, oof_preds, pos_label=1, zero_division=0)),
    'precision':        float(precision_score(all_labels_oof, oof_preds, pos_label=1, zero_division=0)),
    'auroc':            float(roc_auc_score(all_labels_oof, all_probs_oof)),
    'fold_metrics':     fold_metrics,
    'mean_fold_f1':     float(np.mean([m['f1_macro'] for m in fold_metrics])),
    'std_fold_f1':      float(np.std([m['f1_macro'] for m in fold_metrics])),
    'mean_fold_recall': float(np.mean([m['recall'] for m in fold_metrics])),
    'mean_fold_auroc':  float(np.mean([m['auroc'] for m in fold_metrics])),
    'mean_fold_threshold': float(np.mean(thresholds)),
    'std_fold_threshold':  float(np.std(thresholds)),
}

print(f"\n[pneumonia_cv] Mean Fold F1: {cv_results['mean_fold_f1']:.3f} ± {cv_results['std_fold_f1']:.3f}")
print(f"[pneumonia_cv] OOF AUROC: {cv_results['auroc']:.3f}")

# ── Retrain final model on full dataset ───────────────────────────────────────
print(f"\n[pneumonia_cv] Retraining final model on full dataset...")

full_ds      = EmbeddingDatasetFromDF(df, augment=True)
sampler      = make_sampler(df['label'].values)
full_loader  = DataLoader(full_ds, batch_size=BATCH_SIZE, sampler=sampler)

final_model  = BinaryMLPClassifier(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
criterion    = FocalLoss(alpha=0.25, gamma=2.0)
optimizer    = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

# Train for same number of epochs as median early-stopping point
TRAIN_EPOCHS = 80
for epoch in range(TRAIN_EPOCHS):
    final_model.train()
    for emb, lbl in full_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(emb), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}/{TRAIN_EPOCHS}")

final_state = {k: v.clone() for k, v in final_model.state_dict().items()}

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs('saved_models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

torch.save({
    'model_state_dict': final_state,
    'threshold':        final_threshold,
    'threshold_objective': THRESHOLD_OBJECTIVE,
    'threshold_metrics': final_threshold_metrics,
    'fold_thresholds':   [float(t) for t in thresholds],
    'hidden_dims':      HIDDEN_DIMS,
    'input_dim':        INPUT_DIM,
    'test_results':     cv_results,
    'disease':          DISEASE,
}, MODEL_SAVE_PATH)

with open(RESULTS_PATH, 'w') as f:
    json.dump(cv_results, f, indent=2)

print(f"\n[pneumonia_cv] Model saved → {MODEL_SAVE_PATH}")
print(f"[pneumonia_cv] OOF F1 Macro: {cv_results['f1_macro']:.4f}")
print(f"[pneumonia_cv] OOF AUROC:    {cv_results['auroc']:.4f}")
print(f"[pneumonia_cv] OOF Recall:   {cv_results['recall']:.4f}")
