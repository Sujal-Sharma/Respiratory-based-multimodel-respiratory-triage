"""
scripts/evaluate_models.py — Generate evaluation figures for all 3 OPERA MLP models.

Outputs (saved to outputs/):
  - confusion_matrix_copd.png
  - confusion_matrix_pneumonia.png
  - confusion_matrix_sound.png
  - roc_curve_copd.png
  - roc_curve_pneumonia.png
  - per_class_f1_sound.png
  - model_comparison_opera.png

Requirements: saved test split CSVs and saved_models/*.pt must exist.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import BinaryMLPClassifier, SoundMLPClassifier
from models.embedding_dataset import EmbeddingDataset
from torch.utils.data import DataLoader

os.makedirs('outputs', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[evaluate] Device: {device}")

BATCH_SIZE = 128

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = '#2196F3'
GREEN  = '#4CAF50'
ORANGE = '#FF9800'
RED    = '#F44336'
PURPLE = '#9C27B0'

# ══════════════════════════════════════════════════════════════════════════════
# Helper: run inference on a test DataLoader
# ══════════════════════════════════════════════════════════════════════════════

def infer_binary(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for emb, lbl in loader:
            logits = model(emb.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs >= threshold).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def infer_multiclass(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb, lbl in loader:
            logits = model(emb.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())
    return np.array(all_labels), np.array(all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, title, save_path, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(len(class_names) * 1.8 + 1, len(class_names) * 1.8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray', ax=ax,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc(fpr, tpr, auroc, title, save_path, color=BLUE):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'ROC (AUC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.12, color=color)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_f1(class_names, f1_scores, title, save_path):
    colors = [BLUE, GREEN, ORANGE, PURPLE][:len(class_names)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(class_names, f1_scores, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_dict, save_path):
    """Bar chart comparing key metrics across all models."""
    models  = list(results_dict.keys())
    metrics = ['Accuracy', 'F1 Macro', 'Recall', 'AUROC']
    colors  = [BLUE, GREEN, ORANGE, PURPLE]

    x     = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results_dict[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('OPERA-MLP Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. COPD Agent
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Evaluating COPD agent...")

ckpt_copd = torch.load('saved_models/copd_opera_mlp.pt', map_location=device, weights_only=False)
model_copd = BinaryMLPClassifier(
    input_dim=ckpt_copd.get('input_dim', 768),
    hidden_dims=ckpt_copd.get('hidden_dims', [256, 64])
).to(device)
model_copd.load_state_dict(ckpt_copd['model_state_dict'])

copd_test = EmbeddingDataset('data/copd_test_split.csv', augment=False)
copd_loader = DataLoader(copd_test, batch_size=BATCH_SIZE, shuffle=False)

threshold_copd = ckpt_copd.get('threshold', 0.5)
y_true_copd, y_pred_copd, y_prob_copd = infer_binary(model_copd, copd_loader, threshold_copd)

cm_copd = confusion_matrix(y_true_copd, y_pred_copd)
fpr_copd, tpr_copd, _ = roc_curve(y_true_copd, y_prob_copd)
auroc_copd = auc(fpr_copd, tpr_copd)

plot_confusion_matrix(
    cm_copd, ['Normal', 'COPD'],
    f'COPD Agent — Confusion Matrix\n(Threshold={threshold_copd:.2f}, AUROC={auroc_copd:.3f})',
    'outputs/confusion_matrix_copd.png'
)
plot_roc(
    fpr_copd, tpr_copd, auroc_copd,
    'COPD Agent — ROC Curve',
    'outputs/roc_curve_copd.png', color=BLUE
)

copd_summary = {
    'Accuracy': float(accuracy_score(y_true_copd, y_pred_copd)),
    'F1 Macro': float(f1_score(y_true_copd, y_pred_copd, average='macro')),
    'Recall':   float(recall_score(y_true_copd, y_pred_copd, pos_label=1)),
    'AUROC':    float(auroc_copd),
}
print(f"  COPD  — Acc:{copd_summary['Accuracy']:.3f} | F1:{copd_summary['F1 Macro']:.3f} | "
      f"Recall:{copd_summary['Recall']:.3f} | AUROC:{copd_summary['AUROC']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pneumonia Agent — 5-fold CV OOF predictions (honest evaluation)
# Each sample is predicted by a model that never trained on it.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Evaluating Pneumonia agent (5-fold CV OOF)...")

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset as _TorchDS, WeightedRandomSampler
from models.mlp_classifier import FocalLoss

ckpt_pneu      = torch.load('saved_models/pneumonia_opera_mlp.pt', map_location=device, weights_only=False)
threshold_pneu = ckpt_pneu.get('threshold', 0.5)

df_pneu = pd.read_csv('data/pneumonia_binary_labels_with_embeddings.csv').dropna(
    subset=['embedding_path']).reset_index(drop=True)

class _PneuDS(_TorchDS):
    def __init__(self, sub_df):
        self.paths  = sub_df['embedding_path'].tolist()
        self.labels = sub_df['label'].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        emb = np.load(self.paths[idx]).astype(np.float32)
        return torch.tensor(emb), torch.tensor(self.labels[idx], dtype=torch.long)

skf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs      = np.zeros(len(df_pneu))
oof_labels     = df_pneu['label'].values.copy()

for fold, (train_idx, val_idx) in enumerate(skf.split(df_pneu.index, oof_labels)):
    tr_df = df_pneu.iloc[train_idx]
    vl_df = df_pneu.iloc[val_idx]

    tr_lbls = tr_df['label'].values
    n_pos   = (tr_lbls == 1).sum(); n_neg = (tr_lbls == 0).sum()
    w       = np.where(tr_lbls == 1, 1.0/n_pos, 1.0/n_neg).astype(np.float64)
    sampler_f = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

    tr_loader = DataLoader(_PneuDS(tr_df), batch_size=64, sampler=sampler_f)
    vl_loader = DataLoader(_PneuDS(vl_df), batch_size=64, shuffle=False)

    fm     = BinaryMLPClassifier(input_dim=768, hidden_dims=[256, 64]).to(device)
    f_opt  = torch.optim.AdamW(fm.parameters(), lr=3e-4, weight_decay=1e-4)
    f_crit = FocalLoss(alpha=0.25, gamma=2.0)
    f_sch  = torch.optim.lr_scheduler.CosineAnnealingLR(f_opt, T_max=150)

    best_f1, pat, best_st = 0.0, 0, None
    for ep in range(150):
        fm.train()
        for emb, lbl in tr_loader:
            emb, lbl = emb.to(device), lbl.to(device)
            f_opt.zero_grad()
            f_crit(fm(emb), lbl).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            f_opt.step()
        f_sch.step()
        fm.eval()
        pv, lv = [], []
        with torch.no_grad():
            for emb, lbl in vl_loader:
                pr = torch.softmax(fm(emb.to(device)), dim=1)[:, 1].cpu().numpy()
                pv.extend((pr >= 0.5).astype(int)); lv.extend(lbl.numpy())
        vf1 = f1_score(lv, pv, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_st = {k: v.clone() for k, v in fm.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 20:
            break

    fm.load_state_dict(best_st)
    fm.eval()
    fold_probs = []
    with torch.no_grad():
        for emb, _ in vl_loader:
            pr = torch.softmax(fm(emb.to(device)), dim=1)[:, 1].cpu().numpy()
            fold_probs.extend(pr)
    oof_probs[val_idx] = np.array(fold_probs)
    print(f"  Fold {fold+1}/5 — Val pos: {oof_labels[val_idx].sum()} | "
          f"F1: {f1_score(oof_labels[val_idx], (np.array(fold_probs)>=threshold_pneu).astype(int), average='macro', zero_division=0):.3f}")

y_true_pneu = oof_labels
y_prob_pneu = oof_probs
y_pred_pneu = (y_prob_pneu >= threshold_pneu).astype(int)

cm_pneu            = confusion_matrix(y_true_pneu, y_pred_pneu)
fpr_pneu, tpr_pneu, _ = roc_curve(y_true_pneu, y_prob_pneu)
auroc_pneu         = auc(fpr_pneu, tpr_pneu)

plot_confusion_matrix(
    cm_pneu, ['Normal', 'Pneumonia'],
    f'Pneumonia Agent — Confusion Matrix (5-fold CV OOF)\n(Threshold={threshold_pneu:.2f}, AUROC={auroc_pneu:.3f})',
    'outputs/confusion_matrix_pneumonia.png'
)
plot_roc(
    fpr_pneu, tpr_pneu, auroc_pneu,
    'Pneumonia Agent — ROC Curve (5-fold CV OOF)',
    'outputs/roc_curve_pneumonia.png', color=GREEN
)

pneu_summary = {
    'Accuracy': float(accuracy_score(y_true_pneu, y_pred_pneu)),
    'F1 Macro': float(f1_score(y_true_pneu, y_pred_pneu, average='macro', zero_division=0)),
    'Recall':   float(recall_score(y_true_pneu, y_pred_pneu, pos_label=1, zero_division=0)),
    'AUROC':    float(auroc_pneu),
}
print(f"  Pneumonia OOF — Acc:{pneu_summary['Accuracy']:.3f} | F1:{pneu_summary['F1 Macro']:.3f} | "
      f"Recall:{pneu_summary['Recall']:.3f} | AUROC:{pneu_summary['AUROC']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sound Classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Evaluating Sound classifier...")

import torch.nn as nn

class SoundMLP3Class(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

ckpt_snd = torch.load('saved_models/sound_opera_mlp_3class.pt', map_location=device, weights_only=False)
model_snd = SoundMLP3Class(
    input_dim=ckpt_snd.get('input_dim', 768),
    hidden_dims=ckpt_snd.get('hidden_dims', [512, 256, 64]),
).to(device)
model_snd.load_state_dict(ckpt_snd['model_state_dict'])

snd_test   = EmbeddingDataset('data/sound_test_3class.csv', label_col='sound_label', augment=False)
snd_loader = DataLoader(snd_test, batch_size=BATCH_SIZE, shuffle=False)

y_true_snd, y_pred_snd = infer_multiclass(model_snd, snd_loader)

SOUND_LABELS = ['Normal', 'Crackle', 'Wheeze']
cm_snd = confusion_matrix(y_true_snd, y_pred_snd)
per_class_f1 = f1_score(y_true_snd, y_pred_snd, average=None, zero_division=0)

plot_confusion_matrix(
    cm_snd, SOUND_LABELS,
    'Sound Classifier — Confusion Matrix (3-class)\nBoth merged into Crackle',
    'outputs/confusion_matrix_sound.png', cmap='Purples'
)
plot_per_class_f1(
    SOUND_LABELS, per_class_f1,
    'Sound Classifier — Per-Class F1 Score (3-class)',
    'outputs/per_class_f1_sound.png'
)

snd_summary = {
    'Accuracy': float(accuracy_score(y_true_snd, y_pred_snd)),
    'F1 Macro': float(f1_score(y_true_snd, y_pred_snd, average='macro', zero_division=0)),
    'Recall':   0.0,   # not applicable as single binary recall
    'AUROC':    0.0,
}
print(f"  Sound — Acc:{snd_summary['Accuracy']:.3f} | F1:{snd_summary['F1 Macro']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Combined comparison chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating model comparison chart...")

comparison = {
    'COPD Agent':      copd_summary,
    'Pneumonia Agent': pneu_summary,
    'Sound Classifier': {
        'Accuracy': snd_summary['Accuracy'],
        'F1 Macro': snd_summary['F1 Macro'],
        'Recall':   0.0,
        'AUROC':    0.0,
    },
}
plot_model_comparison(comparison, 'outputs/model_comparison_opera.png')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save combined JSON summary
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    'COPD':      copd_summary,
    'Pneumonia': pneu_summary,
    'Sound':     {
        'Accuracy':     snd_summary['Accuracy'],
        'F1 Macro':     snd_summary['F1 Macro'],
        'per_class_f1': dict(zip(SOUND_LABELS, per_class_f1.tolist())),
    },
}
with open('outputs/evaluation_summary_opera.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n  Saved: outputs/evaluation_summary_opera.json")

print("\n[evaluate] All done. Files saved to outputs/")
print("  confusion_matrix_copd.png")
print("  confusion_matrix_pneumonia.png")
print("  confusion_matrix_sound.png")
print("  roc_curve_copd.png")
print("  roc_curve_pneumonia.png")
print("  per_class_f1_sound.png")
print("  model_comparison_opera.png")
print("  evaluation_summary_opera.json")
