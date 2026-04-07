"""
scripts/train_baselines.py — Baseline comparison models for the paper.

Baselines:
  1. MFCC+MLP     — 40 MFCCs (mean+std = 80 features) → 2-layer MLP
  2. MFCC+LR      — same features → Logistic Regression (linear baseline)
  3. Random        — majority class predictor

Runs on: COPD binary and Pneumonia binary datasets.
Uses same train/val/test splits as OPERA models for fair comparison.
Pneumonia uses 5-fold CV (same as OPERA evaluation).

Output: outputs/results_baselines.json
        outputs/baseline_comparison.png
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, accuracy_score, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.makedirs('outputs', exist_ok=True)

SAMPLE_RATE = 16000
DURATION    = 8        # seconds — same as OPERA
N_MFCC      = 40
RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_mfcc(file_path: str) -> np.ndarray | None:
    """
    Extract 40 MFCCs → mean + std = 80-dim feature vector.
    Returns None on failure.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        if len(y) < SAMPLE_RATE:          # skip files shorter than 1 second
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    except Exception:
        return None


def extract_features_from_df(df: pd.DataFrame, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract MFCC features for all rows. Returns (X, y) dropping failed files."""
    X, y = [], []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        feat = extract_mfcc(str(row['file_path']))
        if feat is not None:
            X.append(feat)
            y.append(int(row['label']))
        else:
            failed += 1
    if failed:
        print(f"  Skipped {failed} files (too short or unreadable)")
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════════════════
# Model evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def eval_binary(y_true, y_pred, y_prob, model_name, disease):
    metrics = {
        'model':     model_name,
        'disease':   disease,
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1_macro':  float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'auroc':     float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    print(f"  {model_name:20s} | Acc:{metrics['accuracy']:.3f} | "
          f"F1:{metrics['f1_macro']:.3f} | Recall:{metrics['recall']:.3f} | AUROC:{metrics['auroc']:.3f}")
    return metrics


def run_baselines_single_split(train_df, test_df, disease):
    """Run baselines on a fixed train/test split (COPD)."""
    print(f"\n  Extracting MFCC features...")
    X_train, y_train = extract_features_from_df(train_df, f"  Train ({disease})")
    X_test,  y_test  = extract_features_from_df(test_df,  f"  Test  ({disease})")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression (linear probe)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, C=1.0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results.append(eval_binary(y_test, y_pred, y_prob, 'MFCC + LR', disease))

    # 2. MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation='relu',
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_STATE, learning_rate_init=1e-3
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    results.append(eval_binary(y_test, y_pred, y_prob, 'MFCC + MLP', disease))

    # 3. Majority class baseline
    majority = int(np.bincount(y_train).argmax())
    y_pred   = np.full_like(y_test, majority)
    y_prob   = np.zeros_like(y_test, dtype=float)
    results.append(eval_binary(y_test, y_pred, y_prob, 'Majority Class', disease))

    return results


def run_baselines_cv(df, disease, n_folds=5):
    """Run baselines with stratified CV (Pneumonia)."""
    print(f"\n  Extracting MFCC features (full dataset)...")
    X_all, y_all = extract_features_from_df(df, f"  {disease}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    model_names = ['MFCC + LR', 'MFCC + MLP']
    oof_probs   = {m: np.zeros(len(X_all)) for m in model_names}
    oof_preds   = {m: np.zeros(len(X_all), dtype=int) for m in model_names}

    for fold, (ti, vi) in enumerate(skf.split(X_all, y_all)):
        X_tr, X_vl = X_all[ti], X_all[vi]
        y_tr        = y_all[ti]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_vl   = scaler.transform(X_vl)

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        lr.fit(X_tr, y_tr)
        oof_probs['MFCC + LR'][vi] = lr.predict_proba(X_vl)[:, 1]
        oof_preds['MFCC + LR'][vi] = lr.predict(X_vl)

        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu',
                            max_iter=300, early_stopping=True, validation_fraction=0.1,
                            random_state=RANDOM_STATE, learning_rate_init=1e-3)
        mlp.fit(X_tr, y_tr)
        oof_probs['MFCC + MLP'][vi] = mlp.predict_proba(X_vl)[:, 1]
        oof_preds['MFCC + MLP'][vi] = mlp.predict(X_vl)

        print(f"    Fold {fold+1}/{n_folds} done")

    results = []
    for m in model_names:
        results.append(eval_binary(y_all, oof_preds[m], oof_probs[m], m, disease))

    majority = int(np.bincount(y_all).argmax())
    y_pred   = np.full_like(y_all, majority)
    y_prob   = np.zeros_like(y_all, dtype=float)
    results.append(eval_binary(y_all, y_pred, y_prob, 'Majority Class', disease))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# COPD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COPD BASELINES (train/test split)")
print("="*60)

copd_train = pd.read_csv('data/copd_train_split.csv')
copd_val   = pd.read_csv('data/copd_val_split.csv')
copd_test  = pd.read_csv('data/copd_test_split.csv')
# Combine train+val to match OPERA training set size
copd_trainval = pd.concat([copd_train, copd_val], ignore_index=True)

print(f"  Train+Val: {len(copd_trainval)} | Test: {len(copd_test)}")
print(f"  Test positives: {copd_test['label'].sum()}")

copd_results = run_baselines_single_split(copd_trainval, copd_test, 'COPD')

# ══════════════════════════════════════════════════════════════════════════════
# Pneumonia
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PNEUMONIA BASELINES (5-fold CV — same as OPERA)")
print("="*60)

pneu_df = pd.read_csv('data/pneumonia_binary_labels_with_embeddings.csv').dropna(
    subset=['embedding_path'])
print(f"  Total: {len(pneu_df)} | Positives: {pneu_df['label'].sum()}")

pneu_results = run_baselines_cv(pneu_df, 'Pneumonia')

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
all_results = {
    'COPD':      copd_results,
    'Pneumonia': pneu_results,
}

with open('outputs/results_baselines.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\n  Saved: outputs/results_baselines.json")

# ══════════════════════════════════════════════════════════════════════════════
# Comparison chart — baselines vs OPERA
# ══════════════════════════════════════════════════════════════════════════════
opera_results = {
    'COPD':      {'f1_macro': 0.947, 'recall': 0.959, 'auroc': 0.995},
    'Pneumonia': {'f1_macro': 0.869, 'recall': 0.731, 'auroc': 0.984},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics_to_plot = ['f1_macro', 'recall', 'auroc']
metric_labels   = ['Macro F1', 'Recall', 'AUROC']
colors = ['#90CAF9', '#A5D6A7', '#FFCC80', '#EF9A9A']

for ax, (disease, baseline_list) in zip(axes, all_results.items()):
    models = [r['model'] for r in baseline_list] + ['OPERA-MLP (ours)']
    x      = np.arange(len(models))
    width  = 0.25

    opera_row = opera_results[disease]
    all_rows  = baseline_list + [{'f1_macro': opera_row['f1_macro'],
                                   'recall':   opera_row['recall'],
                                   'auroc':    opera_row['auroc']}]

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        vals = [r[metric] for r in all_rows]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=9, rotation=10, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{disease} Detection — Baseline vs OPERA-MLP', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.axvline(x=len(models) - 1.4, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('outputs/baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: outputs/baseline_comparison.png")

# Print final summary table
print("\n" + "="*70)
print("FULL COMPARISON TABLE")
print("="*70)
print(f"{'Model':<22} {'Disease':<12} {'F1':>6} {'Recall':>8} {'AUROC':>8}")
print("-"*70)
for disease, results in all_results.items():
    for r in results:
        print(f"  {r['model']:<20} {disease:<12} {r['f1_macro']:>6.3f} {r['recall']:>8.3f} {r['auroc']:>8.3f}")
    opera = opera_results[disease]
    print(f"  {'OPERA-MLP (ours)':<20} {disease:<12} {opera['f1_macro']:>6.3f} {opera['recall']:>8.3f} {opera['auroc']:>8.3f}")
    print()
