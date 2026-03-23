"""
evaluation.py — Evaluation metrics and plots for all trained models.

Produces (saved to outputs/):
  Confusion matrix, classification report, ROC curves for:
    - XGBoost (COUGHVID — 3 classes)
    - MultiTask EfficientNet Disease head (5 classes)
    - MultiTask EfficientNet Sound head   (4 classes)
  model_comparison_table.csv
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, f1_score,
    recall_score, precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, OUTPUTS_DIR, SAVED_MODELS_DIR,
    COUGHVID_CLASSES, COUGHVID_LABELS_CSV, COUGHVID_SPEC_LABELS_CSV,
    LUNG_DISEASE_CLASSES, LUNG_SOUND_CLASSES,
    MULTITASK_LABELS_CSV, BATCH_SIZE,
)
from models.cnn_model import build_multitask_efficientnet, build_light_cough_cnn
from training import MultiTaskDataset, CoughvidSpecDataset
from utils import load_checkpoint

warnings.filterwarnings('ignore')
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n + 2), max(6, n + 1)))

    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    sns.heatmap(
        cm_norm, annot=annot, fmt='', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray',
        vmin=0, vmax=1, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14,
                 fontweight='bold', pad=15)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def _plot_roc_curves(y_true, proba, class_names, model_name):
    n_classes = len(class_names)
    classes   = list(range(n_classes))
    y_bin     = label_binarize(y_true, classes=classes)
    # Binary case: label_binarize returns (N,1) — expand to (N,2)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        auc_d[i]               = auc(fpr_d[i], tpr_d[i])

    if not fpr_d:
        return 0.0

    all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in fpr_d]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr_d:
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr  /= len(fpr_d)
    macro_auc  = auc(all_fpr, mean_tpr)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, color in zip(fpr_d.keys(), colors):
        lbl = class_names[i] if i < len(class_names) else str(i)
        ax.plot(fpr_d[i], tpr_d[i], color=color, lw=1.8,
                label=f"{lbl} (AUC={auc_d[i]:.3f})")

    ax.plot(all_fpr, mean_tpr, 'k--', lw=2.5,
            label=f"Macro Avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1.2, label='Chance')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, f"roc_curve_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve      → {save_path}")
    return macro_auc


def _print_and_save_report(y_true, y_pred, class_names, model_name):
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    path = os.path.join(OUTPUTS_DIR, f"report_{model_name}.txt")
    with open(path, 'w') as f:
        f.write(f"Model: {model_name}\n\n{report}")
    print(f"  Report         → {path}")


def _metrics_summary(y_true, y_pred, proba, class_names, model_name):
    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_auc = _plot_roc_curves(y_true, proba, class_names, model_name)

    print("\n" + "─" * 40)
    print(f"  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"  Recall    : {macro_rec:.4f}")
    print(f"  Precision : {macro_pre:.4f}")
    print(f"  AUROC     : {macro_auc:.4f}")

    target_acc = 0.90
    status = "✓ ≥ 90% ACC" if accuracy >= target_acc else f"✗ < 90% ACC ({accuracy*100:.1f}%)"
    print(f"  Target    : {status}")
    print("─" * 40)

    return {
        'model':     model_name,
        'accuracy':  round(accuracy, 4),
        'f1':        round(macro_f1, 4),
        'recall':    round(macro_rec, 4),
        'precision': round(macro_pre, 4),
        'auroc':     round(macro_auc, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_xgboost(model, X_test: np.ndarray, y_test: np.ndarray,
                      threshold: float = 0.5) -> dict:
    print("\n" + "=" * 60)
    print(f"EVALUATION -- XGBoost (COUGHVID)  threshold={threshold:.3f}")
    print("=" * 60)

    proba  = model.predict_proba(X_test)
    # Binary XGBoost returns shape (N,) for positive class -- expand to (N, 2)
    if proba.ndim == 1 or proba.shape[1] == 1:
        proba = np.column_stack([1 - proba.ravel(), proba.ravel()])

    # Apply tuned threshold instead of default 0.5
    y_pred = (proba[:, 1] >= threshold).astype(int)

    _plot_confusion_matrix(y_test, y_pred, COUGHVID_CLASSES, 'xgboost')
    _print_and_save_report(y_test, y_pred, COUGHVID_CLASSES, 'xgboost')
    return _metrics_summary(y_test, y_pred, proba, COUGHVID_CLASSES, 'xgboost')


# ══════════════════════════════════════════════════════════════════════════════
# MultiTask EfficientNet evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _run_multitask_inference(model, df_test: pd.DataFrame):
    """
    Run forward pass on test set.

    Returns
    -------
    dis_proba  : np.ndarray (N_labelled, 5)
    dis_true   : np.ndarray (N_labelled,)
    snd_proba  : np.ndarray (N_all, 4)
    snd_true   : np.ndarray (N_all,)
    """
    test_ds     = MultiTaskDataset(df_test, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model.eval()
    all_dis_proba, all_dis_true = [], []
    all_snd_proba, all_snd_true = [], []

    with torch.no_grad():
        for batch_mel, batch_dis, batch_snd in test_loader:
            batch_mel = batch_mel.to(DEVICE, non_blocking=True)

            with autocast('cuda'):
                dis_out, snd_out = model(batch_mel)

            dis_proba = torch.softmax(dis_out.float(), dim=1).cpu().numpy()
            snd_proba = torch.softmax(snd_out.float(), dim=1).cpu().numpy()

            batch_dis_np = batch_dis.numpy()
            batch_snd_np = batch_snd.numpy()

            # Sound head: all samples
            all_snd_proba.append(snd_proba)
            all_snd_true.append(batch_snd_np)

            # Disease head: only labelled samples (disease_label >= 0)
            mask = batch_dis_np >= 0
            if mask.any():
                all_dis_proba.append(dis_proba[mask])
                all_dis_true.append(batch_dis_np[mask])

    snd_proba = np.concatenate(all_snd_proba, axis=0)
    snd_true  = np.concatenate(all_snd_true,  axis=0)

    if all_dis_proba:
        dis_proba = np.concatenate(all_dis_proba, axis=0)
        dis_true  = np.concatenate(all_dis_true,  axis=0)
    else:
        dis_proba = np.empty((0, 5))
        dis_true  = np.empty((0,), dtype=int)

    return dis_proba, dis_true, snd_proba, snd_true


def evaluate_multitask(model, df_test: pd.DataFrame) -> tuple:
    """
    Evaluate MultiTaskEfficientNet on both heads separately.

    Returns
    -------
    (metrics_disease, metrics_sound) : tuple of dicts
    """
    print("\n" + "═" * 60)
    print("EVALUATION — MultiTask EfficientNet (ICBHI + KAUH + HF Lung V1)")
    print("═" * 60)

    dis_proba, dis_true, snd_proba, snd_true = _run_multitask_inference(model, df_test)

    # ── Disease head ──────────────────────────────────────────────
    print("\n[Disease Head] 5 classes:", LUNG_DISEASE_CLASSES)
    print(f"  Test samples with disease labels: {len(dis_true):,}")

    if len(dis_true) > 0:
        dis_pred = np.argmax(dis_proba, axis=1)
        _plot_confusion_matrix(dis_true, dis_pred, LUNG_DISEASE_CLASSES,
                               'multitask_disease')
        _print_and_save_report(dis_true, dis_pred, LUNG_DISEASE_CLASSES,
                               'multitask_disease')
        metrics_dis = _metrics_summary(dis_true, dis_pred, dis_proba,
                                       LUNG_DISEASE_CLASSES, 'multitask_disease')
    else:
        print("  WARNING: No disease-labelled test samples found.")
        metrics_dis = {'model': 'multitask_disease', 'accuracy': 0,
                       'f1': 0, 'recall': 0, 'precision': 0, 'auroc': 0}

    # ── Sound head ────────────────────────────────────────────────
    print("\n[Sound Head] 4 classes:", LUNG_SOUND_CLASSES)
    print(f"  Test samples: {len(snd_true):,}")

    snd_pred = np.argmax(snd_proba, axis=1)
    _plot_confusion_matrix(snd_true, snd_pred, LUNG_SOUND_CLASSES,
                           'multitask_sound')
    _print_and_save_report(snd_true, snd_pred, LUNG_SOUND_CLASSES,
                           'multitask_sound')
    metrics_snd = _metrics_summary(snd_true, snd_pred, snd_proba,
                                   LUNG_SOUND_CLASSES, 'multitask_sound')

    return metrics_dis, metrics_snd


# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID EfficientNet evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_coughvid_efficientnet(model, df_test: pd.DataFrame) -> dict:
    """
    Evaluate binary EfficientNet on COUGHVID test set.
    Returns metrics dict compatible with save_comparison_table().
    """
    print("\n" + "═" * 60)
    print("EVALUATION — EfficientNet Binary (COUGHVID: Healthy vs Symptomatic)")
    print("═" * 60)

    test_loader = DataLoader(
        CoughvidSpecDataset(df_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model.eval()
    all_proba, all_true = [], []

    with torch.no_grad():
        for batch_mel, batch_lbl in test_loader:
            batch_mel = batch_mel.to(DEVICE, non_blocking=True)
            logits    = model(batch_mel)
            proba     = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
            all_true.extend(batch_lbl.numpy())

    proba  = np.concatenate(all_proba, axis=0)
    y_true = np.array(all_true)
    y_pred = np.argmax(proba, axis=1)

    _plot_confusion_matrix(y_true, y_pred, COUGHVID_CLASSES, 'coughvid_efficientnet')
    _print_and_save_report(y_true, y_pred, COUGHVID_CLASSES, 'coughvid_efficientnet')
    return _metrics_summary(y_true, y_pred, proba, COUGHVID_CLASSES, 'coughvid_efficientnet')


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison_table(metrics_list: list) -> None:
    df = pd.DataFrame(metrics_list)
    df.columns = ['Model', 'Accuracy', 'F1', 'Recall', 'Precision', 'AUROC']
    path = os.path.join(OUTPUTS_DIR, "model_comparison_table.csv")
    df.to_csv(path, index=False)
    print(f"\nComparison table → {path}")
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Main — full evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from training import train_xgboost, train_multitask_efficientnet, train_coughvid_efficientnet

    # Usage:
    #   python evaluation.py            — evaluate all available models (default)
    #   python evaluation.py xgboost    — evaluate XGBoost only
    #   python evaluation.py cough      — evaluate LightCoughCNN only
    #   python evaluation.py multitask  — evaluate MultiTaskEfficientNet only

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    metrics_all = []

    # ── XGBoost ───────────────────────────────────────────────────
    xgb_path = os.path.join(SAVED_MODELS_DIR, "xgboost_coughvid.pkl")
    if mode in ("all", "xgboost") and os.path.exists(xgb_path):
        print("[evaluation] Loading saved XGBoost model ...")
        with open(xgb_path, 'rb') as f:
            data = pickle.load(f)

        xgb_model     = data['model']
        selected_idx  = data.get('selected_idx', None)
        xgb_threshold = data.get('threshold', 0.5)

        df_cv      = pd.read_csv(COUGHVID_LABELS_CSV)
        meta_cols  = ['age_norm', 'gender_enc', 'fever_muscle_pain_enc',
                       'resp_cond_enc', 'cough_score',
                       'dyspnea_enc', 'wheezing_enc', 'congestion_enc']

        n_mfcc = 13
        audio_cols = (
            [f'mfcc_{s}_{i}' for s in ['mean','std','max','min'] for i in range(n_mfcc)] +
            [f'delta_{s}_{i}' for s in ['mean','std','max','min'] for i in range(n_mfcc)] +
            [f'delta2_{s}_{i}' for s in ['mean','std','max','min'] for i in range(n_mfcc)] +
            [f'spec_centroid_{s}' for s in ['mean','std','max','min']] +
            [f'spec_bandwidth_{s}' for s in ['mean','std','max','min']] +
            [f'spec_contrast_{s}_{i}' for s in ['mean','std','max','min'] for i in range(7)] +
            [f'spec_rolloff_{s}' for s in ['mean','std','max','min']] +
            [f'spec_flatness_{s}' for s in ['mean','std','max','min']] +
            [f'zcr_{s}' for s in ['mean','std','max','min']] +
            [f'rms_{s}' for s in ['mean','std','max','min']] +
            [f'chroma_{s}_{i}' for s in ['mean','std','max','min'] for i in range(12)]
        )

        feat_cols = [c for c in meta_cols + audio_cols if c in df_cv.columns]
        X = df_cv[feat_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df_cv['label'].values.astype(np.int32)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        # Apply feature selection if v2
        if selected_idx is not None:
            X_test = X_test[:, selected_idx]
        metrics_xgb = evaluate_xgboost(xgb_model, X_test, y_test, threshold=xgb_threshold)
        metrics_all.append(metrics_xgb)
    elif mode in ("all", "xgboost"):
        print("[evaluation] XGBoost not found -- training ...")
        xgb_model, X_test, y_test, _ = train_xgboost()
        metrics_xgb = evaluate_xgboost(xgb_model, X_test, y_test)
        metrics_all.append(metrics_xgb)

    # ── COUGHVID EfficientNet ──────────────────────────────────────
    if mode in ("all", "cough"):
        cough_path = os.path.join(SAVED_MODELS_DIR, "coughvid_efficientnet.pt")
        if os.path.exists(cough_path):
            print("\n[evaluation] Loading saved COUGHVID EfficientNet ...")
            cough_model = build_light_cough_cnn()
            cough_model = load_checkpoint(cough_model, cough_path)
            df_cough    = pd.read_csv(COUGHVID_SPEC_LABELS_CSV)
            df_cough    = df_cough[df_cough['spec_path'].apply(os.path.exists)].reset_index(drop=True)
            y_cough     = df_cough['label'].values
            _, idx_cough_test = train_test_split(
                np.arange(len(df_cough)), test_size=0.15, stratify=y_cough, random_state=42
            )
            df_cough_test = df_cough.iloc[idx_cough_test].reset_index(drop=True)
        else:
            print("\n[evaluation] COUGHVID EfficientNet not found -- training ...")
            cough_model, df_cough_test = train_coughvid_efficientnet()
        metrics_cough_eff = evaluate_coughvid_efficientnet(cough_model, df_cough_test)
        metrics_all.append(metrics_cough_eff)

    # ── MultiTask EfficientNet ─────────────────────────────────────
    if mode in ("all", "multitask"):
        mt_path = os.path.join(SAVED_MODELS_DIR, "multitask_efficientnet.pt")
        if os.path.exists(mt_path):
            print("\n[evaluation] Loading saved MultiTask EfficientNet ...")
            mt_model = build_multitask_efficientnet(pretrained=False)
            mt_model = load_checkpoint(mt_model, mt_path)
            df_all = pd.read_csv(MULTITASK_LABELS_CSV)
            df_all = df_all[df_all['spec_path'].apply(os.path.exists)].reset_index(drop=True)
            y_all  = df_all['sound_label'].values
            idx    = np.arange(len(df_all))
            _, idx_test = train_test_split(idx, test_size=0.15, stratify=y_all, random_state=42)
            df_test = df_all.iloc[idx_test].reset_index(drop=True)
        else:
            print("\n[evaluation] MultiTask model not found -- training ...")
            mt_model, df_test = train_multitask_efficientnet()
        metrics_dis, metrics_snd = evaluate_multitask(mt_model, df_test)
        metrics_all.append(metrics_dis)
        metrics_all.append(metrics_snd)

    # ── Summary table ─────────────────────────────────────────────
    if metrics_all:
        save_comparison_table(metrics_all)