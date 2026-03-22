"""
training.py — Model training for XGBoost (COUGHVID), LightCoughCNN (COUGHVID),
              and MultiTaskEfficientNet (lung).

Run order:
  1. train_xgboost()               — tabular, GPU-accelerated XGBoost
  2. train_coughvid_efficientnet()  — LightCoughCNN, phone cough audio
  3. train_multitask_efficientnet() — dual-head EfficientNet, lung sounds
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

from config import (
    DEVICE, BATCH_SIZE, WEIGHT_DECAY, PATIENCE,
    COUGHVID_LABELS_CSV, COUGHVID_SPEC_LABELS_CSV, COUGHVID_CLASSES,
    MULTITASK_LABELS_CSV, SAVED_MODELS_DIR, OUTPUTS_DIR,
    NUM_DISEASE_CLASSES, NUM_SOUND_CLASSES,
)
from models.cnn_model import (
    build_multitask_efficientnet,
    build_light_cough_cnn,
    freeze_entire_backbone,
    unfreeze_last_n_blocks,
    unfreeze_all,
)
from models.xgboost_model import build_xgboost
from preprocessing import preprocess_coughvid, preprocess_coughvid_spectrograms
from utils import set_seed, plot_training_curves, print_gpu_stats, save_checkpoint

warnings.filterwarnings('ignore')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)



# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID — XGBoost Training
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost():
    """
    Train XGBoost on COUGHVID metadata + MFCC features (GPU-accelerated).

    Returns
    -------
    model     : trained XGBClassifier
    X_test    : np.ndarray
    y_test    : np.ndarray
    label_map : dict {class_name: int}
    """
    print("\n" + "═" * 60)
    print("TRAINING — XGBoost on COUGHVID")
    print("═" * 60)
    set_seed(42)

    META_COLS = [
        'age_norm', 'gender_enc', 'fever_muscle_pain_enc',
        'resp_cond_enc', 'cough_score',
        'dyspnea_enc', 'wheezing_enc', 'congestion_enc'
    ]
    MFCC_COLS = (
        [f'mfcc_mean_{i}'  for i in range(40)] +
        [f'mfcc_std_{i}'   for i in range(40)] +
        [f'delta_mean_{i}' for i in range(40)] +
        [f'delta_std_{i}'  for i in range(40)]
    )
    FEATURE_COLS = META_COLS + MFCC_COLS   # 8 metadata + 160 MFCC = 168 total

    if os.path.exists(COUGHVID_LABELS_CSV):
        print(f"Loading preprocessed features from {COUGHVID_LABELS_CSV}")
        df   = pd.read_csv(COUGHVID_LABELS_CSV)
        cols = [c for c in FEATURE_COLS if c in df.columns]
        X    = df[cols].values.astype(np.float32)
        y    = df['label'].values.astype(np.int32)
        label_map = {cls: i for i, cls in enumerate(COUGHVID_CLASSES)}
        print(f"  Features loaded: {X.shape}  ({len(cols)} cols found)")
    else:
        X, y, label_map = preprocess_coughvid()

    print(f"Feature matrix: {X.shape} | Labels: {np.unique(y, return_counts=True)}")

    # 70 / 15 / 15 stratified split
    X_tv,   X_test,  y_tv,   y_test  = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val,  y_train, y_val  = train_test_split(X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42)
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    model = build_xgboost(num_classes=len(label_map))
    print("\nTraining XGBoost (CUDA) …")

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    save_path = os.path.join(SAVED_MODELS_DIR, "xgboost_coughvid.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved → {save_path}")

    val_acc = np.mean(model.predict(X_val) == y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    print("═" * 60)
    return model, X_test, y_test, label_map


# ══════════════════════════════════════════════════════════════════════════════
# MultiTask EfficientNet — Dataset
# ══════════════════════════════════════════════════════════════════════════════

class MultiTaskDataset(Dataset):
    """
    PyTorch Dataset for multi-task lung classification.

    Returns
    -------
    mel_tensor    : Tensor (1, N_MELS, T) float32
    disease_label : int  (-1 = masked for HF Lung samples)
    sound_label   : int  (always valid)
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _time_mask(self, mel: np.ndarray, frac: float = 0.15) -> np.ndarray:
        T     = mel.shape[1]
        width = max(1, int(T * frac))
        start = np.random.randint(0, max(1, T - width))
        mel   = mel.copy()
        mel[:, start:start + width] = 0.0
        return mel

    def _freq_mask(self, mel: np.ndarray, frac: float = 0.15) -> np.ndarray:
        F     = mel.shape[0]
        width = max(1, int(F * frac))
        start = np.random.randint(0, max(1, F - width))
        mel   = mel.copy()
        mel[start:start + width, :] = 0.0
        return mel

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        mel   = np.load(row['spec_path']).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                mel = self._time_mask(mel)
            if np.random.rand() < 0.5:
                mel = self._freq_mask(mel)
            if np.random.rand() < 0.4:
                mel = np.clip(
                    mel + np.random.normal(0, 0.005, mel.shape).astype(np.float32),
                    0.0, 1.0
                )
            if np.random.rand() < 0.3:
                mel = np.clip(mel * np.random.uniform(0.8, 1.2), 0.0, 1.0)

        mel_tensor    = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        disease_label = int(row['disease_label'])
        sound_label   = int(row['sound_label'])
        return mel_tensor, disease_label, sound_label


# ══════════════════════════════════════════════════════════════════════════════
# Data split helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_multitask_splits():
    """
    Load multitask_labels.csv and return stratified train/val/test splits.
    Stratify on sound_label (always valid for all samples).
    """
    df = pd.read_csv(MULTITASK_LABELS_CSV)
    df = df[df['spec_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"[training] Valid spectrograms: {len(df):,}")

    y = df['sound_label'].values
    idx = np.arange(len(df))

    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    y_tv = y[idx_tv]
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)

    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")
    return df_train, df_val, df_test


def _make_disease_sampler(df_dis: pd.DataFrame) -> WeightedRandomSampler:
    """Balance the disease-only loader across all 5 disease classes."""
    dis_labels  = df_dis['disease_label'].values
    dis_counts  = np.bincount(dis_labels, minlength=NUM_DISEASE_CLASSES)
    weights     = 1.0 / (dis_counts[dis_labels] + 1e-6)
    weights     = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _build_class_weights(labels: np.ndarray, n_classes: int,
                          max_weight: float = 20.0) -> torch.Tensor:
    """
    Inverse-frequency class weights, clipped to max_weight to prevent
    explosion when a class has very few samples.
    """
    counts  = np.bincount(labels, minlength=n_classes).astype(float)
    w       = 1.0 / (counts + 1e-6)
    w       = np.clip(w, 0, max_weight)
    w       = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# MultiTask EfficientNet Training
# ══════════════════════════════════════════════════════════════════════════════

def train_multitask_efficientnet():
    """
    Train MultiTaskEfficientNet-B0 on ICBHI + KAUH + HF Lung V1.
    """
    print("\n" + "═" * 60)
    print("TRAINING — MultiTask EfficientNet-B0 (ICBHI + KAUH + HF Lung V1)")
    print("═" * 60)
    set_seed(42)

    TOTAL_EPOCHS   = 80
    PATIENCE_LIMIT = 15
    LR_BACKBONE    = 3e-5
    LR_HEADS       = 3e-4

    df_train, df_val, df_test = _get_multitask_splits()

    train_ds = MultiTaskDataset(df_train, augment=True)
    val_ds   = MultiTaskDataset(df_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    df_disease = df_train[df_train['disease_label'] >= 0].reset_index(drop=True)
    dis_loader = DataLoader(
        MultiTaskDataset(df_disease, augment=True),
        batch_size=BATCH_SIZE // 2,
        sampler=_make_disease_sampler(df_disease),
        num_workers=0, pin_memory=True, drop_last=True
    )
    dis_iter = iter(dis_loader)

    print(f"  Main  loader : {len(df_train):,} samples")
    print(f"  Disease loader: {len(df_disease):,} samples")

    snd_weights = _build_class_weights(
        df_train['sound_label'].values, NUM_SOUND_CLASSES, max_weight=5.0
    )
    print(f"  Sound class weights: {snd_weights.cpu().numpy().round(3)}")

    sound_criterion   = nn.CrossEntropyLoss(label_smoothing=0.1, weight=snd_weights)
    disease_criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-1)

    model = build_multitask_efficientnet(pretrained=True)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(),      'lr': LR_BACKBONE},
        {'params': model.shared.parameters(),        'lr': LR_HEADS},
        {'params': model.disease_head.parameters(),  'lr': LR_HEADS},
        {'params': model.sound_head.parameters(),    'lr': LR_HEADS},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS
    )
    scaler = GradScaler('cuda')

    save_path    = os.path.join(SAVED_MODELS_DIR, "multitask_efficientnet.pt")
    best_val_acc = 0.0
    patience_ctr = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\nDevice: {DEVICE} | Epochs: {TOTAL_EPOCHS} | Patience: {PATIENCE_LIMIT}")
    print_gpu_stats()

    for epoch in range(1, TOTAL_EPOCHS + 1):

        model.train()
        running_loss = 0.0
        n_batches    = 0

        for batch_mel, _, batch_snd in train_loader:
            batch_mel = batch_mel.to(DEVICE, non_blocking=True)
            batch_snd = batch_snd.to(DEVICE, non_blocking=True)

            try:
                d_mel, d_dis, _ = next(dis_iter)
            except StopIteration:
                dis_iter = iter(dis_loader)
                d_mel, d_dis, _ = next(dis_iter)
            d_mel = d_mel.to(DEVICE, non_blocking=True)
            d_dis = d_dis.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda'):
                _, snd_out   = model(batch_mel)
                sound_loss   = sound_criterion(snd_out.float(), batch_snd)
                dis_out, _   = model(d_mel)
                disease_loss = disease_criterion(dis_out.float(), d_dis)
                loss = sound_loss + 2.0 * disease_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches    += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        snd_correct = snd_total = 0
        dis_correct = dis_total = 0

        with torch.no_grad():
            for batch_mel, batch_dis, batch_snd in val_loader:
                batch_mel = batch_mel.to(DEVICE, non_blocking=True)
                batch_dis = batch_dis.to(DEVICE, non_blocking=True)
                batch_snd = batch_snd.to(DEVICE, non_blocking=True)

                with autocast('cuda'):
                    dis_out, snd_out = model(batch_mel)
                    s_loss = sound_criterion(snd_out.float(), batch_snd)
                    d_loss = disease_criterion(dis_out.float(), batch_dis)

                val_loss_sum += (s_loss + 2.0 * d_loss).item()

                snd_correct += (snd_out.argmax(1) == batch_snd).sum().item()
                snd_total   += batch_snd.size(0)

                mask = batch_dis >= 0
                if mask.any():
                    dis_correct += (dis_out.argmax(1)[mask] == batch_dis[mask]).sum().item()
                    dis_total   += mask.sum().item()

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        snd_acc = snd_correct / max(snd_total, 1)
        dis_acc = dis_correct / max(dis_total, 1)
        val_acc = 0.6 * snd_acc + 0.4 * dis_acc

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        scheduler.step()

        mem_str = (f" | GPU {torch.cuda.memory_allocated()/1e9:.2f}GB"
                   if DEVICE.type == 'cuda' else "")
        print(f"Ep[{epoch:02d}/{TOTAL_EPOCHS}] "
              f"Loss {avg_train_loss:.4f}→{avg_val_loss:.4f} | "
              f"Sound {snd_acc:.4f} | Disease {dis_acc:.4f}{mem_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch, val_acc, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE_LIMIT:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best={best_val_acc:.4f})")
                break

        torch.cuda.empty_cache()

    plot_training_curves(train_losses, val_losses, val_accs,
                         save_path=os.path.join(OUTPUTS_DIR, "training_curves_multitask.png"))
    print(f"\nBest combined val accuracy: {best_val_acc:.4f}")
    print(f"Model saved → {save_path}")
    print("═" * 60)
    return model, df_test


# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID EfficientNet — Dataset
# ══════════════════════════════════════════════════════════════════════════════

class CoughvidSpecDataset(Dataset):
    """
    Dataset for binary COUGHVID cough classification using LightCoughCNN.

    Input: mel spectrograms (128, 173) saved as .npy files.
    Output: (1, 128, 173) float32 tensor + label int (0=Healthy, 1=Symptomatic)

    Augmentations (training only):
      - SpecAugment: time mask (15%) + frequency mask (15%)
      - Amplitude jitter (±20%)
      - Gaussian noise (σ=0.005)
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _time_mask(self, mel: np.ndarray, frac: float = 0.15) -> np.ndarray:
        T     = mel.shape[1]
        width = max(1, int(T * frac))
        start = np.random.randint(0, max(1, T - width))
        mel   = mel.copy()
        mel[:, start:start + width] = 0.0
        return mel

    def _freq_mask(self, mel: np.ndarray, frac: float = 0.15) -> np.ndarray:
        F     = mel.shape[0]
        width = max(1, int(F * frac))
        start = np.random.randint(0, max(1, F - width))
        mel   = mel.copy()
        mel[start:start + width, :] = 0.0
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row['spec_path']).astype(np.float32)  # (128, T), [0,1]

        if self.augment:
            if np.random.rand() < 0.5:
                mel = self._time_mask(mel)
            if np.random.rand() < 0.5:
                mel = self._freq_mask(mel)
            if np.random.rand() < 0.3:
                mel = np.clip(mel * np.random.uniform(0.8, 1.2), 0.0, 1.0)
            if np.random.rand() < 0.3:
                mel = np.clip(
                    mel + np.random.normal(0, 0.005, mel.shape).astype(np.float32),
                    0.0, 1.0
                )

        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0), int(row['label'])


# ══════════════════════════════════════════════════════════════════════════════
# COUGHVID LightCoughCNN Training
# ══════════════════════════════════════════════════════════════════════════════

def train_coughvid_efficientnet():
    """
    Train LightCoughCNN on COUGHVID mel spectrograms.

    Binary task: Healthy (0) vs Symptomatic (1) — COVID-19 merged into Symptomatic.

    Architecture: LightCoughCNN (~500K params, 1-channel mel input)
      - Properly sized for ~1400 training samples
      - Trained from scratch (no ImageNet — proven not to transfer to spectrograms)
      - 3 conv blocks + AdaptiveAvgPool + 2 FC layers

    Preprocessing (no pre-emphasis — preserves low-freq cough characteristics):
      1. Load & resample to 22050 Hz
      2. Trim silence (top_db=20)
      3. Peak-normalise
      4. Pad/truncate to 4s
      5. Log-mel (n_fft=2048, hop=512, 128 mels, fmin=50, fmax=8000)
      6. Normalise to [0,1]

    Training: AdamW, LR=1e-3, CosineAnnealing, no label smoothing.
    Expected accuracy: 60–70% (COUGHVID audio-only with crowd-sourced labels)
    """
    print("\n" + "═" * 60)
    print("TRAINING — LightCoughCNN (1-ch mel, 128×173)")
    print("Binary: Healthy vs Symptomatic | From scratch")
    print("═" * 60)
    set_seed(42)

    TOTAL_EPOCHS   = 80
    PATIENCE_LIMIT = 20
    LR             = 1e-3

    # ── Data ───────────────────────────────────────────────────────
    if not os.path.exists(COUGHVID_SPEC_LABELS_CSV):
        print("  Spectrograms not found — running preprocessing …")
        preprocess_coughvid_spectrograms()

    df = pd.read_csv(COUGHVID_SPEC_LABELS_CSV)
    df = df[df['spec_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"  Valid spectrograms: {len(df):,}")

    y   = df['label'].values
    idx = np.arange(len(df))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=y[idx_tv], random_state=42
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # ── WeightedRandomSampler for class balance ─────────────────
    train_labels = df_train['label'].values
    class_counts = np.bincount(train_labels, minlength=2)
    sample_weights = 1.0 / (class_counts[train_labels] + 1e-6)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        CoughvidSpecDataset(df_train, augment=True),
        batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        CoughvidSpecDataset(df_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ── Class weights — mild Symptomatic boost for balanced recall ──
    cw = _build_class_weights(df_train['label'].values, n_classes=2, max_weight=5.0)
    cw[1] = cw[1] * 1.2   # mild boost — missing sick patients is worse than false alarm
    cw = cw / cw.sum() * 2  # re-normalise
    print(f"  Class weights: Healthy={cw[0]:.3f}  Symptomatic={cw[1]:.3f}")
    print(f"  Class counts : Healthy={class_counts[0]:,}  Symptomatic={class_counts[1]:,}")

    # No label smoothing — clean gradient signal for from-scratch training
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── Model — LightCoughCNN (~500K params, from scratch) ────────
    model = build_light_cough_cnn()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)  # 1e-4

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )

    save_path    = os.path.join(SAVED_MODELS_DIR, "coughvid_efficientnet.pt")
    best_val_acc = 0.0
    patience_ctr = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"\nDevice: {DEVICE} | Epochs: {TOTAL_EPOCHS} | Patience: {PATIENCE_LIMIT}")
    print(f"LightCoughCNN (~500K params) | LR={LR} | CosineAnnealingLR | WD={WEIGHT_DECAY}")
    print_gpu_stats()

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for batch_img, batch_lbl in train_loader:
            batch_img = batch_img.to(DEVICE, non_blocking=True)
            batch_lbl = batch_lbl.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(batch_img)
            loss   = criterion(logits, batch_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        correct = total = 0

        with torch.no_grad():
            for batch_img, batch_lbl in val_loader:
                batch_img = batch_img.to(DEVICE, non_blocking=True)
                batch_lbl = batch_lbl.to(DEVICE, non_blocking=True)

                logits = model(batch_img)
                loss   = criterion(logits, batch_lbl)

                val_loss_sum += loss.item()
                correct      += (logits.argmax(1) == batch_lbl).sum().item()
                total        += batch_lbl.size(0)

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc      = correct / max(total, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        scheduler.step()

        mem_str = (f" | GPU {torch.cuda.memory_allocated()/1e9:.2f}GB"
                   if DEVICE.type == 'cuda' else "")
        print(f"Ep[{epoch:02d}/{TOTAL_EPOCHS}] "
              f"Loss {avg_train_loss:.4f}→{avg_val_loss:.4f} | "
              f"Acc {val_acc:.4f}{mem_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch, val_acc, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE_LIMIT:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best={best_val_acc:.4f})")
                break

        torch.cuda.empty_cache()

    plot_training_curves(train_losses, val_losses, val_accs,
                         save_path=os.path.join(OUTPUTS_DIR,
                                                "training_curves_coughvid_efficientnet.png"))
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Model saved       → {save_path}")
    print("═" * 60)
    return model, df_test


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1a: XGBoost (COUGHVID — metadata + MFCC features, fast baseline)
    xgb_model, X_test, y_test, xgb_label_map = train_xgboost()

    # Step 1b: LightCoughCNN (COUGHVID — mel spectrograms, no pre-emphasis)
    cough_model, df_cough_test = train_coughvid_efficientnet()

    # Step 2: MultiTask EfficientNet (lung) — uncomment to retrain
    # mt_model, df_test = train_multitask_efficientnet()
