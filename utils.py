"""
utils.py — Helper functions used across the project.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from config import DEVICE, OUTPUTS_DIR


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[utils] Random seed set to {seed}")


def get_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Returns a float32 tensor on DEVICE.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"[utils] Class weights: {dict(zip(classes, weights.round(3)))}")
    return weight_tensor


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_accs: list,
    save_path: str = None
) -> None:
    """
    Plot training/validation loss and validation accuracy side-by-side.
    Saves to outputs/training_loss_efficientnet.png.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "training_loss_efficientnet.png")

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   linewidth=2, markersize=4)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, val_accs, 'g-o', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=0.88, color='r', linestyle='--', alpha=0.7, label='Target (88%)')
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[utils] Training curves saved → {save_path}")


def print_gpu_stats() -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        free_mb   = 4096 - torch.cuda.memory_reserved() / 1e6
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  Alloc : {allocated:.2f} GB")
        print(f"  Reserv: {reserved:.2f} GB")
        print(f"  Free  : {free_mb:.0f} MB")
    else:
        print("  [utils] CUDA not available — running on CPU")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str
) -> None:
    """Save model checkpoint with training metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'val_acc':    val_acc,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"[utils] Checkpoint saved -> {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load model weights from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch   = checkpoint.get('epoch', '?')
    val_acc = checkpoint.get('val_acc', '?')
    print(f"[utils] Checkpoint loaded <- {path}  (epoch={epoch}, val_acc={val_acc})")
    return model
