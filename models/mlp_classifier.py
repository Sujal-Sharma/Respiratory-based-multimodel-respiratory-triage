"""
models/mlp_classifier.py — MLP heads for binary disease classification
and 4-class lung sound classification on top of OPERA embeddings.

Both COPD and Pneumonia agents share the BinaryMLPClassifier architecture,
trained on different disease-specific datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMLPClassifier(nn.Module):
    """
    Binary disease classifier on OPERA 512-dim embeddings.
    Input : (B, 512) embedding tensor
    Output: (B, 2) logits — class 0 = Negative, class 1 = Positive
    """

    def __init__(self, input_dim: int = 512,
                 hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SoundMLPClassifier(nn.Module):
    """
    4-class lung sound classifier on OPERA 512-dim embeddings.
    Classes: 0=Normal, 1=Crackle, 2=Wheeze, 3=Both
    Input : (B, 512)
    Output: (B, 4) logits
    """

    def __init__(self, input_dim: int = 512,
                 hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 4))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced medical classification.
    Downweights easy examples, focuses training on hard/minority cases.

    alpha=0.25, gamma=2.0 are standard for medical binary classification.
    Reference: Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(logits, targets, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
