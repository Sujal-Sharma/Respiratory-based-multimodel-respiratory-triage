"""
models/cnn_model.py — Neural network architectures for respiratory audio classification.

Contains:
  MultiTaskEfficientNet — EfficientNet-B0 with two output heads (disease + sound)
  LightCoughCNN         — Lightweight 4-block CNN for COUGHVID binary classification
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import DEVICE, NUM_DISEASE_CLASSES, NUM_SOUND_CLASSES


class MultiTaskEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone shared across two classification heads.

    Input  : (batch, 1, N_MELS, T)  — single-channel mel spectrogram
    Outputs: (disease_logits, sound_logits)
               disease_logits : (batch, 5)  — Normal/COPD/Pneumonia/Asthma/Heart_Failure
               sound_logits   : (batch, 4)  — Normal/Crackle/Wheeze/Both

    Architecture:
      EfficientNet-B0 features (1-channel) → AdaptiveAvgPool → Flatten → (batch, 1280)
      Shared dense : Linear(1280→512) → ReLU → Dropout(0.4)
      Disease head : Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→5)
      Sound head   : Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→4)
    """

    def __init__(self,
                 num_disease_classes: int = NUM_DISEASE_CLASSES,
                 num_sound_classes:   int = NUM_SOUND_CLASSES,
                 pretrained:          bool = True):
        super().__init__()

        # ── Load EfficientNet-B0 backbone ──────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base    = efficientnet_b0(weights=weights)

        # Adapt first conv: 3 channels → 1 channel
        orig_conv = base.features[0][0]
        new_conv  = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )
        base.features[0][0] = new_conv

        # Backbone: features + avgpool + flatten → output dim 1280
        self.backbone = nn.Sequential(
            base.features,
            base.avgpool,     # AdaptiveAvgPool2d(1,1)
            nn.Flatten(),     # (batch, 1280)
        )

        # ── Shared dense ──────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        # ── Disease head ──────────────────────────────────────────
        self.disease_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_disease_classes),
        )

        # ── Sound head ────────────────────────────────────────────
        self.sound_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_sound_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor shape (batch, 1, N_MELS, T)

        Returns
        -------
        disease_logits : Tensor (batch, num_disease_classes)
        sound_logits   : Tensor (batch, num_sound_classes)
        """
        features     = self.backbone(x)          # (batch, 1280)
        shared       = self.shared(features)     # (batch, 512)
        disease_out  = self.disease_head(shared) # (batch, 5)
        sound_out    = self.sound_head(shared)   # (batch, 4)
        return disease_out, sound_out


def build_multitask_efficientnet(pretrained: bool = True) -> MultiTaskEfficientNet:
    """
    Build and return MultiTaskEfficientNet on DEVICE in float16.

    float16 is used to fit GTX 1650 4 GB VRAM.
    Mixed-precision autocast handles training stability.
    """
    model = MultiTaskEfficientNet(pretrained=pretrained)
    model = model.to(DEVICE)
    # NOTE: Do NOT call model.half() here.
    # Parameters stay float32; autocast() handles float16 during forward pass.
    # GradScaler requires float32 master params to unscale gradients correctly.

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cnn_model] MultiTaskEfficientNet: {trainable:,} / {total:,} params")
    return model


def freeze_entire_backbone(model: MultiTaskEfficientNet) -> None:
    """
    Freeze the ENTIRE EfficientNet backbone (all feature blocks + avgpool).
    Only shared dense and both heads remain trainable.
    Use in Phase 1 to prevent overfitting on small datasets.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[cnn_model] Full backbone frozen | Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


def unfreeze_last_n_blocks(model: MultiTaskEfficientNet, n: int = 2) -> None:
    """
    Unfreeze the last n feature blocks of the EfficientNet backbone.
    Used in Phase 2 for fine-tuning with a low learning rate.
    """
    features   = model.backbone[0]        # the Sequential of feature blocks
    n_blocks   = len(features)
    start_block = max(0, n_blocks - n)

    for i in range(start_block, n_blocks):
        for param in features[i].parameters():
            param.requires_grad = True
    # Also unfreeze avgpool and flatten (they have no params but included for clarity)
    for param in model.backbone[1].parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[cnn_model] Unfroze last {n} backbone blocks | "
          f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def unfreeze_all(model: MultiTaskEfficientNet) -> None:
    """Unfreeze all parameters for fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
    print("[cnn_model] All layers unfrozen for fine-tuning.")


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight Cough CNN — COUGHVID (properly sized for small dataset)
# ══════════════════════════════════════════════════════════════════════════════

class LightCoughCNN(nn.Module):
    """
    Lightweight 4-block CNN for COUGHVID binary classification.

    ~500K parameters — properly sized for ~1,400 training samples.
    Trained from scratch on mel-spectrograms (no ImageNet pretrain bias).
    BatchNorm in every block prevents NaN gradients.

    Input  : (batch, 1, 128, 94)
    Output : (batch, 2)  — [Healthy, Symptomatic] logits

    Block layout:
      Conv(1→32)  → BN → ReLU → Conv(32→32)  → BN → ReLU → MaxPool(2)
      Conv(32→64) → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool(2)
      Conv(64→128)→ BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool(2)
      AdaptiveAvgPool(1) → Flatten(128) → Linear(128→64) → Dropout(0.5) → Linear(64→2)
    """

    def __init__(self):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            _block(1,  32),  nn.MaxPool2d(2, 2),   # → (32, 64, 47)
            _block(32, 64),  nn.MaxPool2d(2, 2),   # → (64, 32, 23)
            _block(64, 128), nn.MaxPool2d(2, 2),   # → (128,16, 11)
            nn.AdaptiveAvgPool2d(1),                # → (128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_light_cough_cnn() -> LightCoughCNN:
    """Build and return LightCoughCNN on DEVICE."""
    model     = LightCoughCNN().to(DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[cnn_model] LightCoughCNN: {total:,} params (trained from scratch)")
    return model


if __name__ == "__main__":
    model = build_multitask_efficientnet(pretrained=False)
    freeze_entire_backbone(model)

    dummy = torch.zeros(2, 1, 128, 188, dtype=torch.float16).to(DEVICE)
    with torch.no_grad():
        d_out, s_out = model(dummy)
    print(f"[cnn_model] Disease head output: {d_out.shape}")  # (2, 5)
    print(f"[cnn_model] Sound head output  : {s_out.shape}")  # (2, 4)

    unfreeze_all(model)

    cough_model = build_light_cough_cnn()
    dummy_cough = torch.zeros(2, 1, 128, 173).to(DEVICE)
    with torch.no_grad():
        cough_out = cough_model(dummy_cough)
    print(f"[cnn_model] LightCoughCNN output: {cough_out.shape}")  # (2, 2)