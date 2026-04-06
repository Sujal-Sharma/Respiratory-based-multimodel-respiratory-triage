"""
agents/sound_agent.py — Lung sound classifier agent (Tier 2 only).

Classifies stethoscope audio into: Normal / Crackle / Wheeze.
"Both" class removed — merged into Crackle (see train_sound_3class.py).
Used only when a lung recording is provided (clinical setting).
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.opera_encoder import OPERAEncoder

_DEFAULT_MODEL_PATH = './saved_models/sound_opera_mlp_3class.pt'

SOUND_LABELS = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze'}


class SoundMLP3Class(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SoundAgent:
    """
    Lung sound classifier. Tier 2 only.
    Input : stethoscope audio file path
    Output: sound type classification dict
    """

    AGENT_NAME = 'Sound Agent'

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH, device: str = 'cuda'):
        self.device = device
        self.encoder = OPERAEncoder(pretrain='operaCT')

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        hidden_dims = ckpt.get('hidden_dims', [512, 256, 64])
        input_dim   = ckpt.get('input_dim', 768)

        self.classifier = SoundMLP3Class(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0
        ).to(device)
        self.classifier.load_state_dict(ckpt['model_state_dict'])
        self.classifier.eval()

        print(f"[SoundAgent] Loaded {model_path} (3-class: Normal/Crackle/Wheeze)")

    def predict(self, audio_path: str) -> dict:
        try:
            embedding = self.encoder.encode(audio_path)
            emb_t     = torch.tensor(embedding).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(emb_t)
                probs  = torch.softmax(logits, dim=1)[0]

            pred_class = probs.argmax().item()
            pred_label = SOUND_LABELS[pred_class]
            confidence = probs[pred_class].item()

            return {
                'agent':      self.AGENT_NAME,
                'sound_type': pred_label,
                'confidence': round(confidence, 4),
                'all_probabilities': {
                    SOUND_LABELS[i]: round(probs[i].item(), 4)
                    for i in range(3)
                },
                'error': None,
            }

        except Exception as e:
            return {
                'agent':      self.AGENT_NAME,
                'sound_type': 'Normal',
                'confidence': 0.0,
                'all_probabilities': {v: 0.0 for v in SOUND_LABELS.values()},
                'error': str(e),
            }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python agents/sound_agent.py <audio_path>")
        sys.exit(1)
    agent  = SoundAgent()
    result = agent.predict(sys.argv[1])
    for k, v in result.items():
        print(f"  {k}: {v}")
