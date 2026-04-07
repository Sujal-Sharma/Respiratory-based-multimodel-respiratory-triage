"""
agents/pneumonia_agent.py — Pneumonia binary specialist agent.

Identical architecture to COPDAgent — different trained weights.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import BinaryMLPClassifier
from models.opera_encoder import OPERAEncoder

_DEFAULT_MODEL_PATH = './saved_models/pneumonia_opera_mlp.pt'


class PneumoniaAgent:
    """
    Pneumonia specialist agent.

    Input : audio file path (cough or lung recording)
    Output: standardised result dict
    """

    AGENT_NAME = 'Pneumonia Agent'
    DISEASE    = 'Pneumonia'

    def __init__(self,
                 model_path: str = _DEFAULT_MODEL_PATH,
                 device: str = 'cuda'):
        self.device = device

        self.encoder = OPERAEncoder(pretrain='operaCT')

        ckpt           = torch.load(model_path, map_location=device, weights_only=False)
        self.threshold = ckpt.get('threshold', 0.5)
        hidden_dims    = ckpt.get('hidden_dims', [256, 64])
        input_dim      = ckpt.get('input_dim', 768)

        self.classifier = BinaryMLPClassifier(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0
        ).to(device)
        self.classifier.load_state_dict(ckpt['model_state_dict'])
        self.classifier.eval()

        print(f"[PneumoniaAgent] Loaded {model_path} | threshold={self.threshold:.3f}")

    def predict(self, audio_path: str) -> dict:
        """
        Predict Pneumonia probability from audio.

        Returns same standardised dict structure as COPDAgent.
        """
        try:
            embedding = self.encoder.encode(audio_path)
            emb_t     = torch.tensor(embedding).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits    = self.classifier(emb_t)
                probs     = torch.softmax(logits, dim=1)
                pneu_prob = probs[0, 1].item()

            detected = pneu_prob >= self.threshold

            if pneu_prob >= 0.80:
                severity_hint = 'HIGH'
            elif pneu_prob >= 0.60:
                severity_hint = 'MODERATE'
            else:
                severity_hint = 'LOW'

            return {
                'agent':          self.AGENT_NAME,
                'disease':        self.DISEASE,
                'detected':       detected,
                'confidence':     round(pneu_prob, 4),
                'probability':    round(pneu_prob, 4),
                'severity_hint':  severity_hint,
                'threshold_used': self.threshold,
                'error':          None,
            }

        except Exception as e:
            return {
                'agent':          self.AGENT_NAME,
                'disease':        self.DISEASE,
                'detected':       False,
                'confidence':     0.0,
                'probability':    0.0,
                'severity_hint':  'LOW',
                'threshold_used': self.threshold,
                'error':          str(e),
            }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agents/pneumonia_agent.py <audio_path>")
        sys.exit(1)
    agent  = PneumoniaAgent()
    result = agent.predict(sys.argv[1])
    for k, v in result.items():
        print(f"  {k}: {v}")
