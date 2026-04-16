"""
agents/copd_agent.py — COPD binary specialist agent.

Loads a trained OPERA + MLP model and predicts COPD probability
from a cough or lung audio recording.

Standard output dict is consumed by fusion_agent and rule_engine.
"""

import os
import sys
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.mlp_classifier import BinaryMLPClassifier
from models.opera_encoder import OPERAEncoder

_DEFAULT_MODEL_PATH = './saved_models/copd_opera_mlp.pt'
logger = logging.getLogger(__name__)


class COPDAgent:
    """
    COPD specialist agent.

    Input : audio file path (cough or lung stethoscope recording)
    Output: standardised result dict consumed by the orchestrator
    """

    AGENT_NAME = 'COPD Agent'
    DISEASE    = 'COPD'

    def __init__(self,
                 model_path: str = _DEFAULT_MODEL_PATH,
                 device: str = 'cpu'):
        self.device = device

        # OPERA encoder (checkpoint auto-downloads from HuggingFace)
        self.encoder = OPERAEncoder(pretrain='operaCT')

        # Trained MLP classifier
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.threshold  = ckpt.get('threshold', 0.5)
        hidden_dims     = ckpt.get('hidden_dims', [256, 64])
        input_dim       = ckpt.get('input_dim', 768)

        self.classifier = BinaryMLPClassifier(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0
        ).to(device)
        self.classifier.load_state_dict(ckpt['model_state_dict'])
        self.classifier.eval()

        logger.info(
            "COPD model loaded",
            extra={'model_path': model_path, 'threshold': round(float(self.threshold), 4)},
        )

    def predict(self, audio_path: str) -> dict:
        """
        Predict COPD probability from audio.

        Returns
        -------
        dict with keys:
            agent, disease, detected, confidence, probability,
            severity_hint, threshold_used, error
        """
        try:
            embedding = self.encoder.encode(audio_path)
            emb_t     = torch.tensor(embedding).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits   = self.classifier(emb_t)
                probs    = torch.softmax(logits, dim=1)
                copd_prob = probs[0, 1].item()

            detected = copd_prob >= self.threshold

            if copd_prob >= 0.80:
                severity_hint = 'HIGH'
            elif copd_prob >= 0.60:
                severity_hint = 'MODERATE'
            else:
                severity_hint = 'LOW'

            return {
                'agent':          self.AGENT_NAME,
                'disease':        self.DISEASE,
                'detected':       detected,
                'confidence':     round(copd_prob, 4),
                'probability':    round(copd_prob, 4),
                'severity_hint':  severity_hint,
                'threshold_used': self.threshold,
                'error':          None,
            }

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            logger.error("COPD inference failed", extra={'error': str(e)})
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
        logger.error("Usage: python agents/copd_agent.py <audio_path>")
        sys.exit(1)
    agent  = COPDAgent()
    result = agent.predict(sys.argv[1])
    for k, v in result.items():
        logger.info("Prediction field", extra={'field': k, 'value': v})
