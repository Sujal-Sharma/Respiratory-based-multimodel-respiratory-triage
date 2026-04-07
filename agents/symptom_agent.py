"""
agents/symptom_agent.py — Clinical heuristic symptom agent.

No ML model required. Derives symptomatic risk and disease-specific
probability hints from patient metadata using published clinical
heuristics (GOLD 2024, BTS 2023 guidelines).

Outputs:
  copd_probability_hint      — supports rule engine COPD scoring
  pneumonia_probability_hint — supports rule engine Pneumonia scoring
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class SymptomAgent:
    """
    Symptom-based risk agent using clinical heuristics.

    Outputs symptomatic risk score and disease-specific probability
    hints based on patient metadata.
    """

    AGENT_NAME = 'Symptom Agent'

    def predict(self,
                age: float,
                gender: str,
                fever_muscle_pain: bool,
                dyspnea: bool,
                wheezing: bool,
                congestion: bool,
                resp_condition: bool,
                cough_severity: float) -> dict:
        """
        Estimate symptomatic risk from patient metadata.

        Parameters
        ----------
        cough_severity : 0-10 scale

        Returns
        -------
        dict with keys:
            agent, symptomatic_probability, copd_probability_hint,
            pneumonia_probability_hint, detected, confidence, error
        """
        try:
            # ── Symptomatic risk score (0–1) ──────────────────────────────
            score = 0.0
            score += min(cough_severity / 10.0, 1.0) * 0.25   # cough severity
            score += 0.20 if dyspnea          else 0.0
            score += 0.15 if wheezing         else 0.0
            score += 0.15 if fever_muscle_pain else 0.0
            score += 0.10 if resp_condition    else 0.0
            score += 0.05 if congestion        else 0.0
            # Age risk factor: older patients have higher baseline risk
            if age >= 65:
                score = min(score + 0.10, 1.0)
            elif age >= 50:
                score = min(score + 0.05, 1.0)

            symptomatic_prob = round(min(score, 1.0), 4)

            # ── COPD hints (GOLD 2024) ────────────────────────────────────
            # Key indicators: chronic respiratory condition + dyspnea + age
            copd_factor = 0.0
            if resp_condition:  copd_factor += 0.40
            if dyspnea:         copd_factor += 0.25
            if wheezing:        copd_factor += 0.15
            if age >= 50:       copd_factor += 0.10
            if age >= 65:       copd_factor += 0.10
            copd_hint = round(min(symptomatic_prob * copd_factor, 1.0), 4)

            # ── Pneumonia hints (BTS 2023) ────────────────────────────────
            # Key indicators: fever + acute dyspnea + high cough severity
            pneu_factor = 0.0
            if fever_muscle_pain: pneu_factor += 0.45
            if dyspnea:           pneu_factor += 0.25
            if cough_severity >= 7: pneu_factor += 0.20
            if congestion:        pneu_factor += 0.10
            pneu_hint = round(min(symptomatic_prob * pneu_factor, 1.0), 4)

            return {
                'agent':                      self.AGENT_NAME,
                'symptomatic_probability':    symptomatic_prob,
                'copd_probability_hint':      copd_hint,
                'pneumonia_probability_hint': pneu_hint,
                'detected':                   symptomatic_prob >= 0.4,
                'confidence':                 symptomatic_prob,
                'error':                      None,
            }

        except Exception as e:
            return {
                'agent':                      self.AGENT_NAME,
                'symptomatic_probability':    0.0,
                'copd_probability_hint':      0.0,
                'pneumonia_probability_hint': 0.0,
                'detected':                   False,
                'confidence':                 0.0,
                'error':                      str(e),
            }


def predict_symptom(age, gender, fever_muscle_pain, respiratory_condition,
                    cough_detected=0.5, dyspnea=False, wheezing=False,
                    congestion=False, audio_path=None) -> dict:
    """Backward-compatible convenience function."""
    return SymptomAgent().predict(
        age=age, gender=gender,
        fever_muscle_pain=fever_muscle_pain,
        dyspnea=dyspnea, wheezing=wheezing,
        congestion=congestion,
        resp_condition=respiratory_condition,
        cough_severity=float(cough_detected) * 10,
    )


if __name__ == '__main__':
    agent  = SymptomAgent()
    result = agent.predict(
        age=58, gender='male',
        fever_muscle_pain=False, dyspnea=True,
        wheezing=True, congestion=False,
        resp_condition=True, cough_severity=7,
    )
    print("[SymptomAgent] Example prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
