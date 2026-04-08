"""
agents/symptom_agent.py — CAT-style validated symptom scoring agent.

Computes a clinically validated Symptom Severity Index (0-1) using
a composite of CAT (COPD Assessment Test) items + mMRC Dyspnea Scale
+ BTS/GOLD clinical heuristics.

The composite score is designed for longitudinal tracking — it changes
meaningfully as the patient's condition improves or worsens.

Citations:
  - Jones et al. (2009) ERJ — CAT development and validation
  - Kon et al. (2014) Thorax — CAT MCID = 2 points (maps to ~0.05 on 0-1 scale)
  - GOLD 2024 — COPD diagnosis and management
  - BTS 2023 — Community-acquired pneumonia guidelines
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class SymptomAgent:
    AGENT_NAME = 'Symptom Agent'

    # MCID (Minimum Clinically Important Difference) = 0.05 on 0-1 scale
    # Corresponds to CAT MCID of 2 points on 0-40 scale
    MCID = 0.05

    def compute_symptom_index(self,
                               dyspnea_level: int,
                               cough_severity: float,
                               chest_tightness: int,
                               sleep_quality: int,
                               energy_level: int,
                               sputum: int,
                               fever: bool,
                               wheeze: bool) -> float:
        """
        Compute CAT-style Symptom Severity Index (0-1).

        Parameters (all required for longitudinal tracking):
          dyspnea_level   : 0-4 (mMRC scale: 0=none, 4=too breathless to leave house)
          cough_severity  : 0.0-1.0
          chest_tightness : 0-4 (CAT item 4: 0=none, 4=very tight)
          sleep_quality   : 0-4 (CAT item 7: 0=good sleep, 4=very poor)
          energy_level    : 0-4 (CAT item 8: 0=lots of energy, 4=no energy)
          sputum          : 0-3 (0=none, 1=clear, 2=coloured, 3=thick/dark)
          fever           : bool
          wheeze          : bool
        """
        # Normalize each item to [0,1]
        d   = dyspnea_level  / 4.0
        c   = float(cough_severity)            # already 0-1
        ct  = chest_tightness / 4.0
        sq  = sleep_quality  / 4.0
        el  = energy_level   / 4.0
        sp  = sputum         / 3.0
        fv  = 1.0 if fever  else 0.0
        wh  = 1.0 if wheeze else 0.0

        # Weighted composite (weights from clinical importance, sum=1.0)
        score = (
            d  * 0.25 +   # dyspnea — strongest indicator (mMRC validated)
            c  * 0.20 +   # cough severity
            ct * 0.15 +   # chest tightness
            fv * 0.12 +   # fever (acute infection marker)
            sq * 0.10 +   # sleep quality (CAT item 7)
            el * 0.08 +   # energy level (CAT item 8)
            sp * 0.06 +   # sputum character
            wh * 0.04     # wheeze
        )
        return round(min(score, 1.0), 4)

    def predict(self,
                age: float,
                gender: str,
                fever_muscle_pain: bool,
                dyspnea: bool,
                wheezing: bool,
                congestion: bool,
                resp_condition: bool,
                cough_severity: float,
                # CAT-style extended fields (optional, default to basic mode)
                dyspnea_level: int = -1,
                chest_tightness: int = 0,
                sleep_quality: int = 0,
                energy_level: int = 0,
                sputum: int = 0) -> dict:
        """
        Full symptom analysis with CAT-style scoring.

        If dyspnea_level is provided (>=0), uses full CAT-style index.
        Otherwise falls back to basic binary symptom scoring for backward compat.
        """
        try:
            # ── CAT-style symptom index ───────────────────────────────────
            if dyspnea_level >= 0:
                # Full CAT mode — use mMRC dyspnea level
                dyspnea_lvl = dyspnea_level
            else:
                # Basic mode — map boolean dyspnea to mMRC level
                dyspnea_lvl = 2 if dyspnea else 0

            symptom_index = self.compute_symptom_index(
                dyspnea_level   = dyspnea_lvl,
                cough_severity  = float(cough_severity) / 10.0
                                  if cough_severity > 1.0 else float(cough_severity),
                chest_tightness = chest_tightness,
                sleep_quality   = sleep_quality,
                energy_level    = energy_level,
                sputum          = sputum,
                fever           = fever_muscle_pain,
                wheeze          = wheezing,
            )

            # Age risk adjustment (GOLD 2024: age >= 40 = COPD screening warranted)
            age_factor = 0.0
            if age >= 65:   age_factor = 0.08
            elif age >= 50: age_factor = 0.05
            elif age >= 40: age_factor = 0.02
            symptomatic_prob = round(min(symptom_index + age_factor, 1.0), 4)

            # ── COPD hints (GOLD 2024) ────────────────────────────────────
            copd_factor = 0.0
            if resp_condition:    copd_factor += 0.40
            if dyspnea_lvl >= 2:  copd_factor += 0.25
            if wheezing:          copd_factor += 0.15
            if age >= 40:         copd_factor += 0.10
            if age >= 65:         copd_factor += 0.10
            copd_hint = round(min(symptomatic_prob * copd_factor, 1.0), 4)

            # ── Pneumonia hints (BTS 2023) ────────────────────────────────
            pneu_factor = 0.0
            if fever_muscle_pain: pneu_factor += 0.45
            if dyspnea_lvl >= 2:  pneu_factor += 0.25
            cough_norm = float(cough_severity) / 10.0 \
                if cough_severity > 1.0 else float(cough_severity)
            if cough_norm >= 0.7: pneu_factor += 0.20
            if congestion:        pneu_factor += 0.10
            pneu_hint = round(min(symptomatic_prob * pneu_factor, 1.0), 4)

            return {
                'agent':                      self.AGENT_NAME,
                'symptom_index':              symptom_index,
                'symptomatic_probability':    symptomatic_prob,
                'copd_probability_hint':      copd_hint,
                'pneumonia_probability_hint': pneu_hint,
                'detected':                   symptomatic_prob >= 0.35,
                'confidence':                 symptomatic_prob,
                'mcid':                       self.MCID,
                'error':                      None,
            }

        except Exception as e:
            return {
                'agent':                      self.AGENT_NAME,
                'symptom_index':              0.0,
                'symptomatic_probability':    0.0,
                'copd_probability_hint':      0.0,
                'pneumonia_probability_hint': 0.0,
                'detected':                   False,
                'confidence':                 0.0,
                'mcid':                       self.MCID,
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
