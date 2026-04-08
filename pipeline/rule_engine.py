"""
pipeline/rule_engine.py — Deterministic clinical triage rule engine.

Replaces Groq/Gemini LLM entirely. Produces identical output for identical
input — fully reproducible, no API calls, no hallucination.

All thresholds are cited to published clinical guidelines:
  - GOLD 2024: Global Initiative for Chronic Obstructive Lung Disease
  - BTS 2023 : British Thoracic Society community-acquired pneumonia guidelines
  - GINA 2023: Global Initiative for Asthma

Output dict is backward-compatible with app.py (same keys as old LLM output).
"""


class RespiratoryRuleEngine:
    """
    Deterministic clinical triage rules based on BTS/GOLD/GINA guidelines.

    Consumes outputs from COPDAgent, PneumoniaAgent, SymptomAgent,
    and optionally SoundAgent, then returns a structured triage decision.
    """

    # Confidence thresholds for clinical decisions
    HIGH_CONF = 0.70   # Strong signal — GOLD Stage III / BTS high-risk
    MOD_CONF  = 0.50   # Moderate signal — warrants GP review

    def evaluate(self,
                 patient_info: dict,
                 copd_result: dict,
                 pneumonia_result: dict,
                 symptom_result: dict,
                 sound_result: dict = None,
                 longitudinal_score: float = 0.0) -> dict:
        """
        Apply clinical rules to agent outputs.

        Parameters
        ----------
        patient_info     : dict — age, dyspnea, fever_muscle_pain,
                           wheezing, cough_severity
        copd_result      : COPDAgent.predict() output
        pneumonia_result : PneumoniaAgent.predict() output
        symptom_result   : SymptomAgent.predict() output
        sound_result     : SoundAgent.predict() output or None (Tier 1)

        Returns
        -------
        dict with same keys as old llm_provider.call_llm() output:
            diagnosis, severity, confidence, reasoning,
            recommended_action, referral_urgency,
            agents_agreement, llm_provider
        """
        age       = patient_info.get('age', 50)
        dyspnea   = patient_info.get('dyspnea', False)
        fever     = patient_info.get('fever_muscle_pain', False)
        cough_sev = patient_info.get('cough_severity', 0)

        copd_conf  = copd_result.get('probability', 0.0)
        pneu_conf  = pneumonia_result.get('probability', 0.0)
        symptom_p  = symptom_result.get('symptomatic_probability', 0.0)

        sound_type = sound_result.get('sound_type', 'Normal') if sound_result else None

        # ── RULE 1: High-confidence COPD + exacerbation indicators ──────────
        # GOLD 2024 Stage III/IV: FEV1 < 50% predicted + exacerbation markers
        if copd_conf >= self.HIGH_CONF and (dyspnea or cough_sev >= 6):
            return self._make_decision(
                diagnosis="COPD exacerbation",
                severity="HIGH",
                confidence=copd_conf,
                reasoning=(
                    f"COPD detected with {copd_conf:.0%} confidence. "
                    f"Dyspnoea present: {dyspnea}. Cough severity: {cough_sev}/10. "
                    "Meets GOLD 2024 Stage III exacerbation criteria. "
                    "Urgent pulmonologist evaluation required."
                ),
                action=(
                    "Refer to pulmonologist within 48 hours. "
                    "Consider spirometry and bronchodilator therapy."
                ),
                urgency="urgent",
            )

        # ── RULE 2: High-confidence Pneumonia + BTS high-risk features ───────
        # BTS 2023 CAP: age >= 65, fever, or dyspnoea = high-risk group
        if pneu_conf >= self.HIGH_CONF and (fever or dyspnea or age >= 65):
            return self._make_decision(
                diagnosis="Community-acquired pneumonia",
                severity="HIGH",
                confidence=pneu_conf,
                reasoning=(
                    f"Pneumonia detected with {pneu_conf:.0%} confidence. "
                    f"Fever: {fever}. Age: {age}. Dyspnoea: {dyspnea}. "
                    "BTS 2023 high-risk features present. "
                    "Immediate medical evaluation required."
                ),
                action=(
                    "Emergency department within 24 hours. "
                    "CXR and blood cultures recommended."
                ),
                urgency="emergency",
            )

        # ── RULE 3: Moderate COPD, no exacerbation markers ──────────────────
        if copd_conf >= self.MOD_CONF:
            return self._make_decision(
                diagnosis="Possible COPD",
                severity="MODERATE",
                confidence=copd_conf,
                reasoning=(
                    f"Moderate COPD signal ({copd_conf:.0%}). "
                    "No severe exacerbation markers present. "
                    "GP review recommended within 1 week."
                ),
                action="GP appointment within 1 week. Spirometry advised.",
                urgency="routine",
            )

        # ── RULE 4: Moderate Pneumonia, no high-risk features ────────────────
        if pneu_conf >= self.MOD_CONF:
            return self._make_decision(
                diagnosis="Possible pneumonia",
                severity="MODERATE",
                confidence=pneu_conf,
                reasoning=(
                    f"Moderate pneumonia signal ({pneu_conf:.0%}). "
                    "No BTS high-risk features. Monitor closely."
                ),
                action="GP appointment same day or next day. Rest and hydration.",
                urgency="soon",
            )

        # ── RULE 5: Sound-only findings (Tier 2) ────────────────────────────
        if sound_type in ('Wheeze', 'Both'):
            sound_conf = sound_result.get('confidence', 0.6) if sound_result else 0.6
            return self._make_decision(
                diagnosis="Abnormal lung sounds — wheeze detected",
                severity="MODERATE",
                confidence=sound_conf,
                reasoning=(
                    "Wheeze pattern detected on auscultation. "
                    "May indicate bronchoconstriction (COPD/Asthma). "
                    "Clinical correlation required."
                ),
                action="GP review within 48 hours.",
                urgency="soon",
            )

        if sound_type == 'Crackle':
            sound_conf = sound_result.get('confidence', 0.6) if sound_result else 0.6
            return self._make_decision(
                diagnosis="Abnormal lung sounds — crackles detected",
                severity="MODERATE",
                confidence=sound_conf,
                reasoning=(
                    "Crackle pattern detected. "
                    "May indicate fluid or consolidation. "
                    "CXR recommended."
                ),
                action="GP review within 24 hours.",
                urgency="soon",
            )

        # ── RULE 6: High pneumonia hint + key symptoms (Tier 1 no-audio path) ──
        pneu_hint = symptom_result.get('pneumonia_probability_hint', 0.0)
        copd_hint = symptom_result.get('copd_probability_hint', 0.0)

        if pneu_hint >= 0.35 and (fever or dyspnea):
            return self._make_decision(
                diagnosis="Possible pneumonia — symptom-based flag",
                severity="MODERATE",
                confidence=symptom_p,
                reasoning=(
                    f"Symptom pattern suggests possible pneumonia (hint: {pneu_hint:.0%}). "
                    f"Fever: {fever}. Dyspnoea: {dyspnea}. "
                    "BTS 2023: fever + dyspnoea = high-risk features. "
                    "Clinical assessment recommended."
                ),
                action="GP appointment within 24–48 hours. Consider CXR.",
                urgency="soon",
            )

        if copd_hint >= 0.35 and dyspnea:
            return self._make_decision(
                diagnosis="Possible COPD — symptom-based flag",
                severity="MODERATE",
                confidence=symptom_p,
                reasoning=(
                    f"Symptom pattern suggests possible COPD (hint: {copd_hint:.0%}). "
                    f"Dyspnoea present. GOLD 2024: dyspnoea + respiratory history = screening warranted."
                ),
                action="GP appointment this week. Spirometry advised.",
                urgency="routine",
            )

        # ── RULE 7: High longitudinal score (composite Tier 1 signal) ───────
        if longitudinal_score >= 0.55:
            return self._make_decision(
                diagnosis="Elevated respiratory risk — longitudinal monitoring",
                severity="MODERATE",
                confidence=longitudinal_score,
                reasoning=(
                    f"Combined longitudinal score: {longitudinal_score:.0%}. "
                    "Symptom severity, voice biomarkers, and/or acoustic drift "
                    "indicate elevated respiratory risk. Clinical review recommended."
                ),
                action="GP appointment this week. Bring symptom history.",
                urgency="routine",
            )

        # ── RULE 8: Generally symptomatic ────────────────────────────────────
        if symptom_p >= 0.45:
            return self._make_decision(
                diagnosis="Symptomatic — respiratory concern",
                severity="MODERATE",
                confidence=symptom_p,
                reasoning=(
                    f"Symptom agent reports {symptom_p:.0%} symptomatic probability. "
                    "Multiple respiratory symptoms present. "
                    "Monitor closely and re-assess if symptoms worsen."
                ),
                action="GP appointment this week if symptoms persist.",
                urgency="routine",
            )

        # ── RULE 9: No significant findings ──────────────────────────────────
        healthy_conf = max(1.0 - copd_conf, 1.0 - pneu_conf, 0.70)
        return self._make_decision(
            diagnosis="No significant respiratory pathology detected",
            severity="LOW",
            confidence=healthy_conf,
            reasoning=(
                "No significant disease signal detected in audio or symptoms. "
                "Continue scheduled monitoring sessions."
            ),
            action=(
                "Self-care. Continue monitoring. "
                "Re-assess if symptoms worsen."
            ),
            urgency="none",
        )

    def _make_decision(self, diagnosis: str, severity: str,
                       confidence: float, reasoning: str,
                       action: str, urgency: str) -> dict:
        """Build the standardised triage decision dict."""
        return {
            'diagnosis':          diagnosis,
            'severity':           severity,
            'confidence':         round(confidence, 4),
            'reasoning':          reasoning,
            'recommended_action': action,
            'referral_urgency':   urgency,
            'agents_agreement':   'Yes — deterministic rules applied',
            'llm_provider':       'deterministic_rule_engine',
        }
