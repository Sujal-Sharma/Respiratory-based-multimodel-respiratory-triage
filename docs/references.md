# References and Threshold Provenance

This file maps the important numeric values in the codebase to their source class and provides a short summary of the supporting paper or guideline.

Status labels used below:
- **Paper-backed**: the numeric value is directly traceable to a paper or guideline result.
- **Guideline-inspired**: the logic is consistent with a guideline, but the exact number is an implementation choice.
- **Data-calibrated**: selected from validation performance during training.
- **Heuristic**: engineering or UI choice.

## 1) Value-to-Source Mapping

| Value / Boundary | Used in | Source / reason | Status |
|---|---|---|---|
| `MCID = 0.05` | `pipeline/longitudinal.py`, `agents/symptom_agent.py` | CAT minimum clinically important difference mapped from CAT literature | Paper-backed |
| `W_SYMPTOM = 0.50` | `pipeline/longitudinal.py` | Symptom signal weighted most heavily in the Tier 1 composite | Heuristic |
| `W_VOICE = 0.35` | `pipeline/longitudinal.py` | Voice biomarkers contribute secondary longitudinal signal | Heuristic |
| `W_DRIFT = 0.15` | `pipeline/longitudinal.py` | Cough embedding drift contributes but is least dominant | Heuristic |
| `HIGH_CONF = 0.70` | `pipeline/rule_engine.py` | High-confidence disease branch for COPD/pneumonia | Guideline-inspired |
| `MOD_CONF = 0.50` | `pipeline/rule_engine.py` | Moderate-risk GP review branch | Heuristic |
| `cough_sev >= 6` | `pipeline/rule_engine.py` | Escalation marker for COPD exacerbation branch | Guideline-inspired |
| `age >= 65` | `pipeline/rule_engine.py` | Higher-risk pneumonia branch, consistent with CAP triage logic | Guideline-inspired |
| `pneu_hint >= 0.35` | `pipeline/rule_engine.py` | Symptom-only pneumonia flag | Heuristic |
| `copd_hint >= 0.35` | `pipeline/rule_engine.py` | Symptom-only COPD flag | Heuristic |
| `longitudinal_score >= 0.40` | `pipeline/rule_engine.py` | Moderate longitudinal risk escalation | Heuristic |
| `symptom_p >= 0.40` | `pipeline/rule_engine.py` | General symptomatic escalation | Heuristic |
| `healthy_conf floor = 0.70` | `pipeline/rule_engine.py` | Prevents overly weak “healthy” confidence | Heuristic |
| `duration < 1.0` sec rejected | `agents/voice_agent.py` | Too short for stable voice biomarker extraction | Heuristic |
| `duration > 8.0` sec trimmed | `agents/voice_agent.py` | Use the middle segment for a stable phonation window | Heuristic |
| `RMS < 0.001` rejected | `agents/voice_agent.py` | Reject near-silent or unusable audio | Heuristic |
| `symptomatic_probability >= 0.35` | `agents/symptom_agent.py` | Binary symptom flag | Heuristic |
| `age >= 40` | `agents/symptom_agent.py` | Early risk enrichment for COPD screening | Guideline-inspired |
| `age >= 50` | `agents/symptom_agent.py` | Intermediate age-risk enrichment | Guideline-inspired |
| `age >= 65` | `agents/symptom_agent.py` | High age-risk enrichment | Guideline-inspired |
| `resp_condition += 0.40` | `agents/symptom_agent.py` | Strong weight for known respiratory history | Guideline-inspired |
| `dyspnea_lvl >= 2 += 0.25` | `agents/symptom_agent.py` | Breathlessness as major severity signal | Guideline-inspired |
| `wheezing += 0.15` | `agents/symptom_agent.py` | Obstructive airway symptom support | Guideline-inspired |
| `fever_muscle_pain += 0.45` | `agents/symptom_agent.py` | Strong pneumonia/infection marker | Guideline-inspired |
| `cough_norm >= 0.7 += 0.20` | `agents/symptom_agent.py` | High cough burden threshold | Heuristic |
| `sound_conf default = 0.6` | `pipeline/rule_engine.py` | Fallback confidence when auscultation confidence is missing | Heuristic |
| `sound_type in {Wheeze, Both}` | `pipeline/rule_engine.py` | Lung-sound routing to obstructive pattern branch | Guideline-inspired |
| `sound_type == Crackle` | `pipeline/rule_engine.py` | Crackle branch for consolidation/fluid suspicion | Guideline-inspired |

## 2) Source Summaries and Why the Values Exist

### A. Jones PW et al., 2009, European Respiratory Journal
**Citation:** Jones PW, Harding G, Berry P, Wiklund I, Chen WH, Kline Leidy N. *Development and first validation of the COPD Assessment Test.* Eur Respir J. 2009;34(3):648-654. PMID: 19720809.

**What it contributed:**
- First validated the **COPD Assessment Test (CAT)** as a short, practical patient-reported measure of COPD health status.
- The breakthrough was simplicity: a standardized, easy-to-use tool that made symptom burden measurable in routine care.

**How it maps to this project:**
- The CAT-style composite in `agents/symptom_agent.py` is built around the same idea: turn symptoms into a normalized severity index.
- This is the conceptual basis for the symptom composite weights and the Tier 1 symptom score.

**Relevant code values influenced by it:**
- `agents/symptom_agent.py` composite symptom weights.
- `pipeline/longitudinal.py` symptom-dominant fusion weight `0.50`.

---

### B. Kon SS et al., 2014, Lancet Respiratory Medicine
**Citation:** Kon SS, Canavan JL, Jones SE, Nolan CM, Clark AL, Dickson MJ, Haselden BM, Polkey MI, Man WD. *Minimum clinically important difference for the COPD Assessment Test: a prospective analysis.* Lancet Respir Med. 2014;2(3):195-203. PMID: 24621681. doi: 10.1016/S2213-2600(14)70001-3.

**What it contributed:**
- Established the **MCID** for the CAT.
- The breakthrough is that it gave a clinically interpretable threshold for change, not just a score.

**How it maps to this project:**
- The project uses `MCID = 0.05` as the normalized 0-1 equivalent of a meaningful CAT change.
- This is the clearest paper-backed boundary in the longitudinal system.

**Relevant code values influenced by it:**
- `pipeline/longitudinal.py`: `MCID = 0.05`
- `agents/symptom_agent.py`: `MCID = 0.05`

---

### C. GOLD report series (COPD guideline document)
**Source:** Global Initiative for Chronic Obstructive Lung Disease (GOLD). The GOLD website states that the strategy document is an evidence-based COPD diagnosis/management/prevention resource updated yearly.

**What it contributed:**
- Not a single paper, but a continuously updated evidence-based COPD strategy.
- The breakthrough is standardization: a global COPD management framework used in clinical practice.

**How it maps to this project:**
- Supports the COPD-oriented thresholds, age-risk enrichment, symptom escalation, and triage language.
- The specific numeric values in the code are implementation choices built on top of the GOLD framework, not copied from one single GOLD number.

**Relevant code values influenced by it:**
- `HIGH_CONF = 0.70`
- `MOD_CONF = 0.50`
- `cough_sev >= 6`
- `age >= 40`, `age >= 65`
- symptom-based COPD hint logic

---

### D. BTS community-acquired pneumonia guideline series
**Source:** British Thoracic Society adult CAP guidance and related summaries indexed in PubMed.

**What it contributed:**
- Not a single paper with one magic threshold, but a guideline framework for CAP risk stratification.
- The breakthrough is operationalizing adult pneumonia triage using age, fever, dyspnea, and clinical severity markers.

**How it maps to this project:**
- Supports the pneumonia risk logic in `pipeline/rule_engine.py` and the symptom-based pneumonia hint in `agents/symptom_agent.py`.
- The exact numeric boundary `age >= 65` is a guideline-inspired implementation choice rather than a direct copied output from one paper.

**Relevant code values influenced by it:**
- `HIGH_CONF = 0.70`
- `age >= 65`
- `fever` and `dyspnea` high-risk routing
- `fever_muscle_pain += 0.45`
- `pneu_hint >= 0.35`

---

### E. GINA report series (asthma guideline document)
**Source:** Global Initiative for Asthma (GINA) report series.

**What it contributed:**
- Comprehensive, evidence-based global asthma strategy.
- The breakthrough is a single standardized clinical strategy updated over time.

**How it maps to this project:**
- Used as background clinical guidance for wheeze/airway symptom interpretation.
- The code does not directly encode asthma-specific thresholds from GINA, but GINA is part of the clinical rationale behind the rule-engine wording.

**Relevant code values influenced by it:**
- Lung-sound interpretation logic around wheeze
- General respiratory risk framing

---

### F. Voice biomarker literature used for the writeup

**1) Bartl-Pokorny et al., 2021, Journal of the Acoustical Society of America**
**Citation:** Bartl-Pokorny KD, Pokorny FB, Batliner A, Amiriparian S, Semertzidou A, Eyben F, Kramer E, Schmidt F, Schonweiler R, Wehler M, Schuller BW. *The voice of COVID-19: acoustic correlates of infection in sustained vowels.* J Acoust Soc Am. 2021;149(6):4377-4383. doi: 10.1121/10.0005194.

**What it contributed:**
- Showed that sustained-vowel voice acoustics can separate COVID-positive and COVID-negative speakers.
- The main breakthrough was proof-of-concept that voice carries measurable infection-related signal.

**How it maps to this project:**
- Supports keeping sustained-vowel voice biomarkers in the pipeline.
- Supports the idea that voice can be one longitudinal signal among several, not a standalone diagnosis.

**Relevant code values influenced by it:**
- `jitter/shimmer/hnr/f0_std/phonation_duration` weights
- `duration < 1.0`, `duration > 8.0`, `RMS < 0.001`

**2) Shastry et al., 2014, International Journal of Phonosurgery and Laryngology**
**Citation:** Shastry A, et al. *Voice analysis in individuals with chronic obstructive pulmonary disease.* Int J Phonosurg Laryngol. 2014;4(2):45-48.

**What it contributed:**
- Explored voice analysis in COPD rather than only general dysphonia.
- The breakthrough is the respiratory-specific framing of voice as a screening signal.

**How it maps to this project:**
- This is the closest verified COPD voice-analysis source for the voice biomarker branch.
- It is the better citation for the COPD-oriented voice comments than the earlier unverified Gupta reference.

**Relevant code values influenced by it:**
- `W_VOICE = 0.35`
- Voice biomarker feature weighting
- MCID-style interpretation of change in sustained-vowel features

**Note on the older Gupta citation:**
- I could not fully verify the bibliographic details for the earlier Gupta reference during this session.
- If you want, I can do a second pass specifically to replace every remaining placeholder citation in the code comments with exact IEEE or Vancouver entries.

## 3) Data-Calibrated Model Thresholds

These thresholds are not taken directly from a paper; they come from validation-based operating-point selection.

### COPD binary model
Source: `scripts/train_binary_agent.py`
- Threshold sweep over validation set.
- Constraint: `TARGET_RECALL = 0.80`.
- Rationale: prioritize sensitivity so the system does not miss likely cases.

### Pneumonia binary model
Source: `scripts/train_pneumonia_cv.py`
- 5-fold CV with threshold tuning per fold.
- Final threshold is the median of fold-specific thresholds.
- Rationale: robust operating point selection under limited positive samples.

### Cough model
Source: `scripts/train_cough_agent.py`
- Threshold tuned on validation set over `0.30` to `0.80`.
- Rationale: maximize macro F1 on symptomatic vs healthy labels.

### Sound model
Source: `scripts/train_sound_3class.py`
- `Both` class merged into `Crackle`.
- Rationale: class sparsity and clinical dominance of crackles when both are present.

## 4) Suggested wording for the paper

If you want to describe these values in a methods section, use this split:
- **Paper-backed values**: CAT MCID, CAT-style symptom framing.
- **Guideline-inspired values**: GOLD/BTS/GINA clinical routing cutoffs.
- **Data-calibrated values**: model thresholds selected on validation.
- **Heuristic values**: UI bins, fallback confidences, audio quality filters.

This avoids overstating what is directly reported in the literature versus what is a project-specific calibration choice.
