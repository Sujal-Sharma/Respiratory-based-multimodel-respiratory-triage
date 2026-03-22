# Multimodal Respiratory Triage Using Agentic AI

Research project for a biomedical clinical AI journal paper.

## Overview

A multimodal agentic AI system for respiratory disease triage that:
1. Takes cough audio + lung audio + patient symptoms as input
2. Runs them through 3 specialist ML agents (one per modality)
3. Orchestrates agents via a LangGraph StateGraph pipeline with two-tier routing
4. Passes all agent results to an LLM (Groq Llama 3.3 70B) for clinical reasoning
5. Outputs a structured triage decision: diagnosis + severity + action + urgency

## Architecture

```
Agent 1 — Cough Agent (cough_agent.py)
  Model: LightCoughCNN (~295K params, trained from scratch)
  Input: Mel spectrogram (1, 128, 173) from phone-recorded cough audio
  Output: Healthy / Symptomatic + confidence
  Dataset: COUGHVID (crowd-sourced, ~8,992 samples with pitch-shift augmentation)
  Accuracy: 78.95% | Macro F1: 0.7403 | AUROC: 0.753
  Note: No pre-emphasis (destroys low-freq cough signal).
        COVID-19 merged into Symptomatic (symptoms overlap).
        ImageNet transfer learning proven NOT to work for spectrograms.

Agent 2 — Lung Agent (lung_agent.py)
  Model: MultiTaskEfficientNet-B0 (shared backbone, two classification heads)
  Input: Mel spectrogram (1, 128, 250) from stethoscope lung recording
  Disease head: Normal / COPD / Pneumonia / Asthma / Heart_Failure (5 classes)
                Trained on ICBHI + KAUH datasets
                Accuracy: 93.67%
  Sound head:   Normal / Crackle / Wheeze / Both (4 classes)
                Trained on ICBHI + KAUH + HF Lung V1 datasets
  Note: float16 mixed-precision for GTX 1650 4GB VRAM.

Agent 3 — Symptom Agent (symptom_agent.py)
  Model: XGBoost (GPU-accelerated)
  Input: 168 features (8 patient metadata + 160 MFCC features)
  Output: Healthy / Symptomatic + confidence
  Dataset: COUGHVID metadata
  Accuracy: 61.67%
  Note: When called without audio (pipeline mode), MFCC features are
        zero-padded. Metadata alone gives ~51% confidence — acts as
        supplementary signal, not primary classifier.

LangGraph Triage Pipeline (pipeline/triage_graph.py)
  StateGraph with conditional routing:
    START -> analyze_cough -> analyze_symptoms -> route_tier
      Tier 1 (no lung audio): -> llm_triage -> END
      Tier 2 (lung audio):    -> analyze_lung -> llm_triage -> END

LLM Clinical Reasoning (pipeline/llm_provider.py)
  Primary: Groq API — Llama 3.3 70B (30 RPM, 14,400 req/day, free)
  Fallback: Google Gemini 2.5 Flash API (10 RPM, 250 req/day, free)
  Output: JSON with diagnosis, severity, confidence, reasoning,
          recommended_action, referral_urgency, agents_agreement

Streamlit Web App (app.py)
  Two-tier interface:
    Tier 1: Cough upload + symptom form -> risk screening
    Tier 2: + Lung upload -> full triage with all 3 agents + LLM
  Run: streamlit run app.py
```

## Two-Tier Triage Design

**Tier 1 — Patient Self-Screening (phone/web, no stethoscope needed)**
- Patient uploads cough recording from phone + fills symptom form
- Cough Agent + Symptom Agent analyze the inputs
- LLM provides preliminary risk flag
- Output example: "Respiratory Infection, MODERATE, referral: soon"

**Tier 2 — Clinician Confirmation (clinic with stethoscope)**
- Clinician uploads stethoscope lung recording
- All 3 agents run (cough + symptoms + lung disease/sound)
- LLM synthesises all data for specific diagnosis
- Output example: "Asthma with wheeze pattern, MODERATE, bronchodilator + follow-up"

## Current Model Results

| Model | Task | Accuracy | Macro F1 | AUROC |
|-------|------|----------|----------|-------|
| XGBoost (COUGHVID) | Symptom: Healthy vs Symptomatic | 61.67% | -- | -- |
| LightCoughCNN (COUGHVID) | Cough: Healthy vs Symptomatic | 78.95% | 0.7403 | 0.753 |
| MultiTaskEfficientNet Disease | 5-class lung disease | 93.67% | -- | -- |
| MultiTaskEfficientNet Sound | 4-class lung sound | -- | -- | -- |

**Key context:**
- COUGHVID labels are crowd-sourced self-reports — published papers report 65-80% for audio-only
- LightCoughCNN Symptomatic recall is 47.3% (model bias toward Healthy) — data quality issue
- MultiTaskEfficientNet is the strongest model and core of Tier 2
- The agentic design means the system is stronger than any individual model

## GPU Requirements

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- CUDA 12.6 / PyTorch with cu121
- float16 mixed-precision for MultiTaskEfficientNet inference
- LightCoughCNN runs in float32 (small model, ~295K params)
- LLM runs on cloud (Groq/Gemini) — 0 MB local VRAM

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv triage_env
triage_env\Scripts\activate      # Windows
source triage_env/bin/activate   # Linux/Mac

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 5. Set up API keys (copy .env.example to .env and add your keys)
cp .env.example .env
# Edit .env with your Groq key (https://console.groq.com/keys)
# and Gemini key (https://aistudio.google.com/apikey)
```

## Running

```bash
# Train models (if not already trained)
python preprocessing.py
python training.py       # XGBoost + LightCoughCNN (MultiTask commented out)
python evaluation.py     # generates confusion matrices, ROC curves, reports

# Run the Streamlit web app
streamlit run app.py

# Or run the pipeline from Python
from pipeline.triage_graph import run_triage
result = run_triage(
    patient_info={"age": 45, "gender": "male", "symptoms": ["fever", "cough"],
                  "fever_muscle_pain": True, "respiratory_condition": False,
                  "cough_detected": 0.9, "dyspnea": True,
                  "wheezing": False, "congestion": False},
    cough_audio_path="path/to/cough.wav",
    lung_audio_path="path/to/lung.wav",  # empty string for Tier 1
)
```

## Datasets

| Dataset | Source | Used By | Task |
|---------|--------|---------|------|
| COUGHVID | Crowd-sourced cough recordings (~7,310 quality-filtered) | XGBoost + LightCoughCNN | Healthy vs Symptomatic |
| ICBHI | Respiratory Sound Database (stethoscope) | MultiTaskEfficientNet (both heads) | Disease + sound |
| KAUH | Clinical lung recordings | MultiTaskEfficientNet (both heads) | Disease + sound |
| HF Lung V1 | HuggingFace lung sounds | MultiTaskEfficientNet (sound head only) | Sound classification |

## Project Structure

```
├── app.py                 # Streamlit two-tier triage web app
├── config.py              # All paths, hyperparameters, constants
├── preprocessing.py       # Dataset loading + feature extraction
├── training.py            # XGBoost + LightCoughCNN + MultiTaskEfficientNet training
├── evaluation.py          # Metrics, confusion matrices, ROC curves
├── utils.py               # Seed, checkpoints, plotting helpers
├── .env.example           # API key template (Groq + Gemini)
├── requirements.txt       # All Python dependencies
├── agents/
│   ├── __init__.py        # Agent registry docs
│   ├── cough_agent.py     # LightCoughCNN inference (phone cough audio)
│   ├── lung_agent.py      # MultiTaskEfficientNet inference (stethoscope audio)
│   └── symptom_agent.py   # XGBoost inference (patient metadata)
├── models/
│   ├── __init__.py
│   ├── cnn_model.py       # MultiTaskEfficientNet + LightCoughCNN architectures
│   └── xgboost_model.py   # XGBoost configuration
├── pipeline/
│   ├── __init__.py
│   ├── llm_provider.py    # Groq (primary) + Gemini (fallback) LLM with clinical prompt
│   └── triage_graph.py    # LangGraph StateGraph with two-tier routing
├── saved_models/          # Trained weights (not in repo — too large)
│   ├── xgboost_coughvid.pkl         # 84 KB
│   ├── coughvid_efficientnet.pt     # 3.5 MB (LightCoughCNN despite filename)
│   └── multitask_efficientnet.pt    # 57 MB
├── data/                  # Processed labels (CSVs in repo, spectrograms not)
├── DATASET/               # Raw audio datasets (not in repo)
└── outputs/               # Evaluation plots + reports (in repo)
```

## Technical Decisions & Lessons Learned

1. **Pre-emphasis destroys cough signal**: Removing pre-emphasis (coef=0.97) was the breakthrough that enabled the cough model to learn. Cough classification relies on low-frequency characteristics.

2. **ImageNet features do NOT transfer to mel spectrograms**: Confirmed across 4 experiments (frozen/unfrozen, 1-ch/3-ch). LightCoughCNN from scratch outperforms EfficientNet transfer learning.

3. **LightCoughCNN (~295K params) is correct for ~6,300 training samples**: EfficientNet-B0 (4.3M params) catastrophically overfits on this dataset size.

4. **Pitch-shifting augmentation (+3% accuracy)**: Applied to minority class (Symptomatic) with n_steps=-4 semitones during spectrogram extraction, per CovidCoughNet 2023.

5. **Dual-provider LLM**: Groq primary (faster, higher limits) + Gemini fallback ensures demos never crash from rate limits.

6. **Symptom Agent zero-padding**: XGBoost trained on 168 features but pipeline sends only 8 metadata features. MFCC columns zero-padded — works as supplementary signal.

## Project Status

- [x] Phase 1: COUGHVID preprocessing, training (XGBoost + LightCoughCNN), evaluation
- [x] Phase 1b: Lung sound preprocessing (ICBHI + KAUH + HF Lung V1), MultiTaskEfficientNet training (93.67%)
- [x] Phase 2: LangGraph agentic triage pipeline with two-tier routing
- [x] Phase 3: LLM orchestrator (Groq Llama 3.3 70B + Gemini 2.5 Flash fallback)
- [x] Phase 4: Streamlit web app (two-tier UI, tested end-to-end)
- [ ] Phase 5: Explainability (SHAP for XGBoost, Grad-CAM for CNNs)
- [ ] Phase 6: Paper writing + final evaluation

## Files NOT in Repository (too large)

- `DATASET/` — raw audio files (COUGHVID, ICBHI, KAUH)
- `data/spectrograms/` — 977 MB of .npy mel spectrograms
- `data/coughvid_spectrograms/` — 722 MB of .npy cough spectrograms
- `saved_models/` — trained weights (61 MB total)
