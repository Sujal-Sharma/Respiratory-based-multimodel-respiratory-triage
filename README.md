# Multimodal Respiratory Triage Using Agentic AI

Research project for a biomedical clinical AI journal paper.

## Overview

A multimodal agentic AI system for respiratory disease triage that:
1. Takes cough audio + lung audio + patient symptoms as input
2. Runs them through specialist ML agents (one per modality)
3. Passes all results to an LLM orchestrator via LangGraph
4. Outputs a clinical triage decision: diagnosis + severity + recommended action

## Architecture

```
Agent 1 — Cough Agent (COUGHVID dataset)
  ├── XGBoost          ← metadata + MFCCs (8 meta + 160 MFCC features)
  │                       → Healthy / Symptomatic
  └── LightCoughCNN    ← mel spectrograms (128×173, 1-channel)
                          → Healthy / Symptomatic (~295K params, from scratch)

Agent 2 — Lung Agent (ICBHI + KAUH + HF Lung V1 datasets)
  └── MultiTaskEfficientNet-B0 (shared backbone, two heads)
      ├── Disease head  → Normal / COPD / Pneumonia / Asthma / Heart_Failure
      │                   (trained on ICBHI + KAUH)
      └── Sound head    → Normal / Crackle / Wheeze / Both
                          (trained on ICBHI + KAUH + HF Lung V1)

Agent 3 — Symptom Agent
  └── XGBoost          ← patient-reported symptoms (8 tabular features)
                          → Healthy / Symptomatic

LangGraph Triage Pipeline (in progress)
  └── Orchestrates all agents → LLM reasoning → clinical JSON output
      LLM: Gemini 2.5 Flash (cloud) + Gemma 2 2B via Ollama (local fallback)
```

## Two-Tier Triage Design

**Tier 1 — Patient Self-Screening (phone/web)**
- Cough audio + symptom form only (no stethoscope needed)
- Output: preliminary risk flag

**Tier 2 — Clinician Confirmation (clinic)**
- Full lung sound recording via stethoscope
- Multi-task disease + sound analysis
- LLM-synthesised triage recommendation with clinician in the loop

## Current Results

| Model | Task | Accuracy | Macro F1 | AUROC |
|-------|------|----------|----------|-------|
| XGBoost (COUGHVID) | Symptom classification | 61.67% | — | — |
| LightCoughCNN (COUGHVID) | Cough classification | 78.95% | 0.7403 | 0.753 |
| MultiTaskEfficientNet (Disease) | 5-class lung disease | 93.67% | — | — |
| MultiTaskEfficientNet (Sound) | 4-class lung sound | — | — | — |

## GPU Requirements

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- CUDA 12.6 / PyTorch with cu121
- float16 mixed-precision for MultiTaskEfficientNet
- LightCoughCNN runs in float32 (small model)

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
```

## Execution Order

```bash
# Step 1: Preprocess datasets
python preprocessing.py

# Step 2: Train all models
python training.py   # trains XGBoost → LightCoughCNN (MultiTask commented out)

# Step 3: Evaluate all models
python evaluation.py
```

## Datasets

| Dataset | Source | Used By | Task |
|---------|--------|---------|------|
| COUGHVID | Crowd-sourced cough recordings | XGBoost + LightCoughCNN | Healthy vs Symptomatic |
| ICBHI | Respiratory Sound Database | MultiTaskEfficientNet | Disease + sound classification |
| KAUH | Clinical lung recordings | MultiTaskEfficientNet | Disease + sound classification |
| HF Lung V1 | HuggingFace lung sounds | MultiTaskEfficientNet (sound head only) | Sound classification |

## Project Structure

```
├── config.py              # All paths, hyperparameters, constants
├── preprocessing.py       # Dataset loading + feature extraction
├── training.py            # XGBoost + LightCoughCNN + MultiTaskEfficientNet training
├── evaluation.py          # Metrics, confusion matrices, ROC curves
├── utils.py               # Seed, checkpoints, plotting helpers
├── agents/
│   ├── cough_agent.py     # LightCoughCNN inference wrapper
│   ├── lung_agent.py      # MultiTaskEfficientNet inference wrapper
│   └── symptom_agent.py   # XGBoost inference wrapper
├── models/
│   ├── cnn_model.py       # MultiTaskEfficientNet + LightCoughCNN architectures
│   └── xgboost_model.py   # XGBoost configuration
├── saved_models/          # Trained model weights (not in repo)
├── data/                  # Processed features + labels (not in repo)
├── DATASET/               # Raw audio datasets (not in repo)
└── outputs/               # Evaluation plots + reports (not in repo)
```

## Project Status

- [x] Phase 1: COUGHVID preprocessing, training, evaluation
- [x] Phase 1b: Lung sound preprocessing (ICBHI + KAUH + HF Lung V1), MultiTaskEfficientNet training
- [ ] Phase 2: LangGraph agentic triage pipeline
- [ ] Phase 3: LLM orchestrator (Gemini + Ollama Gemma 2)
- [ ] Phase 4: Streamlit web app (two-tier UI)
- [ ] Phase 5: Full system evaluation + explainability (SHAP / Grad-CAM)