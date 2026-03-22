# Multimodal Respiratory Triage Using Agentic AI

Research project for a biomedical clinical AI journal paper.

## Overview

A multimodal agentic AI system for respiratory disease triage that:
1. Takes respiratory audio recordings + patient symptoms as input
2. Runs them through specialist ML models (one per dataset)
3. Passes all results to an LLM orchestrator (LangGraph + Llama 3)
4. Outputs a clinical triage decision: diagnosis + severity + action

## Architecture

```
Layer 1 — Specialist ML Models
  ├── XGBoost        ← COUGHVID metadata     → COVID-19 / healthy / symptomatic
  ├── EfficientNet-B0 ← Resp-229K spectrograms → 7-class respiratory disease
  └── 1D-CNN         ← Coswara audio (Phase 2) → COVID severity

Layer 2 — LangGraph StateGraph Pipeline (Phase 3)
  └── Orchestrates all 3 agents sequentially

Layer 3 — LLM Reasoning Module (Phase 4)
  └── Llama 3 / Gemma 2 → clinical JSON output
```

## GPU Requirements

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- CUDA 12.6 / PyTorch with cu121
- float16 inference for VRAM efficiency

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
# Step 1: Preprocess COUGHVID (fast, tabular)
python preprocessing.py

# Step 2: Train XGBoost
python training.py   # trains XGBoost first, then EfficientNet

# Step 3: Evaluate both models
python evaluation.py
```

## Quality Targets (Test Set)

| Metric    | Target |
|-----------|--------|
| Accuracy  | > 88%  |
| Macro F1  | > 0.85 |
| Recall    | > 0.85 |
| AUROC     | > 0.90 |

## Datasets

| Dataset   | Model           | Task                          |
|-----------|-----------------|-------------------------------|
| COUGHVID  | XGBoost         | COVID status from metadata    |
| Resp-229K | EfficientNet-B0 | 7-class respiratory disease   |
| Coswara   | 1D-CNN (Phase 2)| COVID severity classification |

## Output Files

```
outputs/
  confusion_matrix_xgboost.png
  confusion_matrix_efficientnet.png
  roc_curve_xgboost.png
  roc_curve_efficientnet.png
  training_loss_efficientnet.png
  model_comparison_table.csv

saved_models/
  xgboost_coughvid.pkl
  efficientnet_resp229k.pt
```

## Project Phases

- **Phase 1 (Current)**: COUGHVID + Resp-229K preprocessing, training, evaluation
- **Phase 2**: Coswara dataset + 1D-CNN
- **Phase 3**: LangGraph agentic pipeline
- **Phase 4**: LLM orchestrator (Llama 3 / Gemma 2)
- **Phase 5**: Full system evaluation + SHAP + Grad-CAM + clinical validation