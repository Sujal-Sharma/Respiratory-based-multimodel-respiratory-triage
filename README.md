# Multimodal Respiratory Triage Using Agentic AI

A research project for a biomedical clinical AI journal paper.

## Overview

An agentic AI system for respiratory disease triage using the **OPERA-CT** respiratory foundation model (NeurIPS 2024). The system takes cough/breathing audio and patient symptoms as input, runs them through focused specialist agents, and produces a structured clinical triage decision using deterministic rule-based reasoning.

**Targeted diseases:** COPD and Pneumonia

---

## Architecture

```
Audio Input
    └── OPERA-CT Encoder (768-dim embeddings, HT-SAT backbone)
            ├── COPD Agent       (BinaryMLP, 768→256→64→2)
            ├── Pneumonia Agent  (BinaryMLP, 768→256→64→2)
            └── Sound Agent      (MLP, 768→512→256→64→3) [Tier 2 only]

Symptom Agent  (clinical heuristics, no ML)

LangGraph StateGraph
    analyze_symptoms → run_copd_agent → run_pneumonia_agent
        → [analyze_lung — Tier 2 only]
        → apply_rules → record_session → END

Rule Engine (deterministic BTS/GOLD/GINA clinical guidelines)
    → Triage Decision: diagnosis, severity, confidence, referral urgency

Session Agent (SQLite + linear regression deterioration detection)
```

---

## Two-Tier Design

**Tier 1 — Patient Self-Screening**
- Patient uploads a cough or breathing recording + fills symptom form
- COPD Agent + Pneumonia Agent + Symptom Agent run
- Rule engine produces: severity, diagnosis, recommended action

**Tier 2 — Clinician Confirmation**
- Clinician adds a stethoscope lung sound recording
- Sound Agent also runs (Normal / Crackle / Wheeze)
- All signals combined for final triage decision

---

## Model Results

| Model | Task | Accuracy | Macro F1 | Recall | AUROC |
|---|---|---|---|---|---|
| COPD Agent (OPERA-MLP) | COPD vs Normal | 94.8% | 0.947 | 0.959 | 0.995 |
| Pneumonia Agent (OPERA-MLP, 5-fold CV) | Pneumonia vs Normal | 97.7% | 0.869 | 0.750 | 0.984 |
| Sound Classifier (OPERA-MLP, 3-class) | Normal / Crackle / Wheeze | 60.5% | 0.594 | — | — |

**Notes:**
- Pneumonia evaluated via 5-fold stratified cross-validation (only 52 positive samples available)
- Sound classifier "Both" class removed — merged into Crackle due to insufficient samples (190 total)
- All models use pre-computed OPERA-CT embeddings — OPERA runs once, training reads `.npy` files only

---

## Datasets

| Dataset | Task | Samples |
|---|---|---|
| ICBHI Respiratory Sound Database | COPD binary + lung sound labels | 6898 recordings |
| KAUH Clinical Lung Recordings | COPD + Pneumonia binary + sound labels | 147 recordings |
| COUGHVID | COPD binary (cough audio) | ~7,500 recordings |
| HF Lung V1 | Lung sound classification | 7503 recordings |

Total OPERA embeddings extracted: **11,579 `.npy` files** (768-dim each)

---

## Project Structure

```
├── app.py                          # Streamlit two-tier triage web app
├── requirements.txt                # Python dependencies
├── agents/
│   ├── copd_agent.py               # COPD binary specialist (OPERA + MLP)
│   ├── pneumonia_agent.py          # Pneumonia binary specialist (OPERA + MLP)
│   ├── sound_agent.py              # Lung sound classifier — Tier 2 (3-class)
│   ├── symptom_agent.py            # Clinical heuristic symptom scorer
│   └── session_agent.py            # Session recording + deterioration alerts
├── models/
│   ├── opera_encoder.py            # Batched GPU OPERA-CT encoder (ThreadPoolExecutor)
│   ├── mlp_classifier.py           # BinaryMLPClassifier, SoundMLPClassifier, FocalLoss
│   └── embedding_dataset.py        # PyTorch Dataset for pre-computed .npy embeddings
├── pipeline/
│   ├── triage_graph.py             # LangGraph StateGraph orchestration
│   └── rule_engine.py              # Deterministic BTS/GOLD/GINA rule engine
├── database/
│   └── session_store.py            # SQLite session store + linear regression deterioration
├── scripts/
│   ├── build_label_csvs.py         # Build label CSVs from all 4 datasets
│   ├── fix_kauh_parser.py          # Parse KAUH dataset filenames
│   ├── convert_coughvid_webm.py    # Convert COUGHVID .webm → .wav (ffmpeg)
│   ├── extract_opera_embeddings.py # One-time OPERA-CT embedding extraction
│   ├── train_binary_agent.py       # Train COPD or Pneumonia MLP classifier
│   ├── train_pneumonia_cv.py       # Train Pneumonia with 5-fold CV
│   ├── train_sound_3class.py       # Train 3-class sound classifier
│   └── evaluate_models.py          # Generate confusion matrices, ROC curves, bar charts
├── data/
│   ├── copd_binary_labels.csv      # COPD dataset labels (source label CSVs)
│   ├── pneumonia_binary_labels.csv
│   ├── sound_labels.csv
│   └── kauh_parsed.csv
├── saved_models/                   # Trained weights (not in repo — too large)
│   ├── copd_opera_mlp.pt
│   ├── pneumonia_opera_mlp.pt
│   └── sound_opera_mlp_3class.pt
├── outputs/                        # Evaluation figures (not in repo — generated)
└── DATASET/                        # Raw audio files (not in repo — too large)
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv triage_env
triage_env\Scripts\activate      # Windows
source triage_env/bin/activate   # Linux/Mac

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone OPERA (required for embedding extraction)
git clone https://github.com/evelyn0414/OPERA.git
pip install pytorch-lightning torchmetrics efficientnet-pytorch timm torchlibrosa huggingface-hub

# 5. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Running the Pipeline

```bash
# Step 1 — Build label CSVs from raw datasets
python scripts/fix_kauh_parser.py
python scripts/convert_coughvid_webm.py   # only if using COUGHVID .webm files
python scripts/build_label_csvs.py

# Step 2 — Extract OPERA-CT embeddings (one-time, ~2-3 hours)
python scripts/extract_opera_embeddings.py

# Step 3 — Train models
python scripts/train_binary_agent.py       # set DISEASE='COPD', then 'Pneumonia'
python scripts/train_pneumonia_cv.py       # 5-fold CV for Pneumonia (recommended)
python scripts/train_sound_3class.py       # 3-class sound classifier

# Step 4 — Evaluate (generates figures in outputs/)
python scripts/evaluate_models.py

# Step 5 — Run the Streamlit web app
streamlit run app.py
```

**Pipeline from Python:**
```python
from pipeline.triage_graph import run_triage

result = run_triage(
    patient_info={
        "age": 58, "gender": "male",
        "symptoms": ["difficulty breathing", "wheezing"],
        "fever_muscle_pain": False, "respiratory_condition": True,
        "cough_detected": 0.8, "dyspnea": True,
        "wheezing": True, "congestion": False,
    },
    cough_audio_path="path/to/audio.wav",
    lung_audio_path="path/to/lung.wav",   # empty string for Tier 1
)
print(result["triage_decision"])
```

---

## Hardware

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- CUDA 12.6 / PyTorch cu121
- OPERA-CT embedding: batch_size=16, float32
- Training: AdamW + CosineAnnealingLR + FocalLoss + WeightedRandomSampler

---

## Files Not in Repository

| Path | Reason |
|---|---|
| `DATASET/` | Raw audio files — too large |
| `OPERA/` | 514 MB cloned repo — install separately |
| `saved_models/` | Trained model weights |
| `data/opera_embeddings/` | 11,579 `.npy` embedding files |
| `outputs/` | Generated evaluation figures |
| `data/*_with_embeddings.csv` | Generated split CSVs |

---

## Project Status

- [x] Dataset parsing — ICBHI, KAUH, COUGHVID, HF Lung V1
- [x] OPERA-CT embedding extraction — 11,579 embeddings
- [x] COPD Agent — AUROC 0.995, F1 0.947, Recall 0.959
- [x] Pneumonia Agent — AUROC 0.984, F1 0.869 (5-fold CV)
- [x] Sound Classifier — 3-class, F1 0.594
- [x] Rule Engine — deterministic BTS/GOLD/GINA guidelines
- [x] LangGraph pipeline — two-tier routing
- [x] Session monitoring — SQLite + deterioration detection
- [x] Streamlit web app — two-tier UI
- [ ] Baseline comparisons (MFCC+MLP, BEATs+linear)
- [ ] Paper writing + final evaluation
