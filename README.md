---
title: RespiTriage AI
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Multimodal Respiratory Triage Using Agentic AI

An agentic AI system for respiratory disease triage using the **OPERA-CT** respiratory foundation model (NeurIPS 2024). The system processes cough/breathing audio and patient symptoms through specialist AI agents, producing structured clinical triage decisions via deterministic rule-based reasoning.

**Targeted diseases:** COPD · Pneumonia · Abnormal Lung Sounds

---

## Live Demo

**Hugging Face Spaces:** https://huggingface.co/spaces/SujalSha/respitriage

> Run locally: `python server.py` → open `http://127.0.0.1:5000`

---

## Architecture

```
Audio Input
    └── OPERA-CT Encoder (768-dim embeddings, HT-SAT backbone)
            ├── COPD Agent        (BinaryMLP, 768→256→64→2)
            ├── Pneumonia Agent   (BinaryMLP, 768→256→64→2)
            └── Sound Agent       (MLP, 768→512→256→64→3)  [Tier 2 only]

Voice Agent     (MFCC + jitter + shimmer + HNR from vowel phonation)
Symptom Agent   (CAT-style clinical scoring, no ML)
LLM Validator   (Groq llama-3.3-70b — free-text symptom validation)

LangGraph StateGraph
    analyze_symptoms → run_voice_agent → run_cough_drift
        → run_copd_agent → run_pneumonia_agent
        → [analyze_lung — Tier 2 only]
        → compute_longitudinal → apply_rules → record_session → END

Rule Engine (deterministic BTS/GOLD/GINA clinical guidelines)
    → Triage Decision: diagnosis · severity · confidence · referral urgency

Session Store (SQLite + linear regression deterioration detection)
3-Signal Fusion: 50% symptom + 35% voice + 15% cough drift
```

---

## Two-Tier Design

**Tier 1 — Patient Self-Screening**
- Patient uploads vowel recording ("Ahhh") + optional cough recording
- Fills symptom form + optional free-text symptoms (LLM-validated)
- COPD Agent + Pneumonia Agent + Voice Agent + Symptom Agent run
- Longitudinal risk score tracks health trajectory over time
- Rule engine produces: severity, diagnosis, recommended action

**Tier 2 — Clinician Assessment**
- Doctor uploads stethoscope lung sound recording for a specific patient
- Sound Agent also runs (Normal / Crackle / Wheeze)
- Doctor can add free-text symptoms — LLM validates before scoring
- All signals fused for final triage decision

---

## Model Results

| Model | Task | Accuracy | Macro F1 | Recall | AUROC |
|---|---|---|---|---|---|
| COPD Agent (OPERA-MLP) | COPD vs Normal | 94.8% | 0.947 | 0.959 | 0.995 |
| Pneumonia Agent (OPERA-MLP, 5-fold CV) | Pneumonia vs Normal | 97.7% | 0.869 | 0.750 | 0.984 |
| Sound Classifier (OPERA-MLP, 3-class) | Normal / Crackle / Wheeze | 60.5% | 0.594 | — | — |

**Notes:**
- Pneumonia evaluated via 5-fold stratified cross-validation (only 52 positive samples)
- Sound classifier "Both" class merged into Crackle due to insufficient samples
- All models use pre-computed OPERA-CT embeddings — OPERA runs once at extraction time

---

## Datasets

| Dataset | Task | Samples |
|---|---|---|
| ICBHI Respiratory Sound Database | COPD binary + lung sound labels | 6,898 recordings |
| KAUH Clinical Lung Recordings | COPD + Pneumonia + sound labels | 147 recordings |
| COUGHVID | COPD binary (cough audio) | ~7,500 recordings |
| HF Lung V1 | Lung sound classification | 7,503 recordings |

Total OPERA embeddings extracted: **11,579 `.npy` files** (768-dim each)

---

## Project Structure

```
├── server.py                           # Flask web server (main entry point)
├── Dockerfile                          # HF Spaces deployment
├── requirements.txt
├── agents/
│   ├── copd_agent.py                   # COPD binary specialist (OPERA + MLP)
│   ├── pneumonia_agent.py              # Pneumonia binary specialist (OPERA + MLP)
│   ├── sound_agent.py                  # Lung sound classifier — Tier 2 (3-class)
│   ├── voice_agent.py                  # Voice biomarker extractor (MFCC, jitter, shimmer)
│   ├── symptom_agent.py                # CAT-style clinical symptom scorer
│   └── session_agent.py                # Session recording + deterioration alerts
├── models/
│   ├── opera_encoder.py                # Batched OPERA-CT encoder (librosa mel preprocessing)
│   ├── mlp_classifier.py               # BinaryMLP + SoundMLP + FocalLoss
│   └── embedding_dataset.py            # PyTorch Dataset for pre-computed embeddings
├── pipeline/
│   ├── triage_graph.py                 # LangGraph StateGraph orchestration
│   ├── rule_engine.py                  # Deterministic BTS/GOLD/GINA rule engine
│   └── longitudinal.py                 # 3-signal fusion + drift detection
├── database/
│   ├── auth_store.py                   # User auth + patient profiles (SQLite)
│   └── session_store.py                # Session store + deterioration detection
├── utils/
│   └── symptom_validator.py            # Groq LLM free-text symptom validator
├── web/
│   ├── templates/
│   │   ├── login.html                  # Animated landing page + auth
│   │   ├── base.html                   # Shared sidebar layout (mobile responsive)
│   │   ├── patient.html                # Patient self-screening portal
│   │   ├── doctor.html                 # Doctor patient list portal
│   │   └── doctor_patient.html         # Doctor patient detail + Tier 2 assessment
│   └── static/
├── scripts/                            # Training + evaluation scripts
├── saved_models/                       # Trained model weights (4.7 MB)
│   ├── copd_opera_mlp.pt
│   ├── pneumonia_opera_mlp.pt
│   ├── cough_opera_mlp.pt
│   └── sound_opera_mlp_3class.pt
├── data/                               # Label CSVs (embeddings excluded from repo)
└── outputs/                            # Evaluation figures (excluded from repo)
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/SujalSharma123/CAPSTONE_JOURNAL.git
cd CAPSTONE_JOURNAL

# 2. Create virtual environment
python -m venv triage_env
triage_env\Scripts\activate      # Windows
source triage_env/bin/activate   # Linux/Mac

# 3. Install PyTorch with CUDA (local development)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Clone OPERA (required for model loading only)
git clone https://github.com/evelyn0414/OPERA.git
pip install pytorch-lightning torchmetrics efficientnet-pytorch timm torchlibrosa huggingface-hub

# 6. Set environment variables — create a .env file:
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here

# 7. Run the web app
python server.py
# Open http://127.0.0.1:5000
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at huggingface.co → Docker SDK
2. Add repository secrets: `GROQ_API_KEY`, `SECRET_KEY`
3. Enable Persistent Storage and mount at `/data`
4. Upload via HuggingFace Hub API (avoids git LFS issues with `.pt` files):
```python
from huggingface_hub import HfApi
api = HfApi(token='your_hf_token')
api.upload_folder(folder_path='.', repo_id='your-username/space-name', repo_type='space',
                  ignore_patterns=['*.pyc','__pycache__','triage_env','OPERA','data/opera_embeddings'])
```

---

## Training Pipeline (reproducing from scratch)

```bash
python scripts/fix_kauh_parser.py
python scripts/build_label_csvs.py
python scripts/extract_opera_embeddings.py   # ~2-3 hours
python scripts/train_binary_agent.py          # DISEASE='COPD' then 'Pneumonia'
python scripts/train_pneumonia_cv.py
python scripts/train_sound_3class.py
python scripts/evaluate_models.py
```

---

## Hardware

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- CUDA 12.6 / PyTorch cu121
- OPERA-CT inference: batch_size=16, float16
- Training: AdamW + CosineAnnealingLR + FocalLoss + WeightedRandomSampler

---

## Files Not in Repository

| Path | Reason |
|---|---|
| `DATASET/` | Raw audio files — too large |
| `OPERA/` | 514 MB cloned repo — install separately |
| `data/opera_embeddings/` | 11,579 `.npy` embedding files |
| `data/sessions.db` | Runtime database — generated on first run |
| `outputs/` | Generated evaluation figures |
| `data/*_with_embeddings.csv` | Generated split CSVs |
| `.env` | API keys |

---

## Project Status

- [x] Dataset parsing — ICBHI, KAUH, COUGHVID, HF Lung V1
- [x] OPERA-CT embedding extraction — 11,579 embeddings
- [x] COPD Agent — AUROC 0.995, F1 0.947, Recall 0.959
- [x] Pneumonia Agent — AUROC 0.984, F1 0.869 (5-fold CV)
- [x] Sound Classifier — 3-class, F1 0.594
- [x] Voice Agent — MFCC + jitter + shimmer + HNR longitudinal tracking
- [x] Rule Engine — deterministic BTS/GOLD/GINA guidelines
- [x] LangGraph pipeline — two-tier agentic routing
- [x] Longitudinal monitoring — 3-signal fusion + drift detection
- [x] Session store — SQLite + HF persistent storage
- [x] Flask web app — patient + doctor portals
- [x] Animated landing page — Bootstrap + AOS
- [x] Mobile responsive — hamburger sidebar
- [x] LLM symptom validator — Groq free-text validation
- [x] Docker deployment — Hugging Face Spaces
