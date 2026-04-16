# Multimodal Respiratory Triage Using Agentic AI

An agentic AI system for respiratory disease triage using the **OPERA-CT** respiratory foundation model (NeurIPS 2024). The system processes cough/breathing audio and patient symptoms through specialist AI agents, producing structured clinical triage decisions via deterministic rule-based reasoning.

**Targeted diseases:** COPD · Pneumonia · Abnormal Lung Sounds

---

## Live Demo

> Run locally: `python server.py` → open `http://127.0.0.1:5000`

**Demo credentials:** `doctor` / `doctor123`

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
├── requirements.txt
├── agents/
│   ├── copd_agent.py                   # COPD binary specialist (OPERA + MLP)
│   ├── pneumonia_agent.py              # Pneumonia binary specialist (OPERA + MLP)
│   ├── sound_agent.py                  # Lung sound classifier — Tier 2 (3-class)
│   ├── voice_agent.py                  # Voice biomarker extractor (MFCC, jitter, shimmer)
│   ├── symptom_agent.py                # CAT-style clinical symptom scorer
│   └── session_agent.py                # Session recording + deterioration alerts
├── models/
│   ├── opera_encoder.py                # Batched GPU OPERA-CT encoder
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
│       ├── css/
│       └── js/
├── scripts/                            # Training + evaluation scripts
│   ├── build_label_csvs.py
│   ├── extract_opera_embeddings.py
│   ├── train_binary_agent.py
│   ├── train_pneumonia_cv.py
│   ├── train_sound_3class.py
│   └── evaluate_models.py
├── data/                               # Label CSVs (embeddings excluded from repo)
├── saved_models/                       # Trained weights (excluded from repo)
└── outputs/                            # Evaluation figures (excluded from repo)
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Sujal-Sharma/Respiratory-based-multimodel-respiratory-triage.git
cd Respiratory-based-multimodel-respiratory-triage

# 2. Create virtual environment
python -m venv triage_env
triage_env\Scripts\activate      # Windows
source triage_env/bin/activate   # Linux/Mac

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Clone OPERA (required for embedding extraction only)
git clone https://github.com/evelyn0414/OPERA.git
pip install pytorch-lightning torchmetrics efficientnet-pytorch timm torchlibrosa huggingface-hub

# 6. Set environment variables
# Create a .env file:
GROQ_API_KEY=your_groq_api_key_here

# 7. Run the web app
python server.py
# Open http://127.0.0.1:5000
```

---

## Training Pipeline (if reproducing from scratch)

```bash
# Step 1 — Parse datasets + build label CSVs
python scripts/fix_kauh_parser.py
python scripts/convert_coughvid_webm.py   # if using COUGHVID .webm files
python scripts/build_label_csvs.py

# Step 2 — Extract OPERA-CT embeddings (one-time, ~2-3 hours)
python scripts/extract_opera_embeddings.py

# Step 3 — Train models
python scripts/train_binary_agent.py       # DISEASE='COPD' then 'Pneumonia'
python scripts/train_pneumonia_cv.py       # 5-fold CV for Pneumonia
python scripts/train_sound_3class.py       # 3-class sound classifier

# Step 4 — Evaluate
python scripts/evaluate_models.py
```

---

## Using the Pipeline Directly

```python
from pipeline.triage_graph import run_triage

result = run_triage(
    patient_info={
        "age": 58, "gender": "male",
        "symptoms": ["difficulty breathing", "wheezing"],
        "fever_muscle_pain": False,
        "respiratory_condition": True,
        "cough_detected": 0.8,
        "dyspnea": True,
        "dyspnea_level": 2,
        "wheezing": True,
        "congestion": False,
        "chest_tightness": 2,
        "sleep_quality": 1,
        "energy_level": 2,
        "sputum": 1,
    },
    cough_audio_path="path/to/cough.wav",
    vowel_audio_path="path/to/vowel.wav",
    lung_audio_path="",          # empty for Tier 1, stethoscope path for Tier 2
    patient_id="patient_1",
)
print(result["triage_decision"])
# {'diagnosis': ..., 'severity': ..., 'referral_urgency': ..., 'reasoning': ...}
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
| `saved_models/` | Trained model weights |
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
- [x] Session store — SQLite + deterioration alerts
- [x] Flask web app — replaced Streamlit
- [x] Animated landing page — Bootstrap + Tailwind + AOS
- [x] Mobile responsive — hamburger sidebar for patient/doctor portals
- [x] LLM symptom validator — Groq free-text validation in both portals
- [ ] Deployment — Railway + Turso
