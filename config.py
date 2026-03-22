"""
config.py — Central settings for Multimodal Respiratory Triage AI
All paths, hyperparameters, and constants defined here.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── COUGHVID settings ─────────────────────────────────────────────────────────
COUGHVID_METADATA          = "./data/coughvid_metadata_compiled.csv"   # built from JSONs
COUGHVID_AUDIO_DIR         = "./DATASET/COUGHVID_DATASET"
COUGHVID_LABELS_CSV        = "./data/coughvid_labels.csv"          # XGBoost features
COUGHVID_SPEC_LABELS_CSV   = "./data/coughvid_spec_labels.csv"     # EfficientNet spectrograms
COUGHVID_SPEC_DIR          = "./data/coughvid_spectrograms"
COUGHVID_SAMPLES_PER_CLASS = 0           # 0 = use ALL available samples (no cap)
COUGHVID_SR                = 22050       # standard rate for cough classification
COUGHVID_DURATION          = 4.0         # 4s — PMC9034264 standard
# ceil(22050 * 4 / 512) = 173 frames  (resized to 224×224 in Dataset)
COUGHVID_SPEC_FRAMES       = 173
# COVID-19 merged into Symptomatic — symptoms overlap, binary is more robust
COUGHVID_CLASSES           = ['Healthy', 'Symptomatic']
COUGHVID_LABEL_REMAP       = {
    'healthy'     : 'Healthy',
    'symptomatic' : 'Symptomatic',
    'COVID-19'    : 'Symptomatic',   # COVID symptoms overlap with symptomatic
}

# ── ICBHI settings ────────────────────────────────────────────────────────────
ICBHI_ROOT        = "./DATASET/ICBHI_DATASET/Respiratory_Sound_Database/Respiratory_Sound_Database"
ICBHI_AUDIO_DIR   = f"{ICBHI_ROOT}/audio_and_txt_files"
ICBHI_DIAGNOSIS   = f"{ICBHI_ROOT}/patient_diagnosis.csv"

# Map ICBHI raw labels → our 5-class disease set (None = skip disease label)
ICBHI_DISEASE_MAP = {
    'Healthy':        'Normal',
    'COPD':           'COPD',
    'Pneumonia':      'Pneumonia',
    'Asthma':         'Asthma',
    # URTI, LRTI, Bronchiectasis, Bronchiolitis → disease_label=-1 (still used for sound head)
}

# ── KAUH settings ─────────────────────────────────────────────────────────────
KAUH_AUDIO_DIR = "./DATASET/KAUH_DATASET/Audio Files"

# Filename format: BP108_COPD,E W,P R L ,63,M.wav
KAUH_DISEASE_MAP = {
    'N':             'Normal',
    'COPD':          'COPD',
    'Asthma':        'Asthma',
    'Heart Failure': 'Heart_Failure',
    'Pneumonia':     'Pneumonia',
    # Lung Fibrosis, Pleural Effusion, etc. → disease_label=-1
}
KAUH_SOUND_MAP = {
    'N':      'Normal',
    'E W':    'Wheeze',
    'I W':    'Wheeze',
    'B W':    'Wheeze',
    'W':      'Wheeze',
    'C':      'Crackle',
    'Crep':   'Crackle',
    'E Crep': 'Crackle',
    'I Crep': 'Crackle',
}

# ── HF Lung V1 settings ───────────────────────────────────────────────────────
HF_LUNG_MANIFEST = "./data/hf_lung_manifest.csv"  # generated previously

# ── Combined multi-task dataset ───────────────────────────────────────────────
SPEC_DIR             = "./data/spectrograms"
MULTITASK_LABELS_CSV = "./data/multitask_labels.csv"

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION    = 8.0        # seconds — 8s captures full breath cycles (literature standard)
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 1024       # 1024 gives better temporal resolution for short events
N_MFCC      = 40

# Fixed spectrogram width (frames): ceil(16000 * 8 / 512) = 250
SPEC_TIME_FRAMES = 250

# ── Disease and sound classes ─────────────────────────────────────────────────
LUNG_DISEASE_CLASSES = ['Normal', 'COPD', 'Pneumonia', 'Asthma', 'Heart_Failure']  # 5 classes
LUNG_SOUND_CLASSES   = ['Normal', 'Crackle', 'Wheeze', 'Both']                     # 4 classes

NUM_DISEASE_CLASSES = len(LUNG_DISEASE_CLASSES)  # 5
NUM_SOUND_CLASSES   = len(LUNG_SOUND_CLASSES)    # 4

# ── Training settings ─────────────────────────────────────────────────────────
BATCH_SIZE      = 16        # GTX 1650 4 GB VRAM
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
EPOCHS_MULTITASK = 40       # total epochs; first half: frozen backbone
PATIENCE        = 8         # early stopping patience

# ── XGBoost settings ──────────────────────────────────────────────────────────
XGB_N_ESTIMATORS  = 300
XGB_MAX_DEPTH     = 6
XGB_LEARNING_RATE = 0.1

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVED_MODELS_DIR = "./saved_models"
OUTPUTS_DIR      = "./outputs"