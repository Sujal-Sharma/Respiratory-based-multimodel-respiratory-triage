---
license: cc-by-nc-4.0
task_categories:
  - audio-classification
language:
  - en
size_categories:
  - 100K<n<1M
tags:
  - respiratory-sound
  - medical-audio
  - lung-sound
  - covid-19
  - cough-detection
---

# Resp-229K: Respiratory Sound Dataset

> A Large-Scale Respiratory Sound Dataset for Training and Evaluation

---

## 📖 Overview

**Resp-229K** is a comprehensive respiratory sound dataset containing **229,101 valid audio files** with a total duration of over **407 hours**. This dataset is curated for training the **Resp-Agent** system - an intelligent respiratory sound analysis and generation framework.

## 📊 Dataset Statistics

| Split | Valid Files | Total Duration | Avg Duration | Max Duration |
|-------|-------------|----------------|--------------|--------------|
| **Train** | 196,654 | 340h 49m 38s | 6.24s | 86.20s |
| **Valid** | 16,931 | 30h 57m 57s | 6.58s | 71.05s |
| **Test** | 15,516 | 36h 3m 43s | 8.37s | 30.00s |
| **Total** | **229,101** | **407h 51m 18s** | **6.41s** | **86.20s** |

### Sample Rate Distribution

| Sample Rate | Files | Percentage |
|-------------|-------|------------|
| 48000 Hz | 196,282 | 85.67% |
| 44100 Hz | 28,686 | 12.52% |
| 8000 Hz | 2,657 | 1.16% |
| 16000 Hz | 824 | 0.36% |
| 4000 Hz | 312 | 0.14% |
| Other | 340 | 0.15% |

## 📋 Dataset Sources and Licenses

| Dataset | Role | Institution / Source | License |
|---------|------|----------------------|---------|
| [UK COVID-19](https://zenodo.org/records/10043978) | Train / Valid | UK Health Security Agency (UKHSA) | OGL 3.0 |
| [COUGHVID](https://zenodo.org/records/4048312) | Test | École Polytechnique Fédérale de Lausanne (EPFL) | CC BY 4.0 |
| [ICBHI](https://bhichallenge.med.auth.gr) | Train / Valid | ICBHI Organizers | CC0 |
| [HF Lung V1](https://gitlab.com/techsupportHF/HF_Lung_V1) | Train / Valid | Heroic-Faith Medical Science | CC BY 4.0 |
| [KAUH](https://data.mendeley.com/datasets/jwyy9np4gv/3) | Test | King Abdullah University Hospital | CC BY 4.0 |
| [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | Train / Valid | Shanghai Jiao Tong University | CC BY 4.0 |

## 📁 Dataset Structure

```
dataset.zip
├── train/          # 196,654 training samples
├── valid/          # 16,931 validation samples
└── test/           # 15,516 test samples
```

## 📝 Audio Description File

The dataset includes an AI-generated description file for respiratory sounds:

**File**: `audio_descriptions.jsonl` (237,786 entries)

| Field | Description |
|-------|-------------|
| `audio_filename` | Original audio file name |
| `description` | Detailed AI-generated description of respiratory characteristics |
| `disease` | Associated disease label |

**Sample entry:**
```json
{"audio_filename": "172_1b3_Al_mc_AKGC417L.wav", "description": "Respiratory sounds were assessed at the anterior left recording location using the AKG C417L microphone. No crackles or wheezes were detected in the first six cycles...", "disease": "COPD"}
```

The descriptions include:
- Recording location and equipment
- Presence/absence of crackles and wheezes
- Timing information for each respiratory cycle
- Overall clinical observations
- High-confidence LLM artifact records removed (placeholder text, prompt/tag leakage, and clear label-context conflicts)

## 🔧 Usage

**1. Download and extract:**
```python
from huggingface_hub import hf_hub_download

# Download dataset
hf_hub_download(
    repo_id="AustinZhang/resp-agent-dataset",
    filename="dataset.zip",
    repo_type="dataset",
    local_dir="./data"
)
```

```bash
# Extract
unzip ./data/dataset.zip -d ./data/
```

**2. Configure paths in `config.yaml`:**
```yaml
data:
  train_root: "./data/dataset/train"
  val_root: "./data/dataset/valid"
  test_root: "./data/dataset/test"
```

## 📝 Paper

**[Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](https://openreview.net/forum?id=ZkoojtEm3W&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))** (ICLR 2026)

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhangresp,
  title={Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis},
  author={ZHANG, Pengfei and Xie, Tianxin and Yang, Minghao and Liu, Li},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

## 📄 License

This curated dataset is released under **CC BY-NC 4.0** for academic research purposes. Individual source datasets retain their original licenses as listed above.

## 🔗 Related Resources

- **GitHub Repository**: [zpforlove/Resp-Agent](https://github.com/zpforlove/Resp-Agent)
- **Model Weights**: [AustinZhang/resp-agent-models](https://huggingface.co/AustinZhang/resp-agent-models)

---

##  Contact

**Email**: [pzhang176@connect.hkust-gz.edu.cn](mailto:pzhang176@connect.hkust-gz.edu.cn)