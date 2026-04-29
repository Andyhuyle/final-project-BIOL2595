# Weakly Supervised Multimodal Learning for Prostate Cancer Severity Assessment

**Course:** AIH 2025 (Spring 2026) — BIOL 2595  
**Author:** Andy (Huy) Le  
**Institution:** Brown University  
**Target Journal:** JAMIA  

---

## Overview

This project develops a weakly supervised multimodal contrastive learning framework to align histopathology biopsy images (PANDA dataset) with structured electronic health record data (MIMIC IV) for prostate cancer severity assessment. Because the two datasets come from different patient populations and lack direct correspondence, traditional supervised multimodal learning is not feasible. Instead, PSA-derived severity labels (low / moderate / high) serve as the alignment signal — samples with similar severity are encouraged to share similar latent representations without requiring any patient-level pairing.

A late fusion baseline (two independently trained classifiers whose predictions are averaged at inference) is included for comparison.

---

## Repository Structure

```
final-project-BIOL2595/
├── multimodal_model/
│   ├── multimodal_contrastive.py         # Main contrastive learning model (train.py)
│   └── late_fusion_baseline.py           # Late fusion baseline model
├── scripts/
│   ├── extract_mimic_pca.sh              # Extract MIMIC IV features for PSA cohort
│   ├── build_control_cohort.py           # Build balanced PCa / control cohort
│   ├── build_ehr_features.py             # Build normalized EHR feature matrix
│   ├── psa_severity_distribution.py      # Assign PSA severity classes
│   ├── downsample.py                     # Balance both datasets to equal class sizes
│   ├── evaluate_embeddings.py            # Linear probe + retrieval metrics
│   ├── shap_ehr_analysis.py              # SHAP feature importance analysis
│   ├── umap_visualization.py             # UMAP plots of shared latent space
│   └── generate_table1_image.py          # Table 1
├── logs/                                 # SLURM job logs
├── sbatch_multimodal_contrastive.sh      # SLURM submission for contrastive model
├── sbatch_late_fusion.sh                 # SLURM submission for late fusion
└── README.md
```

---

## Data

### MIMIC IV (EHR)
- **Source:** `/oscar/data/shared/ursa/mimic-iv` (version 3.1)
- **Cohort:** Patients with ≥1 PSA lab test in `labevents.csv` (n=16,626)
- **After filtering** (requires procedures AND medications): n=12,701
- **Features:** PSA max, PSA order count, procedure count, distinct medication count, admission LOS, age
- **Severity labels:** PSA-derived using AUA/EAU clinical thresholds
  - Low: PSA < 4.0 ng/mL (n=9,627, 75.8%)
  - Moderate: PSA 4–20 ng/mL (n=2,386, 18.8%)
  - High: PSA > 20 ng/mL (n=688, 5.4%)

### PANDA (Histopathology Images)
- **Source:** `/oscar/data/shared/ursa/kaggle_panda/`
- **Total images:** 10,616 biopsy TIFFs
- **Severity mapping from ISUP grade:**
  - Low: ISUP 0–1 / negative / 3+3 (n=5,558)
  - Moderate: ISUP 2–3 / 3+4 / 4+3 (n=2,585)
  - High: ISUP 4–5 / 4+4 and above (n=2,473)
- **Data providers:** Karolinska Institute (51.4%), Radboud UMC (48.6%)

### Balanced Training Set
Both modalities downsampled to match the binding constraint (high severity, n=688):
- High: 688 | Moderate: 1,376 | Low: 1,376
- **Total: 3,440 samples per modality**

---

## Pipeline

```
MIMIC IV labevents.csv
        ↓
extract_mimic_pca.sh          → mimic_pca_cohort_features.csv
        ↓
build_ehr_features.py         → ehr_features.csv
        ↓
psa_severity_distribution.py  → psa_severity_classes.csv
        ↓
downsample.py                 → ehr_feature_matrix_balanced.csv
                                ehr_severity_balanced.csv
                                panda_balanced.csv
        ↓
multimodal_contrastive.py     → multimodal_model.pt
                                embeddings.csv
        ↓
evaluate_embeddings.py        → results_summary.csv
shap_ehr_analysis.py          → shap_feature_importance.csv + plots
umap_visualization.py         → umap_panel_figure.png
late_fusion_baseline.py       → late_fusion_results.csv
generate_table1_image.py      → table1.png
```

---

## Model Architecture

### Contrastive Model
- **Image encoder:** ResNet18 backbone + 2-layer MLP projection head → 128-d embedding
  - Input: 8 random patches (256×256) per biopsy, read at pyramid level 2
  - Mean pooling across patches
- **EHR encoder:** BatchNorm → 3-layer MLP → 128-d embedding
  - Input: 6 normalized clinical features
- **Loss:** Supervised contrastive loss — same-severity pairs attract, different-severity pairs repel
- **Training:** 25 epochs, Adam (lr=1e-4), CosineAnnealingLR, WeightedRandomSampler

### Late Fusion Baseline
- Image classifier: ResNet18 + classification head, trained on PANDA only
- EHR classifier: BatchNorm MLP, trained on MIMIC IV only
- Fusion: average softmax probabilities at inference

---

## Results

---

## How to Run

### 1. Extract EHR features
```bash
bash extract_mimic_pca.sh /oscar/data/shared/ursa/mimic-iv ./extracted

python build_ehr_features.py \
    --psa     ./extracted/psa_severity_classes.csv \
    --mimic   /oscar/data/shared/ursa/mimic-iv \
    --version 3.1 \
    --out     ./extracted
```

### 2. Downsample both datasets
```bash
python downsample.py
```

### 3. Train contrastive model
```bash
sbatch sbatch_multimodal_contrastive.sh
```

### 4. Evaluate
```bash
python evaluate_embeddings.py
python shap_ehr_analysis.py
python umap_visualization.py
```

### 5. Late fusion baseline
```bash
sbatch sbatch_late_fusion.sh
```

### 6. Generate Table 1
```bash
python generate_table1_image.py \
    --ehr   ./extracted/ehr_features.csv \
    --panda /oscar/data/shared/ursa/kaggle_panda/train.csv \
    --out   ./outputs
```

---

## Requirements

```bash
pip install torch torchvision tifffile tqdm pandas numpy scipy \
            scikit-learn matplotlib shap umap-learn
```

Python 3.9 | PyTorch 2.8 | CUDA 12.8 | Oscar HPC (Brown University)

---

## Key Outputs

| File | Description |
|---|---|
| `outputs/multimodal_model.pt` | Trained contrastive model checkpoint |
| `outputs/embeddings.csv` | 128-d embeddings for all 3,440×2 samples |
| `outputs/results_summary.csv` | All evaluation metrics |
| `outputs/shap/shap_feature_importance.csv` | EHR feature importance ranking |
| `outputs/umap_panel_figure.png` | Shared latent space visualization |
| `outputs/table1.png` | Publication-quality Table 1 |
| `outputs/late_fusion/late_fusion_results.csv` | Baseline comparison metrics |

---

## Limitations

- EHR and image cohorts are from different patient populations — alignment is achieved through shared severity labels only (unpaired multimodal learning)
- PSA-derived severity is a proxy for Gleason grade; some patients may be misclassified (e.g. elevated PSA from BPH rather than cancer)
- MIMIC IV is a single-center dataset (Beth Israel Deaconess Medical Center); PANDA images are from Karolinska and Radboud — generalizability to other populations is unknown
- Moderate severity class has lower retrieval accuracy (38.5%), consistent with the clinical heterogeneity of PSA 4–20 ng/mL and ISUP grades 2–3
