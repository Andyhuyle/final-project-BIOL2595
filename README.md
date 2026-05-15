# Weakly Supervised Multimodal Contrastive Learning for Prostate Cancer Severity

**BIOL 1595/2595 Final Project**

This project investigates whether a weakly supervised multimodal contrastive learning approach — aligning histopathology images (PANDA dataset) with structured EHR data (MIMIC IV) — outperforms a late-fusion baseline at predicting prostate cancer severity. The two datasets are never patient-matched; alignment is achieved at the severity-class level using PSA-derived labels as the weak supervision signal.

---

## Severity Schema

Both modalities use the same three-class scheme:

| Class | Label | PANDA (ISUP grade) | EHR (PSA) |
|---|---|---|---|
| 0 | Low | ISUP 0–1 | PSA < 4.0 |
| 1 | Moderate | ISUP 2–3 | PSA 4–20 |
| 2 | High | ISUP 4–5 | PSA > 20 |

---

## Repository Structure

```
.
├── mimic_eda.py              # Step 1a — PSA label extraction from MIMIC IV
├── pandas_eda.py             # Step 1b — PANDA dataset exploration
├── build_ehr_features.py     # Step 2  — EHR feature matrix construction
├── panda_downsample.py       # Step 3  — Class-balanced downsampling
├── contrastive_model.py      # Step 4a — Multimodal contrastive model (main)
├── late_fusion_baseline.py   # Step 4b — Late fusion baseline
├── evaluate_embeddings.py    # Step 5  — Linear probe + cross-modal retrieval
├── shap_ehr_analysis.py      # Step 6  — SHAP feature importance (EHR)
├── generate_table1.py        # Step 7  — Cohort characteristics table
├── plot_auroc.py             # Step 8  — Combined ROC curve figure (2×2 grid)
└── plot_curves.py            # Step 9  — Training loss curve figures
```

---

## Pipeline

Run the scripts in the order below. Each step depends on the outputs of the previous one.

### Step 1 — Exploratory Data Analysis

**`mimic_eda.py`** — Loads the cleaned PSA lab CSV, computes the max PSA per patient, assigns severity classes, and saves `psa_severity_classes.csv`.

```bash
python mimic_eda.py
# Edit PSA_CSV and OUT_DIR at the top of the file before running
```

**`pandas_eda.py`** — Prints ISUP grade and Gleason score distributions from the PANDA `train.csv`. No outputs; informational only.

```bash
python pandas_eda.py
```

---

### Step 2 — Build EHR Feature Matrix

**`build_ehr_features.py`** — Pulls six clinical features per patient from MIMIC IV source files, imputes missing values with per-feature medians, min-max normalizes to [0, 1], and writes three output files.

Features extracted: `psa_max`, `psa_order_count`, `procedure_count`, `distinct_med_count`, `los_days`, `anchor_age`.

```bash
python build_ehr_features.py \
    --psa   /path/to/psa_severity_classes.csv \
    --mimic /path/to/mimic-iv \
    --version 3.1 \
    --out   /path/to/extracted/
```

Outputs:
- `ehr_features.csv` — raw + normalized features + labels (full cohort)
- `ehr_feature_matrix.csv` — normalized features only (model input)
- `ehr_severity_labels.csv` — `subject_id`, `severity_int`, `severity_class`

---

### Step 3 — Class-Balanced Downsampling

**`panda_downsample.py`** — Creates balanced CSVs for both modalities. The binding constraint is the smallest class across both datasets (EHR high severity, n=688). High is kept in full; moderate and low are downsampled to 2× that count, yielding 3,440 samples per modality.

```bash
python panda_downsample.py
# Edit path constants at the top of the file before running
```

Outputs (written to `extracted/`):
- `panda_balanced.csv` — 3,440 PANDA image rows with severity labels
- `ehr_severity_balanced.csv` — 3,440 EHR severity labels
- `ehr_feature_matrix_balanced.csv` — 3,440 normalized EHR features

---

### Step 4a — Contrastive Model (Main Model)

**`contrastive_model.py`** — Trains a weakly supervised multimodal contrastive model. A ResNet18 image encoder and a 3-layer MLP EHR encoder share a 128-d latent space trained with supervised contrastive loss. Patients are never paired; alignment is class-level only.

Key design choices: `WeightedRandomSampler` for class balance, `CosineAnnealingLR` scheduler, 2-layer MLP projection head on the image encoder, `BatchNorm` on EHR input.

```bash
python contrastive_model.py
# Paths are set via constants near the top of the file
```

Outputs (written to `outputs/contrastive_model/`):
- `contrastive_model.pt` — model checkpoint (includes training history)
- `embeddings.csv` — 128-d embeddings for all samples, both modalities
- `confusion_matrix_contrastive.png`
- `y_true_contrastive.npy`, `y_prob_contrastive.npy` — for ROC curves

---

### Step 4b — Late Fusion Baseline

**`late_fusion_baseline.py`** — Trains the image encoder and EHR encoder independently with cross-entropy (focal) loss, then averages their softmax probabilities at inference. No shared objective, no joint training. This is the direct comparison for the main hypothesis.

```bash
python late_fusion_baseline.py
# Paths are set via constants near the top of the file
```

Outputs (written to `outputs/late_fusion/`):
- `y_true_img.npy`, `y_prob_img.npy`
- `y_true_ehr.npy`, `y_prob_ehr.npy`
- `y_true_fused.npy`, `y_prob_fused.npy`
- `confusion_matrix_*.png`

---

### Step 5 — Evaluate Embeddings

**`evaluate_embeddings.py`** — Evaluates contrastive embedding quality using the standard protocol: frozen embeddings → logistic regression linear probe → 5-fold stratified cross-validation. Also computes cross-modal retrieval accuracy (image→EHR top-1 match by cosine similarity).

```bash
python evaluate_embeddings.py \
    --embeddings /path/to/outputs/embeddings.csv \
    --out        /path/to/outputs/
```

Outputs:
- `confusion_matrix_image_encoder.png`
- `confusion_matrix_ehr_encoder.png`
- `results_summary.csv` — all metrics in one file

---

### Step 6 — SHAP Feature Importance

**`shap_ehr_analysis.py`** — Runs KernelSHAP on a logistic regression trained directly on the 6 raw EHR features (not embeddings) to produce clinically interpretable feature attributions. Explains which features (PSA max, age, LOS, etc.) drive severity predictions for each class.

```bash
python shap_ehr_analysis.py \
    --ehr_matrix  /path/to/ehr_feature_matrix_balanced.csv \
    --ehr_labels  /path/to/ehr_severity_balanced.csv \
    --embeddings  /path/to/embeddings.csv \
    --out         /path/to/outputs/
```

Outputs:
- `shap_summary_bar.png` — mean |SHAP| per feature, stacked by class
- `shap_beeswarm_low/moderate/high.png` — per-class beeswarm plots
- `shap_heatmap.png` — features × classes mean SHAP heatmap
- `shap_feature_importance.csv`
- `shap_values.csv`

---

### Step 7 — Cohort Characteristics Table

**`generate_table1.py`** — Produces the Table 1 cohort characteristics report (TRIPOD/JAMIA format). Continuous variables are reported as median (IQR), categorical as n (%). Kruskal-Wallis and Chi-square p-values are included.

```bash
python generate_table1.py \
    --ehr /path/to/extracted/ehr_features.csv \
    --out /path/to/outputs/
```

Outputs:
- `table1.csv` — machine-readable
- `table1_formatted.txt` — formatted for copy-paste into the paper

---

### Step 8 — ROC Curves

**`plot_auroc.py`** — Generates a 2×2 grid of One-vs-Rest ROC curves comparing all four models: Image Only, EHR Only, Late Fusion, and Contrastive Retrieval. Requires the `.npy` arrays from Steps 4a and 4b.

```bash
python plot_auroc.py
# Edit LATE_FUSION_DIR and CONTRASTIVE_DIR at the top of the file
```

Output: `combined_auroc_curves.png`

---

### Step 9 — Training Curves

**`plot_curves.py`** — Plots training loss curves for the contrastive model and the late fusion baseline (image + EHR separately), plus a combined comparison figure. Reads from `.pt` checkpoints or SLURM log files.

```bash
python plot_curves.py \
    --contrastive_ckpt /path/to/contrastive_model.pt \
    --contrastive_log  /path/to/logs/contrastive_model_*.log \
    --late_fusion_log  /path/to/logs/late_fusion_*.log \
    --out              /path/to/outputs/
```

Outputs:
- `training_curve_contrastive.png`
- `training_curve_late_fusion.png`
- `training_curve_combined.png` ← key figure for paper

> **Note:** Contrastive loss and cross-entropy loss use different scales. Compare trend shapes, not absolute values.

---

### UMAP Visualization (Optional)

**`umap_visualization.py`** — Generates UMAP projections of the shared latent space. Produces four individual plots plus a 2×2 panel figure for the paper. The key diagnostic: if contrastive alignment worked, severity clusters should overlap across modalities in Plot 3, and image/EHR points should be interleaved within clusters in Plot 4.

```bash
python umap_visualization.py \
    --embeddings /path/to/outputs/embeddings.csv \
    --out        /path/to/outputs/
```

Outputs: `umap_image_severity.png`, `umap_ehr_severity.png`, `umap_combined_severity.png`, `umap_combined_modality.png`, `umap_panel_figure.png`

---

## Data Requirements

| Dataset | Source | Notes |
|---|---|---|
| MIMIC IV v3.1 | PhysioNet (credentialed access) | `hosp/` and `icu/` subdirectories required |
| PANDA | Kaggle (`train.csv` + TIFF images) | Pathology Challenge dataset |
| `cleaned_lab.csv` | Derived from MIMIC IV `labevents` | PSA rows pre-extracted |

---

## Dependencies

```
torch torchvision
scikit-learn
pandas numpy scipy
matplotlib
shap
umap-learn
tifffile
```

Install with:
```bash
pip install torch torchvision scikit-learn pandas numpy scipy matplotlib shap umap-learn tifffile
```

---

## Metrics Reported

All models report the same standard set for direct comparison:

- Macro F1, Precision, Recall
- Macro AUROC (One-vs-Rest)
- Per-class F1, Precision, Recall, AUROC
- Top-1 Accuracy, Top-5 Accuracy
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Expected Calibration Error (ECE)
- Confusion matrix

The contrastive model additionally reports cross-modal retrieval accuracy (image→EHR cosine similarity top-1 match).