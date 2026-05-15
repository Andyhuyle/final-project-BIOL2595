"""
contrastive_model.py

Weakly supervised multimodal contrastive learning for prostate cancer severity.

Goal: Align histopathology image representations (PANDA) with structured EHR
representations (MIMIC IV) using shared PSA-derived severity labels as the
weak supervision signal. Patients are NOT paired across datasets — alignment
is achieved at the severity-class level only.

Severity classes (both modalities):
    0 = low      (PSA < 4.0  / ISUP 0-1)
    1 = moderate (PSA 4-20   / ISUP 2-3)
    2 = high     (PSA > 20   / ISUP 4-5)

Inputs (all pre-built by the pipeline scripts):
    ehr_feature_matrix_balanced.csv   normalized [N, 6] EHR features
    ehr_severity_balanced.csv         severity labels for EHR patients
    panda_balanced.csv                balanced PANDA image list with severity

Standard ML metrics reported (consistent with late_fusion_baseline.py):
    - Macro F1, Macro Precision, Macro Recall
    - Macro AUROC (One-vs-Rest, from retrieval probability scores)
    - Per-class F1, Precision, Recall, AUROC
    - Top-1 Accuracy, Top-5 Accuracy (cross-modal retrieval)
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - Calibration: Expected Calibration Error (ECE)
    - Confusion matrix (saved as PNG)
    - Training curves (loss, top-1 acc, positive/negative similarity)

Changes from original:
    - EHR loaded from pre-built balanced feature matrix (not raw CSV)
    - 3-class severity scheme (not 6-class ISUP)
    - ehr_by_severity bucket uses real PSA-derived labels
    - EHR features: psa_max, psa_order_count, procedure_count,
      distinct_med_count, los_days, anchor_age  (6 features, no ICU LOS)
    - EHREncoder has BatchNorm on input for stable training
    - ImageEncoder projection head expanded to 2-layer MLP
    - WeightedRandomSampler for within-epoch class balance
    - CosineAnnealingLR scheduler
    - Standardized cross-modal retrieval evaluation after training
    - Embeddings saved as CSV for UMAP/t-SNE in notebook
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import tifffile
from collections import defaultdict
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    matthews_corrcoef, cohen_kappa_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# CONFIG
# =====================
IMAGE_DIR    = "/oscar/data/shared/ursa/kaggle_panda/train_images"
EHR_MATRIX   = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
EHR_LABELS   = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
PANDA_CSV    = "/oscar/data/class/biol1595_2595/students/hgle/extracted/panda_balanced.csv"
OUT_DIR      = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/contrastive_model"

BATCH_SIZE   = 16
EPOCHS       = 25
PATCH_SIZE   = 256
NUM_PATCHES  = 8
TIFF_LEVEL   = 2       # pyramid level 2 (~1/16 full res) — fast to load
EMB_DIM      = 128
TEMPERATURE  = 0.1     # raised from 0.07 — prevents logit overflow with AMP
NUM_CLASSES  = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SEVERITY MAPPING
# Shared 3-class scheme used by both modalities
# =====================
SEVERITY_MAP    = {"low": 0, "moderate": 1, "high": 2}
SEVERITY_LABELS = {0: "low", 1: "moderate", 2: "high"}
SEVERITY_NAMES  = {0: "Low", 1: "Moderate", 2: "High"}

def isup_to_severity(isup: int) -> int:
    """Map PANDA ISUP grade (0-5) to 3-class severity integer."""
    if isup <= 1: return 0   # low
    if isup <= 3: return 1   # moderate
    return 2                  # high

# =====================
# EHR LOADING
# =====================
print("Loading EHR features...")

ehr_matrix_df = pd.read_csv(EHR_MATRIX)
ehr_labels_df = pd.read_csv(EHR_LABELS, dtype=str)

ehr_features  = ehr_matrix_df.drop(columns=["subject_id"]).values.astype(np.float32)
ehr_severity  = ehr_labels_df["severity_int"].astype(int).values
EHR_DIM       = ehr_features.shape[1]   # 6

print(f"  EHR patients  : {len(ehr_features):,}")
print(f"  Feature dim   : {EHR_DIM}")
print(f"  Severity dist : ", end="")
for sev, name in SEVERITY_LABELS.items():
    print(f"{name}={(ehr_severity == sev).sum()}", end="  ")
print()

# Group EHR feature vectors by severity for weak-supervision pairing
ehr_by_severity = {0: [], 1: [], 2: []}
for feat, sev in zip(ehr_features, ehr_severity):
    ehr_by_severity[sev].append(feat)

# =====================
# IMAGE LOADING
# =====================
def load_tiff_patches(path: str) -> torch.Tensor:
    """
    Load NUM_PATCHES random patches from a TIFF pyramid level.
    """
    with tifffile.TiffFile(path) as tif:
        n_levels = len(tif.pages)
        img = None
        for lvl in range(min(TIFF_LEVEL, n_levels - 1), -1, -1):
            candidate = tif.pages[lvl].asarray()
            h = candidate.shape[0]
            w = candidate.shape[1] if candidate.ndim > 1 else 1
            if h > PATCH_SIZE and w > PATCH_SIZE:
                img = candidate
                break
        if img is None:
            img = tif.pages[0].asarray()

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    elif img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

    H, W, _ = img.shape
    if H <= PATCH_SIZE or W <= PATCH_SIZE:
        dummy = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        dummy[:min(H, PATCH_SIZE), :min(W, PATCH_SIZE)] = \
            img[:min(H, PATCH_SIZE), :min(W, PATCH_SIZE)].astype(np.float32) / 255.0
        patch = np.transpose(dummy, (2, 0, 1))
        return torch.tensor(np.stack([patch] * NUM_PATCHES))

    patches = []
    for _ in range(NUM_PATCHES):
        x = np.random.randint(0, H - PATCH_SIZE)
        y = np.random.randint(0, W - PATCH_SIZE)
        patch = img[x:x+PATCH_SIZE, y:y+PATCH_SIZE].astype(np.float32) / 255.0
        patches.append(np.transpose(patch, (2, 0, 1)))
    return torch.tensor(np.stack(patches))   # [NUM_PATCHES, 3, H, W]

# =====================
# DATASET
# =====================
print("\nLoading PANDA balanced labels...")
labels_df = pd.read_csv(PANDA_CSV)

if "severity_int" in labels_df.columns:
    labels_df["severity"] = labels_df["severity_int"].astype(int)
elif "severity" in labels_df.columns:
    labels_df["severity"] = labels_df["severity"].apply(
        lambda s: SEVERITY_MAP[s] if isinstance(s, str) else int(s)
    )
else:
    labels_df["severity"] = labels_df["isup_grade"].apply(isup_to_severity)

print(f"  PANDA images  : {len(labels_df):,}")
print(f"  Severity dist : ", end="")
for sev, name in SEVERITY_LABELS.items():
    print(f"{name}={(labels_df['severity'] == sev).sum()}", end="  ")
print()


class MultimodalDataset(Dataset):
    def __init__(self, ehr_by_severity, labels_df, image_dir):
        self.ehr_by_severity = ehr_by_severity
        self.labels          = labels_df.reset_index(drop=True)
        self.image_dir       = image_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row      = self.labels.iloc[idx]
        img_id   = row["image_id"]
        severity = int(row["severity"])

        patches = load_tiff_patches(
            os.path.join(self.image_dir, f"{img_id}.tiff")
        )

        pool = self.ehr_by_severity.get(severity, [])
        ehr_vec = (torch.tensor(random.choice(pool)).float()
                   if pool else torch.zeros(EHR_DIM))

        return patches, ehr_vec, severity

# =====================
# WEIGHTED SAMPLER
# =====================
class_counts   = labels_df["severity"].value_counts().to_dict()
sample_weights = labels_df["severity"].map(lambda s: 1.0 / class_counts.get(s, 1)).values
sampler        = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights).double(),
    num_samples = len(labels_df),
    replacement = True
)

dataset = MultimodalDataset(ehr_by_severity, labels_df, IMAGE_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                     num_workers=4, pin_memory=True, prefetch_factor=2)

# =====================
# MODEL
# =====================
class ImageEncoder(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        base          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj     = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        feats = self.backbone(x).view(x.size(0), -1)
        return self.proj(feats)


class EHREncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultimodalModel(nn.Module):
    def __init__(self, ehr_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.img_enc = ImageEncoder(emb_dim)
        self.ehr_enc = EHREncoder(ehr_dim, emb_dim)

    def forward(self, patches_flat, ehr, batch_size):
        patch_embs = self.img_enc(patches_flat)
        img_embs   = patch_embs.view(batch_size, NUM_PATCHES, -1).mean(1)
        ehr_embs   = self.ehr_enc(ehr)
        return img_embs, ehr_embs

model     = MultimodalModel(EHR_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler()

# =====================
# LOSS
# =====================
def supervised_contrastive_loss(img_embs, ehr_embs, severities, temp=TEMPERATURE):
    """
    Numerically stable supervised contrastive loss.
    Any (image_i, EHR_j) pair sharing a severity class is a positive.
    """
    B = img_embs.size(0)
    img_embs = img_embs.float()
    ehr_embs = ehr_embs.float()

    embs = F.normalize(torch.cat([img_embs, ehr_embs], dim=0), dim=1)
    sev2 = torch.cat([severities, severities], dim=0)

    pos_mask = (sev2.unsqueeze(0) == sev2.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    logits   = torch.clamp((embs @ embs.T) / temp, min=-50, max=50)
    eye_mask = torch.eye(2 * B, device=logits.device).bool()
    logits   = logits.masked_fill(eye_mask, -50)

    log_denom = torch.logsumexp(logits, dim=1)
    n_pos     = pos_mask.sum(dim=1).clamp(min=1)
    numerator = (logits * pos_mask).sum(dim=1) / n_pos
    loss      = -numerator + log_denom

    if torch.isnan(loss).any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return loss.mean()

# =====================
# BATCH METRICS
# (computed during training — lightweight, no sklearn)
# =====================
def compute_batch_metrics(img_embs, ehr_embs, severities):
    """
    Fast per-batch metrics computed in-graph for training monitoring.
    Top-1 accuracy, positive/negative cosine similarity, embedding norms.
    These mirror the full evaluation suite but avoid sklearn overhead mid-epoch.
    """
    with torch.no_grad():
        img = F.normalize(img_embs, dim=1)
        ehr = F.normalize(ehr_embs, dim=1)
        sim = img @ ehr.T

        top1     = sim.argmax(dim=1)
        top1_acc = (severities[top1] == severities).float().mean().item()

        pos_mask = severities.unsqueeze(0) == severities.unsqueeze(1)
        pos_sim  = sim[pos_mask].mean().item()
        neg_sim  = sim[~pos_mask].mean().item()

        img_norm = img_embs.norm(dim=1).mean().item()
        ehr_norm = ehr_embs.norm(dim=1).mean().item()

    return {
        "top1_acc": top1_acc,
        "pos_sim":  pos_sim,
        "neg_sim":  neg_sim,
        "img_norm": img_norm,
        "ehr_norm": ehr_norm,
    }

# =====================
# FULL EVALUATION SUITE
# Mirrors late_fusion_baseline.py report_metrics exactly.
# Uses retrieval similarity scores in place of classifier softmax probabilities
# so that AUROC, ECE, and Top-k are computed on a comparable basis.
# =====================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) for multiclass problems.
    Computed per-class (One-vs-Rest) then averaged.
    """
    n_classes    = y_prob.shape[1]
    ece_per_class = []
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        prob_c      = y_prob[:, c]
        bins        = np.linspace(0, 1, n_bins + 1)
        ece         = 0.0
        for i in range(n_bins):
            mask = (prob_c >= bins[i]) & (prob_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc  = binary_true[mask].mean()
            conf = prob_c[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(acc - conf)
        ece_per_class.append(ece)
    return float(np.mean(ece_per_class))


def retrieval_probs(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a [N_img, N_ehr] cosine similarity matrix to per-class probability
    scores suitable for AUROC and ECE computation.

    For each image query i, the probability of class c is the fraction of its
    top-K retrieved EHR records that belong to class c, weighted by similarity.
    Here we use a soft-max over all EHR embeddings so the scores sum to 1.
    """
    # Softmax over EHR axis → distribution over EHR gallery
    soft = np.exp(sim_matrix - sim_matrix.max(axis=1, keepdims=True))
    soft = soft / soft.sum(axis=1, keepdims=True)    # [N_img, N_ehr]
    return soft   # used directly in per-class accumulation below


def report_retrieval_metrics(
    img_embs: np.ndarray,
    ehr_embs: np.ndarray,
    img_labels: np.ndarray,
    ehr_labels: np.ndarray,
    name: str,
    out_dir: str,
):
    """
    Compute and print the full standardized metric suite for cross-modal retrieval.

    Metrics reported
    ----------------
    Aggregate (macro-averaged across classes):
        Macro F1, Macro Precision, Macro Recall, Macro AUROC,
        Top-1 Accuracy, Top-5 Accuracy,
        Matthews Correlation Coefficient (MCC), Cohen's Kappa,
        Expected Calibration Error (ECE)

    Per-class (One-vs-Rest):
        F1, Precision, Recall, AUROC

    Visual outputs:
        Confusion matrix PNG, saved to out_dir

    Notes on probability construction
    ----------------------------------
    AUROC, ECE, and Top-k > 1 require probability scores, not hard labels.
    For retrieval we build a [N_img, 3] class-probability matrix by aggregating
    EHR-gallery similarities per severity class (soft-weighted vote):
        P(class c | image i) ∝ Σ_{j: label[j]==c} softmax_j(sim[i, :])
    This is analogous to using softmax outputs in the classifier baseline and
    makes the two files' metric computations directly comparable.
    """
    classes    = list(range(NUM_CLASSES))
    label_str  = [SEVERITY_NAMES[c] for c in classes]
    N_img      = len(img_labels)

    # --- Similarity matrix and hard top-1 predictions ---
    sim      = img_embs @ ehr_embs.T              # [N_img, N_ehr]
    top1_idx = sim.argmax(axis=1)                 # [N_img]
    y_pred   = ehr_labels[top1_idx]               # hard predictions
    y_true   = img_labels

    # --- Soft class-probability scores for AUROC / ECE / Top-k ---
    # Softmax over gallery, then sum within each class
    soft   = np.exp(sim - sim.max(axis=1, keepdims=True))
    soft  /= soft.sum(axis=1, keepdims=True)      # [N_img, N_ehr]
    y_prob = np.stack(
        [soft[:, ehr_labels == c].sum(axis=1) for c in classes], axis=1
    )                                              # [N_img, 3]
    # Renormalize so rows sum to 1 (numerical safety)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True).clip(min=1e-8)

    # --- Aggregate metrics ---
    y_bin    = label_binarize(y_true, classes=classes)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    top1_acc  = (y_true == y_pred).mean()

    # Top-5: image's true class appears in the 5 nearest EHR records
    k5     = min(5, len(ehr_labels))
    top5   = np.argsort(-sim, axis=1)[:, :k5]
    top5_acc = (ehr_labels[top5] == y_true[:, None]).any(axis=1).mean()

    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    ece   = expected_calibration_error(y_true, y_prob)

    # --- Per-class metrics ---
    per_f1  = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_auc = roc_auc_score(y_bin, y_prob, average=None, multi_class="ovr")

    # Per-class Top-1 accuracy
    correct = (y_true == y_pred)

    # --- Print ---
    sep = "  " + "-" * 52
    print(f"\n  {'━'*52}")
    print(f"  {name}")
    print(f"  {'━'*52}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(sep)
    print(f"  {'Top-1 Accuracy':<30} {top1_acc:>10.3f}")
    print(f"  {'Top-5 Accuracy':<30} {top5_acc:>10.3f}")
    print(sep)
    print(f"  {'Macro F1':<30} {macro_f1:>10.3f}")
    print(f"  {'Macro Precision':<30} {macro_pre:>10.3f}")
    print(f"  {'Macro Recall':<30} {macro_rec:>10.3f}")
    print(f"  {'Macro AUROC (OvR)':<30} {macro_auc:>10.3f}")
    print(sep)
    print(f"  {'MCC':<30} {mcc:>10.3f}")
    print(f"  {'Cohen Kappa':<30} {kappa:>10.3f}")
    print(f"  {'ECE':<30} {ece:>10.3f}")
    print(sep)

    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<12} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUROC':>7} {'Top-1':>7}")
    print(f"  {'-'*52}")
    for i, lbl in enumerate(label_str):
        mask = (y_true == i)
        cls_top1 = correct[mask].mean() if mask.sum() > 0 else 0.0
        print(f"  {lbl:<12} {per_f1[i]:>7.3f} {per_pre[i]:>7.3f} "
              f"{per_rec[i]:>7.3f} {per_auc[i]:>7.3f} {cls_top1:>7.3f}  (n={mask.sum()})")

    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_pred, target_names=label_str,
                                digits=3, zero_division=0))

    # --- Confusion matrix ---
    cm  = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=label_str).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title(
        f"{name}\n"
        f"F1={macro_f1:.3f}  AUROC={macro_auc:.3f}  "
        f"MCC={mcc:.3f}  κ={kappa:.3f}"
    )
    plt.tight_layout()
    cm_path = os.path.join(
        out_dir,
        f"cm_{name.lower().replace(' ','_').replace('/','_')}.png"
    )
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix -> {cm_path}")

    return {
        "name":      name,
        "top1_acc":  float(top1_acc),
        "top5_acc":  float(top5_acc),
        "macro_f1":  float(macro_f1),
        "macro_pre": float(macro_pre),
        "macro_rec": float(macro_rec),
        "macro_auc": float(macro_auc),
        "mcc":       float(mcc),
        "kappa":     float(kappa),
        "ece":       float(ece),
        "per_f1":    {label_str[i]: float(per_f1[i])  for i in range(len(classes))},
        "per_pre":   {label_str[i]: float(per_pre[i]) for i in range(len(classes))},
        "per_rec":   {label_str[i]: float(per_rec[i]) for i in range(len(classes))},
        "per_auc":   {label_str[i]: float(per_auc[i]) for i in range(len(classes))},
        # Raw arrays returned so callers can save them for AUROC plotting
        "y_true":    y_true,
        "y_prob":    y_prob,
    }

# =====================
# TRAIN
# =====================
history = defaultdict(list)

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0.0
    batch_bar  = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for patches, ehr, severities in batch_bar:
        B = patches.size(0)

        patches_flat = patches.view(
            B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE
        ).to(DEVICE)
        ehr        = ehr.to(DEVICE)
        severities = severities.to(DEVICE)

        with torch.cuda.amp.autocast():
            img_embs, ehr_embs = model(patches_flat, ehr, B)

        loss = supervised_contrastive_loss(img_embs, ehr_embs, severities)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        metrics = compute_batch_metrics(img_embs, ehr_embs, severities)
        for k, v in metrics.items():
            history[f"train_{k}"].append(v)

        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            top1=f"{metrics['top1_acc']:.3f}",
            pos=f"{metrics['pos_sim']:.3f}",
            neg=f"{metrics['neg_sim']:.3f}",
        )

    avg_loss = total_loss / len(loader)
    scheduler.step()
    history["train_loss"].append(avg_loss)

    epoch_metrics = {
        k: np.mean(v[-len(loader):])
        for k, v in history.items()
        if k.startswith("train_") and k != "train_loss"
    }

    print(
        f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} | "
        f"Top-1: {epoch_metrics['train_top1_acc']:.3f} | "
        f"PosSim: {epoch_metrics['train_pos_sim']:.3f} | "
        f"NegSim: {epoch_metrics['train_neg_sim']:.3f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e}"
    )

# =====================
# EVALUATION
# Runs the full standardized metric suite on all embeddings.
# =====================
print("\nRunning cross-modal retrieval evaluation...")

model.eval()
all_img_embs, all_ehr_embs, all_sev = [], [], []

eval_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

with torch.no_grad():
    for patches, ehr, severities in tqdm(eval_loader, desc="Embedding"):
        B = patches.size(0)
        patches_flat = patches.view(
            B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE
        ).to(DEVICE)
        ehr = ehr.to(DEVICE)

        img_embs, ehr_embs = model(patches_flat, ehr, B)

        all_img_embs.append(F.normalize(img_embs, dim=1).cpu().numpy())
        all_ehr_embs.append(F.normalize(ehr_embs, dim=1).cpu().numpy())
        all_sev.append(severities.numpy())

all_img = np.vstack(all_img_embs)   # [N, D]
all_ehr = np.vstack(all_ehr_embs)   # [N, D]
all_sev = np.concatenate(all_sev)   # [N]

# Full metric report — same structure as late_fusion_baseline.py
results = report_retrieval_metrics(
    img_embs   = all_img,
    ehr_embs   = all_ehr,
    img_labels = all_sev,
    ehr_labels = all_sev,
    name       = "Contrastive Retrieval",
    out_dir    = OUT_DIR,
)

# --- Save probability arrays for AUROC plotting (plot_auroc.py) ---
np.save(os.path.join(OUT_DIR, "y_true_contrastive.npy"), results["y_true"])
np.save(os.path.join(OUT_DIR, "y_prob_contrastive.npy"), results["y_prob"])
print("  Contrastive probability arrays saved for AUROC plotting.")

# =====================
# SUMMARY TABLE
# =====================
print("\n" + "=" * 70)
print("  Final Results — Contrastive Model")
print("=" * 70)
print(f"  {'Metric':<30} {'Value':>10}")
print(f"  {'-'*42}")
for key, label in [
    ("top1_acc",  "Top-1 Accuracy"),
    ("top5_acc",  "Top-5 Accuracy"),
    ("macro_f1",  "Macro F1"),
    ("macro_pre", "Macro Precision"),
    ("macro_rec", "Macro Recall"),
    ("macro_auc", "Macro AUROC (OvR)"),
    ("mcc",       "MCC"),
    ("kappa",     "Cohen Kappa"),
    ("ece",       "ECE"),
]:
    print(f"  {label:<30} {results[key]:>10.3f}")

print(f"\n  {'─'*42}")
print(f"  Chance baseline")
print(f"  {'─'*42}")
print(f"  {'Top-1 Accuracy':<30} {'0.333':>10}")
print(f"  {'Macro F1':<30} {'0.333':>10}")
print(f"  {'Macro AUROC':<30} {'0.500':>10}")
print(f"  {'MCC':<30} {'0.000':>10}")
print(f"  {'Cohen Kappa':<30} {'0.000':>10}")

# =====================
# SAVE TRAINING CURVES
# =====================
# train_loss is epoch-level (one value per epoch);
# all other history keys are batch-level (one value per batch).
# Save them as separate CSVs to avoid the unequal-length DataFrame error.

epoch_losses = history["train_loss"]   # length = EPOCHS

batch_keys   = [k for k in history if k != "train_loss"]
batch_df     = pd.DataFrame({k: history[k] for k in batch_keys})
batch_path   = os.path.join(OUT_DIR, "contrastive_training_metrics_batch.csv")
batch_df.to_csv(batch_path, index=False)

epoch_top1 = [
    np.mean(history["train_top1_acc"][i * len(loader):(i + 1) * len(loader)])
    for i in range(EPOCHS)
]
epoch_df   = pd.DataFrame({
    "epoch":      list(range(1, EPOCHS + 1)),
    "train_loss": epoch_losses,
    "train_top1_acc": epoch_top1,
    **{
        k: [np.mean(history[k][i * len(loader):(i + 1) * len(loader)])
            for i in range(EPOCHS)]
        for k in batch_keys if k != "train_top1_acc"
    },
})
epoch_path = os.path.join(OUT_DIR, "contrastive_training_metrics_epoch.csv")
epoch_df.to_csv(epoch_path, index=False)
print(f"\n  Batch-level metrics  -> {batch_path}")
print(f"  Epoch-level metrics  -> {epoch_path}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epoch_df["train_loss"])
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Contrastive Loss")
ax1.set_title("Training Loss")
ax2.plot(epoch_df["train_top1_acc"])
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Top-1 Accuracy")
ax2.set_title("Cross-Modal Top-1 Accuracy (per epoch)")
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "contrastive_training_curves.png")
plt.savefig(curves_path, dpi=150); plt.close()
print(f"  Training curves  -> {curves_path}")

# =====================
# SAVE EMBEDDINGS + RESULTS
# =====================
emb_df = pd.DataFrame(all_img, columns=[f"img_{i}" for i in range(all_img.shape[1])])
emb_df["severity"] = all_sev
emb_df.to_csv(os.path.join(OUT_DIR, "contrastive_image_embeddings.csv"), index=False)

ehr_df = pd.DataFrame(all_ehr, columns=[f"ehr_{i}" for i in range(all_ehr.shape[1])])
ehr_df["severity"] = all_sev
ehr_df.to_csv(os.path.join(OUT_DIR, "contrastive_ehr_embeddings.csv"), index=False)

# Flatten per-class dicts for CSV
flat = {k: v for k, v in results.items() if not isinstance(v, dict)}
for metric in ("per_f1", "per_pre", "per_rec", "per_auc"):
    for cls, val in results[metric].items():
        flat[f"{metric}_{cls}"] = val
pd.DataFrame([flat]).to_csv(os.path.join(OUT_DIR, "contrastive_results.csv"), index=False)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "contrastive_model.pt"))
print(f"  All outputs saved to {OUT_DIR}")