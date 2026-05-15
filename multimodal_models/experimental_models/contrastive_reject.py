"""
contrastive_reject.py — Approach 2: Reject Option

Identical to multimodal_contrastive.py (v1) with one addition:
a reject option at inference that flags low-confidence predictions
as Uncertain rather than forcing a hard class assignment.

Clinical rationale
------------------
A borderline Moderate case should be referred to a pathologist rather
than confidently mis-classified. This approach reports:
    - Coverage: fraction of cases receiving a confident prediction
    - Confident accuracy: accuracy on covered cases only
    - Per-class metrics on covered cases

The threshold sweep plot shows the coverage/accuracy/Moderate-recall
tradeoff across thresholds — use it to pick the operating point that
matches the clinical requirement (e.g. refer fewer than 20% of cases).

Tuning CONFIDENCE_THRESHOLD and TOP_K_VOTE
------------------------------------------
The standard full-gallery softmax vote spreads probability mass across
all n=1,376+ EHR records, producing near-uniform scores (range ~0.03)
that make confidence thresholding impossible.

TOP_K_VOTE (default 50) restricts the vote to the K nearest neighbours,
concentrating mass and producing a wider, more informative score range.
Smaller K = sharper scores but noisier hard predictions. Tune first:
    TOP_K_VOTE = 50   balanced (recommended starting point)
    TOP_K_VOTE = 20   sharpest — if 50 still gives narrow range
    TOP_K_VOTE = 100  gentler — if 50 is too noisy

Then tune CONFIDENCE_THRESHOLD using the printed score distribution.
The threshold sweep plot shows coverage/accuracy/Moderate-recall tradeoff.

The threshold is automatically set to the 60th percentile of max-class
scores by default, which gives ~40% referral rate as a starting point.
Override by setting CONFIDENCE_THRESHOLD to a float in [0.0, 1.0], or
set it to None to use the auto-percentile method.

    None  — auto: 60th percentile of max class scores (recommended)
    0.36  — ~20% referral (high coverage)
    0.39  — ~40% referral (balanced default)
    0.42  — ~60% referral (conservative)
    0.46  — ~80% referral (very selective)

Outputs (in addition to standard outputs)
-----------------------------------------
    cm_reject_option.png            confusion matrix on confident cases only
    reject_threshold_sweep.png      coverage/accuracy/recall vs threshold
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
OUT_DIR      = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shared_embedding_reject"

BATCH_SIZE           = 16
EPOCHS               = 25
PATCH_SIZE           = 256
NUM_PATCHES          = 8
TIFF_LEVEL           = 2
EMB_DIM              = 128
TEMPERATURE          = 0.1
NUM_CLASSES          = 3
CONFIDENCE_THRESHOLD = None   # None = auto (60th percentile of scores); or set float
TOP_K_VOTE           = 50     # top-K gallery neighbours used for class vote
                               # smaller K = sharper score distribution = wider
                               # confidence spread; tune between 20 and 100
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SEVERITY MAPPING
# =====================
SEVERITY_MAP    = {"low": 0, "moderate": 1, "high": 2}
SEVERITY_LABELS = {0: "low", 1: "moderate", 2: "high"}
SEVERITY_NAMES  = {0: "Low", 1: "Moderate", 2: "High"}

def isup_to_severity(isup: int) -> int:
    if isup <= 1: return 0
    if isup <= 3: return 1
    return 2

# =====================
# EHR LOADING
# =====================
print("Loading EHR features...")
ehr_matrix_df = pd.read_csv(EHR_MATRIX)
ehr_labels_df = pd.read_csv(EHR_LABELS, dtype=str)
ehr_features  = ehr_matrix_df.drop(columns=["subject_id"]).values.astype(np.float32)
ehr_severity  = ehr_labels_df["severity_int"].astype(int).values
EHR_DIM       = ehr_features.shape[1]

print(f"  EHR patients  : {len(ehr_features):,}")
print(f"  Feature dim   : {EHR_DIM}")
print(f"  Severity dist : ", end="")
for sev, name in SEVERITY_LABELS.items():
    print(f"{name}={(ehr_severity == sev).sum()}", end="  ")
print()

ehr_by_severity = {0: [], 1: [], 2: []}
for feat, sev in zip(ehr_features, ehr_severity):
    ehr_by_severity[sev].append(feat)

# =====================
# IMAGE LOADING
# =====================
def load_tiff_patches(path: str) -> torch.Tensor:
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
    return torch.tensor(np.stack(patches))

# =====================
# DATASET
# =====================
print("\nLoading PANDA balanced labels...")
labels_df = pd.read_csv(PANDA_CSV)
if "severity_int" in labels_df.columns:
    labels_df["severity"] = labels_df["severity_int"].astype(int)
elif "severity" in labels_df.columns:
    labels_df["severity"] = labels_df["severity"].apply(
        lambda s: SEVERITY_MAP[s] if isinstance(s, str) else int(s))
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
        patches  = load_tiff_patches(os.path.join(self.image_dir, f"{img_id}.tiff"))
        pool     = self.ehr_by_severity.get(severity, [])
        ehr_vec  = (torch.tensor(random.choice(pool)).float()
                    if pool else torch.zeros(EHR_DIM))
        return patches, ehr_vec, severity


class_counts   = labels_df["severity"].value_counts().to_dict()
sample_weights = labels_df["severity"].map(lambda s: 1.0 / class_counts.get(s, 1)).values
sampler        = WeightedRandomSampler(
    weights=torch.tensor(sample_weights).double(),
    num_samples=len(labels_df), replacement=True)

dataset = MultimodalDataset(ehr_by_severity, labels_df, IMAGE_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                     num_workers=4, pin_memory=True, prefetch_factor=2)

# =====================
# MODEL — identical to v1
# =====================
class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        base          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj     = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, emb_dim))
    def forward(self, x):
        return self.proj(self.backbone(x).view(x.size(0), -1))

class EHREncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),    nn.ReLU(),
            nn.Linear(128, emb_dim))
    def forward(self, x):
        return self.net(x)

class MultimodalModel(nn.Module):
    def __init__(self, ehr_dim, emb_dim=EMB_DIM):
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
# LOSS — identical to v1
# =====================
def supervised_contrastive_loss(img_embs, ehr_embs, severities, temp=TEMPERATURE):
    B        = img_embs.size(0)
    img_embs = img_embs.float()
    ehr_embs = ehr_embs.float()
    embs     = F.normalize(torch.cat([img_embs, ehr_embs], dim=0), dim=1)
    sev2     = torch.cat([severities, severities], dim=0)
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
# APPROACH 2: REJECT OPTION
# =====================
def resolve_threshold(y_prob, threshold=CONFIDENCE_THRESHOLD,
                      auto_percentile=60):
    """
    Resolve the confidence threshold to a concrete float.

    Retrieval-derived scores are much lower than softmax probabilities
    because the soft vote spreads mass across the full EHR gallery.
    When threshold=None, automatically set to the auto_percentile of the
    max-class score distribution so the referral rate is data-driven.

    Parameters
    ----------
    y_prob           : [N, 3] probability array
    threshold        : float or None
    auto_percentile  : percentile used when threshold is None (default 60)

    Returns
    -------
    thr   : float resolved threshold
    stats : dict of score distribution diagnostics
    """
    max_probs = y_prob.max(axis=1)
    stats = {
        "mean":  float(max_probs.mean()),
        "min":   float(max_probs.min()),
        "max":   float(max_probs.max()),
        "p25":   float(np.percentile(max_probs, 25)),
        "p50":   float(np.percentile(max_probs, 50)),
        "p60":   float(np.percentile(max_probs, 60)),
        "p75":   float(np.percentile(max_probs, 75)),
        "p90":   float(np.percentile(max_probs, 90)),
    }
    if threshold is None:
        thr = float(np.percentile(max_probs, auto_percentile))
    else:
        thr = float(threshold)
    return thr, stats


def predict_with_reject(y_prob, threshold=CONFIDENCE_THRESHOLD):
    """
    Return hard predictions with uncertain cases flagged as -1.

    For each image, if max(class probabilities) < threshold, the case is
    labelled -1 (Uncertain / Refer to pathologist).

    Parameters
    ----------
    y_prob    : [N, 3] float array of class probability scores
    threshold : float or None — if None, auto-set to 60th percentile

    Returns
    -------
    y_pred : [N] int array; -1 = uncertain, 0/1/2 = Low/Moderate/High
    thr    : float resolved threshold used
    """
    thr, _ = resolve_threshold(y_prob, threshold)
    max_prob = y_prob.max(axis=1)
    y_pred   = y_prob.argmax(axis=1).copy()
    y_pred[max_prob < thr] = -1
    return y_pred, thr


def evaluate_reject_option(y_true, y_prob, threshold=CONFIDENCE_THRESHOLD, out_dir=OUT_DIR):
    """Run the full reject-option evaluation and generate plots."""
    classes   = list(range(NUM_CLASSES))
    label_str = [SEVERITY_NAMES[c] for c in classes]

    # ── Score distribution diagnostics ───────────────────────────────────────
    thr, stats = resolve_threshold(y_prob, threshold)
    print(f"\n{'='*60}")
    print(f"  Score Distribution (max class prob per sample)")
    print(f"{'='*60}")
    print(f"  mean={stats['mean']:.4f}  min={stats['min']:.4f}  "
          f"max={stats['max']:.4f}")
    print(f"  p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  "
          f"p60={stats['p60']:.4f}  p75={stats['p75']:.4f}  "
          f"p90={stats['p90']:.4f}")
    thr_src = "auto (60th percentile)" if threshold is None else "manual"
    print(f"  Threshold used: {thr:.4f}  ({thr_src})")

    y_pred_reject, _ = predict_with_reject(y_prob, thr)
    uncertain_mask = (y_pred_reject == -1)
    confident_mask = ~uncertain_mask
    coverage       = confident_mask.mean()

    print(f"\n{'='*60}")
    print(f"  Reject-Option Evaluation  (threshold={thr:.4f})")
    print(f"{'='*60}")
    print(f"  Coverage (confident)  : {coverage*100:.1f}%  "
          f"({confident_mask.sum()}/{len(y_true)} cases)")
    print(f"  Uncertain (referred)  : {uncertain_mask.sum()} cases  "
          f"({uncertain_mask.mean()*100:.1f}%)")

    if confident_mask.sum() == 0:
        print("  No confident predictions at this threshold.")
        print("  All scores fall below threshold — threshold may be too high.")
        return

    y_tc = y_true[confident_mask]
    y_pc = y_pred_reject[confident_mask]
    y_bc = y_prob[confident_mask]

    conf_acc  = (y_tc == y_pc).mean()
    conf_f1   = f1_score(y_tc, y_pc, average="macro", zero_division=0)
    conf_mcc  = matthews_corrcoef(y_tc, y_pc)
    per_f1_c  = f1_score(y_tc, y_pc, average=None, labels=classes, zero_division=0)
    per_rec_c = recall_score(y_tc, y_pc, average=None, labels=classes, zero_division=0)

    print(f"\n  On confident predictions only:")
    print(f"  {'Top-1 Accuracy':<28} {conf_acc:>8.3f}")
    print(f"  {'Macro F1':<28} {conf_f1:>8.3f}")
    print(f"  {'MCC':<28} {conf_mcc:>8.3f}")
    print(f"\n  Per-class (confident cases):")
    print(f"  {'Class':<12} {'F1':>6} {'Rec':>6} {'n_conf':>8} {'n_ref':>8}")
    print(f"  {'-'*44}")
    for i, lbl in enumerate(label_str):
        n_conf = (y_tc == i).sum()
        n_ref  = uncertain_mask[y_true == i].sum()
        print(f"  {lbl:<12} {per_f1_c[i]:>6.3f} {per_rec_c[i]:>6.3f} "
              f"{n_conf:>8} {n_ref:>8}")

    print(f"\n  Full classification report (confident cases):")
    print(classification_report(y_tc, y_pc, target_names=label_str,
                                digits=3, zero_division=0))

    # Confusion matrix — confident predictions only
    cm  = confusion_matrix(y_tc, y_pc, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=label_str).plot(
        ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"Reject Option (threshold={threshold})\n"
        f"Coverage={coverage*100:.1f}%  "
        f"F1={conf_f1:.3f}  Acc={conf_acc:.3f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cm_reject_option.png"), dpi=150)
    plt.close()
    print(f"  Confusion matrix -> cm_reject_option.png")

    # Threshold sweep: adaptive range based on actual score distribution
    max_probs  = y_prob.max(axis=1)
    sweep_low  = float(np.percentile(max_probs, 5))
    sweep_high = float(np.percentile(max_probs, 98))
    thresholds = np.linspace(sweep_low, sweep_high, 25)
    coverages, accuracies, mod_recalls, f1s = [], [], [], []
    for thr_s in thresholds:
        yp, _  = predict_with_reject(y_prob, thr_s)
        cm_ = yp != -1
        if cm_.sum() == 0:
            coverages.append(0); accuracies.append(0)
            mod_recalls.append(0); f1s.append(0)
            continue
        coverages.append(cm_.mean())
        accuracies.append((y_true[cm_] == yp[cm_]).mean())
        mod_recalls.append(recall_score(y_true[cm_], yp[cm_], labels=[1],
                                        average="macro", zero_division=0))
        f1s.append(f1_score(y_true[cm_], yp[cm_], average="macro",
                            zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, coverages,   label="Coverage",           marker="o", linewidth=2)
    ax.plot(thresholds, accuracies,  label="Accuracy (covered)", marker="s", linewidth=2)
    ax.plot(thresholds, f1s,         label="Macro F1 (covered)", marker="D", linewidth=2)
    ax.plot(thresholds, mod_recalls, label="Moderate Recall",    marker="^",
            color="red", linestyle="--", linewidth=2)
    ax.axvline(thr, color="gray", linestyle=":", linewidth=1.5,
               label=f"Used ({thr:.4f})")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Reject Option: Coverage / Accuracy / Moderate Recall vs. Threshold",
                 fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlim(sweep_low - 0.01, sweep_high + 0.01); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reject_threshold_sweep.png"), dpi=150)
    plt.close()
    print(f"  Threshold sweep  -> reject_threshold_sweep.png")

    return {
        "threshold": float(thr),
        "coverage": float(coverage),
        "conf_acc": float(conf_acc),
        "conf_f1":  float(conf_f1),
        "conf_mcc": float(conf_mcc),
    }

# =====================
# BATCH METRICS
# =====================
def compute_batch_metrics(img_embs, ehr_embs, severities):
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
    return {"top1_acc": top1_acc, "pos_sim": pos_sim, "neg_sim": neg_sim,
            "img_norm": img_norm, "ehr_norm": ehr_norm}

# =====================
# EVALUATION HELPERS
# =====================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    n_classes     = y_prob.shape[1]
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
            ece += (mask.sum() / len(y_true)) * abs(
                binary_true[mask].mean() - prob_c[mask].mean())
        ece_per_class.append(ece)
    return float(np.mean(ece_per_class))


def report_retrieval_metrics(img_embs, ehr_embs, img_labels, ehr_labels, name, out_dir):
    classes   = list(range(NUM_CLASSES))
    label_str = [SEVERITY_NAMES[c] for c in classes]

    sim    = img_embs @ ehr_embs.T
    y_true = img_labels
    # y_pred is set below after top-K vote constructs y_prob

    # Top-K sharpened gallery vote
    # Using the full gallery spreads mass too evenly (range ~0.03) making
    # confidence scores uninformative for the reject option.
    # Restricting to the K nearest neighbours concentrates mass on
    # the most similar EHR records, producing a wider score distribution
    # that supports meaningful confidence thresholding.
    K = min(TOP_K_VOTE, sim.shape[1])
    topk_idx = np.argsort(-sim, axis=1)[:, :K]            # [N, K] indices
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)  # [N, K] scores
    topk_lbl = ehr_labels[topk_idx]                        # [N, K] labels

    # Softmax over top-K only — much sharper than full-gallery softmax
    topk_soft  = np.exp(topk_sim - topk_sim.max(axis=1, keepdims=True))
    topk_soft /= topk_soft.sum(axis=1, keepdims=True)      # [N, K]

    # Aggregate weights by class
    y_prob = np.zeros((sim.shape[0], NUM_CLASSES), dtype=np.float32)
    for c in classes:
        mask = (topk_lbl == c)                             # [N, K] bool
        y_prob[:, c] = (topk_soft * mask).sum(axis=1)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True).clip(min=1e-8)

    # Hard prediction from top-K vote (replaces raw sim argmax)
    y_pred = y_prob.argmax(axis=1)

    y_bin     = label_binarize(y_true, classes=classes)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    top1_acc  = (y_true == y_pred).mean()
    k5        = min(5, len(ehr_labels))
    top5      = np.argsort(-sim, axis=1)[:, :k5]
    top5_acc  = (ehr_labels[top5] == y_true[:, None]).any(axis=1).mean()
    mcc       = matthews_corrcoef(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    ece       = expected_calibration_error(y_true, y_prob)

    per_f1  = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_auc = roc_auc_score(y_bin, y_prob, average=None, multi_class="ovr")
    correct  = (y_true == y_pred)

    sep = "  " + "-" * 52
    print(f"\n  {'━'*52}")
    print(f"  {name}")
    print(f"  {'━'*52}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(sep)
    print(f"  {'Top-1 Accuracy (all)':<30} {top1_acc:>10.3f}")
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
        mask     = (y_true == i)
        cls_top1 = correct[mask].mean() if mask.sum() > 0 else 0.0
        print(f"  {lbl:<12} {per_f1[i]:>7.3f} {per_pre[i]:>7.3f} "
              f"{per_rec[i]:>7.3f} {per_auc[i]:>7.3f} {cls_top1:>7.3f}  (n={mask.sum()})")

    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_pred, target_names=label_str,
                                digits=3, zero_division=0))

    cm  = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=label_str).plot(
        ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{name}\nF1={macro_f1:.3f}  AUROC={macro_auc:.3f}  "
                 f"MCC={mcc:.3f}  κ={kappa:.3f}")
    plt.tight_layout()
    cm_path = os.path.join(out_dir,
        f"cm_{name.lower().replace(' ', '_').replace('/', '_')}.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"  Confusion matrix -> {cm_path}")

    return {
        "name": name, "top1_acc": float(top1_acc), "top5_acc": float(top5_acc),
        "macro_f1": float(macro_f1), "macro_pre": float(macro_pre),
        "macro_rec": float(macro_rec), "macro_auc": float(macro_auc),
        "mcc": float(mcc), "kappa": float(kappa), "ece": float(ece),
        "per_f1":  {label_str[i]: float(per_f1[i])  for i in range(len(classes))},
        "per_pre": {label_str[i]: float(per_pre[i]) for i in range(len(classes))},
        "per_rec": {label_str[i]: float(per_rec[i]) for i in range(len(classes))},
        "per_auc": {label_str[i]: float(per_auc[i]) for i in range(len(classes))},
        "y_true": y_true, "y_prob": y_prob,
    }

# =====================
# TRAINING LOOP — identical to v1
# =====================
history = defaultdict(list)

print(f"\n{'='*60}")
print(f"  Contrastive Model — Reject Option (Approach 2)")
print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}")
print(f"  Training             : identical to v1 baseline")
print(f"{'='*60}")

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0.0
    batch_bar  = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for patches, ehr, severities in batch_bar:
        B            = patches.size(0)
        patches_flat = patches.view(B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        ehr          = ehr.to(DEVICE)
        severities   = severities.to(DEVICE)

        with torch.cuda.amp.autocast():
            img_embs, ehr_embs = model(patches_flat, ehr, B)
        loss = supervised_contrastive_loss(img_embs, ehr_embs, severities)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()

        metrics = compute_batch_metrics(img_embs, ehr_embs, severities)
        for k, v in metrics.items():
            history[f"train_{k}"].append(v)

        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}", top1=f"{metrics['top1_acc']:.3f}",
            pos=f"{metrics['pos_sim']:.3f}", neg=f"{metrics['neg_sim']:.3f}")

    avg_loss = total_loss / len(loader)
    scheduler.step()
    history["train_loss"].append(avg_loss)
    epoch_metrics = {k: np.mean(v[-len(loader):]) for k, v in history.items()
                     if k.startswith("train_") and k != "train_loss"}
    print(f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} | "
          f"Top-1: {epoch_metrics['train_top1_acc']:.3f} | "
          f"PosSim: {epoch_metrics['train_pos_sim']:.3f} | "
          f"NegSim: {epoch_metrics['train_neg_sim']:.3f} | "
          f"LR: {scheduler.get_last_lr()[0]:.2e}")

# =====================
# EVALUATION
# =====================
print("\nRunning cross-modal retrieval evaluation...")
model.eval()
all_img_embs, all_ehr_embs, all_sev = [], [], []
eval_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

with torch.no_grad():
    for patches, ehr, severities in tqdm(eval_loader, desc="Embedding"):
        B            = patches.size(0)
        patches_flat = patches.view(B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        ehr          = ehr.to(DEVICE)
        img_embs, ehr_embs = model(patches_flat, ehr, B)
        all_img_embs.append(F.normalize(img_embs, dim=1).cpu().numpy())
        all_ehr_embs.append(F.normalize(ehr_embs, dim=1).cpu().numpy())
        all_sev.append(severities.numpy())

all_img = np.vstack(all_img_embs)
all_ehr = np.vstack(all_ehr_embs)
all_sev = np.concatenate(all_sev)

# Standard retrieval metrics (all cases, no reject)
results = report_retrieval_metrics(
    img_embs=all_img, ehr_embs=all_ehr,
    img_labels=all_sev, ehr_labels=all_sev,
    name="Contrastive Retrieval (No Reject)",
    out_dir=OUT_DIR,
)

# ── Approach 2: Reject option evaluation ─────────────────────────────────────
reject_results = evaluate_reject_option(
    y_true    = results["y_true"],
    y_prob    = results["y_prob"],
    threshold = CONFIDENCE_THRESHOLD,
    out_dir   = OUT_DIR,
)

np.save(os.path.join(OUT_DIR, "y_true_contrastive.npy"), results["y_true"])
np.save(os.path.join(OUT_DIR, "y_prob_contrastive.npy"), results["y_prob"])

# =====================
# SAVE TRAINING CURVES
# =====================
epoch_losses = history["train_loss"]
batch_keys   = [k for k in history if k != "train_loss"]
epoch_top1   = [np.mean(history["train_top1_acc"][i*len(loader):(i+1)*len(loader)])
                for i in range(EPOCHS)]
epoch_df     = pd.DataFrame({
    "epoch": list(range(1, EPOCHS+1)), "train_loss": epoch_losses,
    "train_top1_acc": epoch_top1,
    **{k: [np.mean(history[k][i*len(loader):(i+1)*len(loader)]) for i in range(EPOCHS)]
       for k in batch_keys if k != "train_top1_acc"},
})
epoch_df.to_csv(os.path.join(OUT_DIR, "training_metrics_epoch.csv"), index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epoch_df["train_loss"]); ax1.set_xlabel("Epoch")
ax1.set_ylabel("Contrastive Loss"); ax1.set_title("Training Loss")
ax2.plot(epoch_df["train_top1_acc"]); ax2.set_xlabel("Epoch")
ax2.set_ylabel("Top-1 Accuracy"); ax2.set_title("Cross-Modal Top-1 Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150); plt.close()

# =====================
# SAVE RESULTS
# =====================
torch.save(model.state_dict(), os.path.join(OUT_DIR, "multimodal_model.pt"))
print(f"\n  All outputs saved to {OUT_DIR}")