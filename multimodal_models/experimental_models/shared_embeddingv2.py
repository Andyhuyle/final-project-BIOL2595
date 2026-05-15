"""
multimodal_contrastive.py  —  v2

Weakly supervised multimodal contrastive learning for prostate cancer severity.

Improvements over v1
--------------------
[1] 4-class boundary split — REVERTED
    The PSA 4-20 split at 10 reduced each Moderate subgroup to ~688 samples,
    too few for the contrastive loss to form dense clusters at batch size 16.
    Moderate recall collapsed from 0.51 to 0.14. Reverted to 3-class training.
    Future work: revisit with larger batch size or more data per subgroup.

[5] Moderate temperature scaling (post-hoc, no retraining)  ← ACTIVE
    Boosts the Moderate class score at inference by MODERATE_BIAS before
    argmax, improving Moderate recall without retraining. Addresses the
    density imbalance where Low and High galleries are denser than Moderate.
    Tune MODERATE_BIAS between 1.1 and 2.0; default 1.3.

[6] Inverse-frequency weighted gallery vote  ← ACTIVE
    Weights the per-class similarity aggregation by gallery size at inference.
    Prevents large Low/High galleries from dominating the soft vote.
    Applied before temperature scaling.

[7] Increased ordinal adjacent weight  ← ACTIVE
    ADJACENT_WEIGHT raised from 0.3 to 0.5, strengthening the cohesive bond
    between Moderate and its neighbors. Requires retraining. Monitor
    pos_sim - neg_sim gap; if it drops below 0.1 the value is too high.

[2] Ordinal-aware contrastive loss  ← ACTIVE
    Replaces binary same/different mask with soft ordinal targets:
        same class     = 1.0  (strong pull)
        adjacent class = 0.3  (weak pull — clinically related)
        far class      = 0.0  (push apart)
    Encodes that Low–High confusion is a worse error than Low–Moderate.
    Operates on 3-class labels.

[3] High-severity oversampling  ← REMOVED
    Oversampling distorts contrastive batch composition — High pairs dominate
    the batch, squeezing Moderate out of a distinct embedding region.
    Contrastive learning handles class imbalance through positive pair structure;
    oversampling on top is counterproductive. Reverted to inverse-frequency only.

[4] Focal auxiliary loss on EHR encoder  ← ACTIVE
    Small classification head on EHR embeddings trained with focal loss
    (γ=2, weight=0.3). Anchors the EHR encoder to correct clinical severity
    ordering, countering spurious correlations from the general MIMIC IV cohort
    (e.g. inverted age signal seen in SHAP analysis).
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
IMAGE_DIR  = "/oscar/data/shared/ursa/kaggle_panda/train_images"
EHR_MATRIX = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
EHR_LABELS = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
PANDA_CSV  = "/oscar/data/class/biol1595_2595/students/hgle/extracted/panda_balanced.csv"
OUT_DIR    = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shared_embedding_v3"

BATCH_SIZE        = 16
EPOCHS            = 25
PATCH_SIZE        = 256
NUM_PATCHES       = 8
TIFF_LEVEL        = 2
EMB_DIM           = 128
TEMPERATURE       = 0.1
NUM_CLASSES       = 3      # 3-class throughout (4-class split reverted)
ADJACENT_WEIGHT   = 0.5    # [2,7] raised from 0.3 — stronger Moderate cohesion
MODERATE_BIAS     = 1.3    # [5] post-hoc Moderate score boost at inference (tune 1.1–2.0)
AUX_WEIGHT        = 0.3    # [4] focal loss weight
FOCAL_GAMMA       = 2.0    # [4] focal loss gamma
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SEVERITY MAPPING
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

ehr_features = ehr_matrix_df.drop(columns=["subject_id"]).values.astype(np.float32)
EHR_DIM      = ehr_features.shape[1]   # 6

ehr_severity = ehr_labels_df["severity_int"].astype(int).values

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
        severity = int(row["severity"])   # 4-class

        patches = load_tiff_patches(
            os.path.join(self.image_dir, f"{img_id}.tiff")
        )
        pool    = self.ehr_by_severity.get(severity, [])
        ehr_vec = (torch.tensor(random.choice(pool)).float()
                   if pool else torch.zeros(EHR_DIM))

        return patches, ehr_vec, severity


# =====================
# WEIGHTED SAMPLER — inverse-frequency only
# High oversampling removed: distorts contrastive batch composition.
# Contrastive learning handles imbalance through positive pair structure.
# =====================
class_counts   = labels_df["severity"].value_counts().to_dict()
sample_weights = labels_df["severity"].map(
    lambda s: 1.0 / class_counts.get(s, 1)
).values
sampler = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights).double(),
    num_samples = len(labels_df),
    replacement = True,
)

dataset = MultimodalDataset(ehr_by_severity, labels_df, IMAGE_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                     num_workers=4, pin_memory=True, prefetch_factor=2)

# =====================
# MODEL — [4] EHR auxiliary classification head
# =====================
class ImageEncoder(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        base          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj     = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return self.proj(self.backbone(x).view(x.size(0), -1))


class EHREncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),     nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalModel(nn.Module):
    def __init__(self, ehr_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.img_enc = ImageEncoder(emb_dim)
        self.ehr_enc = EHREncoder(ehr_dim, emb_dim)
        # [4] Auxiliary head — anchors EHR encoder to clinical severity order
        self.ehr_cls = nn.Linear(emb_dim, NUM_CLASSES)

    def forward(self, patches_flat, ehr, batch_size):
        patch_embs = self.img_enc(patches_flat)
        img_embs   = patch_embs.view(batch_size, NUM_PATCHES, -1).mean(1)
        ehr_embs   = self.ehr_enc(ehr)
        ehr_logits = self.ehr_cls(ehr_embs)
        return img_embs, ehr_embs, ehr_logits


model     = MultimodalModel(EHR_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler()

# =====================
# LOSS FUNCTIONS
# =====================

def ordinal_contrastive_loss(img_embs, ehr_embs, severities,
                              temp=TEMPERATURE, adj_w=ADJACENT_WEIGHT):
    """
    [2] Ordinal-aware supervised contrastive loss.

    Soft targets based on ordinal distance between 4-class severity labels:
        distance 0 (same)     -> 1.0
        distance 1 (adjacent) -> adj_w  (default 0.3)
        distance 2+ (far)     -> 0.0
    """
    B        = img_embs.size(0)
    img_embs = img_embs.float()
    ehr_embs = ehr_embs.float()

    embs = F.normalize(torch.cat([img_embs, ehr_embs], dim=0), dim=1)
    sev2 = torch.cat([severities, severities], dim=0)

    dist = (sev2.unsqueeze(0) - sev2.unsqueeze(1)).abs().float()
    soft_target = torch.where(
        dist == 0, torch.ones_like(dist),
        torch.where(dist == 1,
                    torch.full_like(dist, adj_w),
                    torch.zeros_like(dist))
    )
    soft_target.fill_diagonal_(0)

    logits   = torch.clamp((embs @ embs.T) / temp, min=-50, max=50)
    eye_mask = torch.eye(2 * B, device=logits.device).bool()
    logits   = logits.masked_fill(eye_mask, -50)

    log_denom = torch.logsumexp(logits, dim=1)
    n_pos     = soft_target.sum(dim=1).clamp(min=1)
    numerator = (logits * soft_target).sum(dim=1) / n_pos
    loss      = -numerator + log_denom

    if torch.isnan(loss).any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return loss.mean()


class FocalLoss(nn.Module):
    """[4] Focal loss for EHR auxiliary head. Down-weights easy examples."""
    def __init__(self, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


focal_criterion = FocalLoss()

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

    return {"top1_acc": top1_acc, "pos_sim": pos_sim,
            "neg_sim": neg_sim, "img_norm": img_norm, "ehr_norm": ehr_norm}

# =====================
# EVALUATION
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
            ece += (mask.sum() / len(y_true)) * abs(binary_true[mask].mean() - prob_c[mask].mean())
        ece_per_class.append(ece)
    return float(np.mean(ece_per_class))


def report_retrieval_metrics(img_embs, ehr_embs, img_labels, ehr_labels, name, out_dir):
    classes   = list(range(NUM_CLASSES))
    label_str = [SEVERITY_NAMES[c] for c in classes]

    sim    = img_embs @ ehr_embs.T
    y_true = img_labels

    # [6] Inverse-frequency weighted gallery vote
    # Prevents larger Low/High galleries from dominating the soft class score.
    gallery_sizes   = np.array([(ehr_labels == c).sum() for c in classes],
                                dtype=np.float32)
    gallery_weights = gallery_sizes.sum() / (gallery_sizes * len(classes))

    soft  = np.exp(sim - sim.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    y_prob = np.stack(
        [soft[:, ehr_labels == c].sum(axis=1) * gallery_weights[c]
         for c in classes], axis=1
    )
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True).clip(min=1e-8)

    # [5] Moderate temperature scaling — boost Moderate score before argmax
    # Addresses Low/High embedding density advantage at inference.
    y_prob_biased = y_prob.copy()
    y_prob_biased[:, 1] *= MODERATE_BIAS   # class 1 = Moderate
    y_prob_biased = y_prob_biased / y_prob_biased.sum(axis=1, keepdims=True).clip(min=1e-8)
    y_pred = y_prob_biased.argmax(axis=1)

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
    for label, val in [("Top-1 Accuracy", top1_acc), ("Top-5 Accuracy", top5_acc)]:
        print(f"  {label:<30} {val:>10.3f}")
    print(sep)
    for label, val in [("Macro F1", macro_f1), ("Macro Precision", macro_pre),
                       ("Macro Recall", macro_rec), ("Macro AUROC (OvR)", macro_auc)]:
        print(f"  {label:<30} {val:>10.3f}")
    print(sep)
    for label, val in [("MCC", mcc), ("Cohen Kappa", kappa), ("ECE", ece)]:
        print(f"  {label:<30} {val:>10.3f}")
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
# TRAINING LOOP
# =====================
history = defaultdict(list)

print(f"\n{'='*60}")
print(f"  Contrastive Model v3 (ordinal+focal+gallery+bias)")
print(f"  Classes              : {NUM_CLASSES} (3-class throughout)")
print(f"  Ordinal adj. weight  : {ADJACENT_WEIGHT} (raised from 0.3)")
print(f"  High oversample      : disabled (inverse-frequency only)")
print(f"  Focal aux. weight    : {AUX_WEIGHT}  (gamma={FOCAL_GAMMA})")
print(f"  Moderate bias        : {MODERATE_BIAS} (post-hoc inference)")
print(f"  Gallery weighting    : inverse-frequency")
print(f"{'='*60}")

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = total_contrast = total_focal = 0.0
    batch_bar  = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for patches, ehr, severities in batch_bar:
        B            = patches.size(0)
        patches_flat = patches.view(B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        ehr          = ehr.to(DEVICE)
        severities   = severities.to(DEVICE)   # 4-class

        with torch.cuda.amp.autocast():
            img_embs, ehr_embs, ehr_logits = model(patches_flat, ehr, B)
            loss_contrast = ordinal_contrastive_loss(img_embs, ehr_embs, severities)
            loss_focal    = focal_criterion(ehr_logits, severities)
            loss          = loss_contrast + AUX_WEIGHT * loss_focal

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss     += loss.item()
        total_contrast += loss_contrast.item()
        total_focal    += loss_focal.item()

        metrics = compute_batch_metrics(img_embs, ehr_embs, severities)
        for k, v in metrics.items():
            history[f"train_{k}"].append(v)

        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ctr=f"{loss_contrast.item():.4f}",
            foc=f"{loss_focal.item():.4f}",
            top1=f"{metrics['top1_acc']:.3f}",
        )

    n_batches = len(loader)
    avg_loss     = total_loss     / n_batches
    avg_contrast = total_contrast / n_batches
    avg_focal    = total_focal    / n_batches
    scheduler.step()
    history["train_loss"].append(avg_loss)
    history["train_contrast_loss"].append(avg_contrast)
    history["train_focal_loss"].append(avg_focal)

    epoch_top1 = np.mean(history["train_top1_acc"][-n_batches:])
    epoch_pos  = np.mean(history["train_pos_sim"][-n_batches:])
    epoch_neg  = np.mean(history["train_neg_sim"][-n_batches:])

    print(f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} "
          f"(ctr={avg_contrast:.4f} foc={avg_focal:.4f}) | "
          f"Top-1: {epoch_top1:.3f} | Pos: {epoch_pos:.3f} | "
          f"Neg: {epoch_neg:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}")

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

        img_embs, ehr_embs, _ = model(patches_flat, ehr, B)

        all_img_embs.append(F.normalize(img_embs, dim=1).cpu().numpy())
        all_ehr_embs.append(F.normalize(ehr_embs, dim=1).cpu().numpy())
        all_sev.append(severities.numpy())

all_img = np.vstack(all_img_embs)
all_ehr = np.vstack(all_ehr_embs)
all_sev = np.concatenate(all_sev)   # 3-class labels throughout

results = report_retrieval_metrics(
    img_embs   = all_img,
    ehr_embs   = all_ehr,
    img_labels = all_sev,
    ehr_labels = all_sev,
    name       = "Contrastive Retrieval v3",
    out_dir    = OUT_DIR,
)

np.save(os.path.join(OUT_DIR, "y_true_contrastive.npy"), results["y_true"])
np.save(os.path.join(OUT_DIR, "y_prob_contrastive.npy"), results["y_prob"])
print("  Arrays saved for AUROC plotting.")

# =====================
# SUMMARY TABLE
# =====================
print("\n" + "=" * 70)
print("  Final Results — Contrastive Model v2")
print("  (3-class, ordinal loss + oversampling + focal aux)")
print("=" * 70)
print(f"  {'Metric':<30} {'Value':>10}")
print(f"  {'-'*42}")
for key, label in [
    ("top1_acc", "Top-1 Accuracy"), ("top5_acc", "Top-5 Accuracy"),
    ("macro_f1", "Macro F1"), ("macro_pre", "Macro Precision"),
    ("macro_rec", "Macro Recall"), ("macro_auc", "Macro AUROC (OvR)"),
    ("mcc", "MCC"), ("kappa", "Cohen Kappa"), ("ece", "ECE"),
]:
    print(f"  {label:<30} {results[key]:>10.3f}")

print(f"\n  {'─'*42}")
for label, val in [("Top-1 Accuracy", "0.333"), ("Macro F1", "0.333"),
                   ("Macro AUROC", "0.500"), ("MCC", "0.000"), ("Cohen Kappa", "0.000")]:
    print(f"  {label:<30} {val:>10}  ← chance")

# =====================
# SAVE TRAINING CURVES
# =====================
n_batches  = len(loader)
epoch_top1 = [np.mean(history["train_top1_acc"][i*n_batches:(i+1)*n_batches])
              for i in range(EPOCHS)]

epoch_df = pd.DataFrame({
    "epoch":          list(range(1, EPOCHS + 1)),
    "train_loss":     history["train_loss"],
    "contrast_loss":  history["train_contrast_loss"],
    "focal_loss":     history["train_focal_loss"],
    "train_top1_acc": epoch_top1,
})
epoch_df.to_csv(os.path.join(OUT_DIR, "training_metrics_epoch.csv"), index=False)

batch_keys = ["train_top1_acc", "train_pos_sim", "train_neg_sim",
              "train_img_norm", "train_ehr_norm"]
batch_df   = pd.DataFrame({k: history[k] for k in batch_keys})
batch_df.to_csv(os.path.join(OUT_DIR, "training_metrics_batch.csv"), index=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.plot(epoch_df["train_loss"],    label="Total",       linewidth=2)
ax1.plot(epoch_df["contrast_loss"], label="Contrastive", linestyle="--")
ax1.plot(epoch_df["focal_loss"],    label="Focal",       linestyle=":")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training Loss Components"); ax1.legend()
ax2.plot(epoch_df["contrast_loss"], color="steelblue")
ax2.set_xlabel("Epoch"); ax2.set_title("Ordinal Contrastive Loss")
ax3.plot(epoch_df["train_top1_acc"], color="teal")
ax3.set_xlabel("Epoch"); ax3.set_title("Top-1 Accuracy (3-class, training)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150); plt.close()

# =====================
# SAVE EMBEDDINGS + RESULTS
# =====================
emb_img = pd.DataFrame(all_img, columns=[f"img_{i}" for i in range(all_img.shape[1])])
emb_img["severity"] = all_sev
emb_img.to_csv(os.path.join(OUT_DIR, "image_embeddings.csv"), index=False)

emb_ehr = pd.DataFrame(all_ehr, columns=[f"ehr_{i}" for i in range(all_ehr.shape[1])])
emb_ehr["severity"] = all_sev
emb_ehr.to_csv(os.path.join(OUT_DIR, "ehr_embeddings.csv"), index=False)

flat = {k: v for k, v in results.items()
        if not isinstance(v, dict) and not isinstance(v, np.ndarray)}
for metric in ("per_f1", "per_pre", "per_rec", "per_auc"):
    for cls, val in results[metric].items():
        flat[f"{metric}_{cls}"] = val
pd.DataFrame([flat]).to_csv(os.path.join(OUT_DIR, "contrastive_results.csv"), index=False)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "multimodal_model.pt"))
print(f"\n  All outputs saved to {OUT_DIR}")