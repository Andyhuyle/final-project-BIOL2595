"""
contrastive_cct.py — Approach 3: Class-Conditional Temperature

Identical to multimodal_contrastive.py (v1) with one change:
the contrastive loss uses a lower temperature for Moderate-class
anchors than for Low/High anchors.

Class-Conditional Temperature (CCT) rationale
----------------------------------------------
Standard contrastive loss uses a single temperature T for all anchors.
Lower T = sharper softmax = harder decision boundary = model works harder
to separate an anchor from its negatives.

The Moderate cluster is diffuse because it sits between Low and High with
ambiguous clinical signals. By using MOD_TEMPERATURE < TEMPERATURE for
Moderate-class anchors only, we force the model to learn sharper Moderate
boundaries during training without affecting the already-tight Low/High
clusters.

This is a training-time change only — inference is identical to v1.

Tuning guidance
---------------
    MOD_TEMPERATURE = 0.05   default — half of base temperature
    MOD_TEMPERATURE = 0.07   gentler — if pos_sim - neg_sim drops below 0.05
    MOD_TEMPERATURE = 0.03   more aggressive — only if 0.05 is insufficient

Monitor during training: if pos_sim - neg_sim gap shrinks below 0.05,
MOD_TEMPERATURE is too low — increase it. If Moderate Top-1 accuracy
does not improve over v1 after epoch 10, try 0.03.
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
IMAGE_DIR       = "/oscar/data/shared/ursa/kaggle_panda/train_images"
EHR_MATRIX      = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
EHR_LABELS      = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
PANDA_CSV       = "/oscar/data/class/biol1595_2595/students/hgle/extracted/panda_balanced.csv"
OUT_DIR         = "/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/shared_embedding_cct"

BATCH_SIZE      = 16
EPOCHS          = 25
PATCH_SIZE      = 256
NUM_PATCHES     = 8
TIFF_LEVEL      = 2
EMB_DIM         = 128
TEMPERATURE     = 0.1     # base temperature for Low and High anchors
MOD_TEMPERATURE = 0.05    # lower temp for Moderate anchors — sharper boundary
NUM_CLASSES     = 3
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

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
scaler    = torch.amp.GradScaler("cuda")

# =====================
# APPROACH 3: CLASS-CONDITIONAL TEMPERATURE LOSS
# =====================
def class_conditional_contrastive_loss(img_embs, ehr_embs, severities,
                                        base_temp=TEMPERATURE,
                                        mod_temp=MOD_TEMPERATURE):
    """
    Approach 3 — Class-Conditional Temperature (CCT) contrastive loss.

    Per-anchor temperature:
        Low  (0) anchors  : base_temp  (0.1  — standard)
        Mod  (1) anchors  : mod_temp   (0.05 — sharper, forces tighter Moderate cluster)
        High (2) anchors  : base_temp  (0.1  — standard)

    Lower temperature for Moderate anchors makes the softmax sharper for
    those rows of the similarity matrix, penalizing nearby negatives more
    strongly and forcing the model to learn a crisper Moderate boundary.

    Monitoring (check training logs):
        pos_sim - neg_sim  should stay above 0.05
        If it drops below 0.05: mod_temp is too aggressive — increase to 0.07
    """
    B        = img_embs.size(0)
    img_embs = img_embs.float()
    ehr_embs = ehr_embs.float()

    embs = F.normalize(torch.cat([img_embs, ehr_embs], dim=0), dim=1)
    sev2 = torch.cat([severities, severities], dim=0)   # [2B]

    # Per-anchor temperature vector — Moderate gets mod_temp, others get base_temp
    temps = torch.where(
        sev2 == 1,
        torch.full_like(sev2, mod_temp,  dtype=torch.float32),
        torch.full_like(sev2, base_temp, dtype=torch.float32),
    )   # [2B]

    pos_mask = (sev2.unsqueeze(0) == sev2.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    # Scale each ROW of the similarity matrix by its anchor's temperature
    sim    = embs @ embs.T                           # [2B, 2B]
    logits = sim / temps.unsqueeze(1)                # broadcast: divide row i by temps[i]
    logits = torch.clamp(logits, min=-50, max=50)

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
# BATCH METRICS — also track per-class sim gap for Moderate monitoring
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

        # Per-class sim gap for Moderate monitoring
        # mod_sim: [n_mod, B] similarities from Moderate image queries to all EHR
        # pos for a Moderate query = EHR entries that are also Moderate (sev==1)
        mod_mask    = (severities == 1)
        mod_pos_sim = mod_neg_sim = 0.0
        if mod_mask.sum() > 0:
            mod_sim  = sim[mod_mask]                           # [n_mod, B]
            ehr_mod  = (severities == 1)                       # [B] EHR Moderate mask
            pos_vals = mod_sim[:, ehr_mod].flatten()
            neg_vals = mod_sim[:, ~ehr_mod].flatten()
            mod_pos_sim = pos_vals.mean().item() if pos_vals.numel() > 0 else 0.0
            mod_neg_sim = neg_vals.mean().item() if neg_vals.numel() > 0 else 0.0

        img_norm = img_embs.norm(dim=1).mean().item()
        ehr_norm = ehr_embs.norm(dim=1).mean().item()
    return {
        "top1_acc": top1_acc, "pos_sim": pos_sim, "neg_sim": neg_sim,
        "mod_pos_sim": mod_pos_sim, "mod_neg_sim": mod_neg_sim,
        "img_norm": img_norm, "ehr_norm": ehr_norm,
    }

# =====================
# EVALUATION HELPERS — identical to v1
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

    sim      = img_embs @ ehr_embs.T
    top1_idx = sim.argmax(axis=1)
    y_pred   = ehr_labels[top1_idx]
    y_true   = img_labels

    soft  = np.exp(sim - sim.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    y_prob = np.stack([soft[:, ehr_labels == c].sum(axis=1) for c in classes], axis=1)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True).clip(min=1e-8)

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
    print(f"  {'Top-1 Accuracy':<30} {top1_acc:>10.3f}")
    print(f"  {'Top-5 Accuracy':<30} {top5_acc:>10.3f}")
    print(sep)
    for lbl, val in [("Macro F1", macro_f1), ("Macro Precision", macro_pre),
                     ("Macro Recall", macro_rec), ("Macro AUROC (OvR)", macro_auc)]:
        print(f"  {lbl:<30} {val:>10.3f}")
    print(sep)
    for lbl, val in [("MCC", mcc), ("Cohen Kappa", kappa), ("ECE", ece)]:
        print(f"  {lbl:<30} {val:>10.3f}")
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
print(f"  Contrastive Model — Class-Conditional Temperature (Approach 3)")
print(f"  Base temperature     : {TEMPERATURE}  (Low / High anchors)")
print(f"  Moderate temperature : {MOD_TEMPERATURE}  (half of base — sharper boundary)")
print(f"  Monitor: pos_sim - neg_sim gap; warn if < 0.05")
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

        with torch.amp.autocast("cuda"):
            img_embs, ehr_embs = model(patches_flat, ehr, B)

        # Approach 3: class-conditional temperature replaces standard loss
        loss = class_conditional_contrastive_loss(img_embs, ehr_embs, severities)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()

        metrics = compute_batch_metrics(img_embs, ehr_embs, severities)
        for k, v in metrics.items():
            history[f"train_{k}"].append(v)

        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            top1=f"{metrics['top1_acc']:.3f}",
            gap=f"{metrics['pos_sim']-metrics['neg_sim']:.3f}",
            mgap=f"{metrics['mod_pos_sim']-metrics['mod_neg_sim']:.3f}",
        )

    avg_loss = total_loss / len(loader)
    scheduler.step()
    history["train_loss"].append(avg_loss)
    n = len(loader)
    epoch_metrics = {k: np.mean(v[-n:]) for k, v in history.items()
                     if k.startswith("train_") and k != "train_loss"}

    mod_gap = epoch_metrics.get("train_mod_pos_sim", 0) - epoch_metrics.get("train_mod_neg_sim", 0)
    gap     = epoch_metrics["train_pos_sim"] - epoch_metrics["train_neg_sim"]
    warn    = "  ⚠ mod_temp too low — increase MOD_TEMPERATURE" if mod_gap < 0.05 else ""

    print(f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} | "
          f"Top-1: {epoch_metrics['train_top1_acc']:.3f} | "
          f"Gap: {gap:.3f} | ModGap: {mod_gap:.3f} | "
          f"LR: {scheduler.get_last_lr()[0]:.2e}{warn}")

# =====================
# EVALUATION — identical to v1
# =====================
print("\nRunning cross-modal retrieval evaluation...")
print(f"  Base temp={TEMPERATURE}  Moderate temp={MOD_TEMPERATURE}")
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

results = report_retrieval_metrics(
    img_embs=all_img, ehr_embs=all_ehr,
    img_labels=all_sev, ehr_labels=all_sev,
    name=f"Contrastive Retrieval (CCT T_mod={MOD_TEMPERATURE})",
    out_dir=OUT_DIR,
)

np.save(os.path.join(OUT_DIR, "y_true_contrastive.npy"), results["y_true"])
np.save(os.path.join(OUT_DIR, "y_prob_contrastive.npy"), results["y_prob"])
print("  Arrays saved for AUROC plotting.")

# =====================
# SUMMARY
# =====================
print("\n" + "=" * 70)
print(f"  Final Results — CCT  (T_base={TEMPERATURE}, T_mod={MOD_TEMPERATURE})")
print("=" * 70)
for key, label in [
    ("top1_acc","Top-1 Accuracy"), ("top5_acc","Top-5 Accuracy"),
    ("macro_f1","Macro F1"), ("macro_pre","Macro Precision"),
    ("macro_rec","Macro Recall"), ("macro_auc","Macro AUROC (OvR)"),
    ("mcc","MCC"), ("kappa","Cohen Kappa"), ("ece","ECE"),
]:
    print(f"  {label:<30} {results[key]:>10.3f}")

# =====================
# SAVE TRAINING CURVES
# =====================
epoch_losses = history["train_loss"]
batch_keys   = [k for k in history if k != "train_loss"]
epoch_top1   = [np.mean(history["train_top1_acc"][i*len(loader):(i+1)*len(loader)])
                for i in range(EPOCHS)]
epoch_gap    = [np.mean(history["train_pos_sim"][i*len(loader):(i+1)*len(loader)]) -
                np.mean(history["train_neg_sim"][i*len(loader):(i+1)*len(loader)])
                for i in range(EPOCHS)]
epoch_modgap = [np.mean(history["train_mod_pos_sim"][i*len(loader):(i+1)*len(loader)]) -
                np.mean(history["train_mod_neg_sim"][i*len(loader):(i+1)*len(loader)])
                for i in range(EPOCHS)]

epoch_df = pd.DataFrame({
    "epoch": list(range(1, EPOCHS+1)),
    "train_loss": epoch_losses,
    "train_top1_acc": epoch_top1,
    "pos_neg_gap": epoch_gap,
    "mod_pos_neg_gap": epoch_modgap,
})
epoch_df.to_csv(os.path.join(OUT_DIR, "training_metrics_epoch.csv"), index=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.plot(epoch_df["train_loss"])
ax1.set_xlabel("Epoch"); ax1.set_title("Contrastive Loss (CCT)")
ax2.plot(epoch_df["train_top1_acc"], color="teal")
ax2.set_xlabel("Epoch"); ax2.set_title("Top-1 Accuracy")
ax3.plot(epoch_df["pos_neg_gap"],    label="All classes",  linewidth=2)
ax3.plot(epoch_df["mod_pos_neg_gap"], label="Moderate only", color="red",
         linestyle="--", linewidth=2)
ax3.axhline(0.05, color="orange", linestyle=":", label="Warning threshold (0.05)")
ax3.set_xlabel("Epoch"); ax3.set_title("Pos−Neg Similarity Gap")
ax3.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150); plt.close()

torch.save(model.state_dict(), os.path.join(OUT_DIR, "multimodal_model.pt"))
print(f"\n  All outputs saved to {OUT_DIR}")