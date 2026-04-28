"""
late_fusion_baseline.py

TRUE late fusion baseline for comparison against weakly supervised
contrastive learning.

Training strategy:
    - ImageEncoder trained independently with cross-entropy loss on PANDA
    - EHREncoder trained independently with cross-entropy loss on EHR features
    - At inference: average predicted probabilities from both models
    - No shared objective, no contrastive loss, no joint training

This is the correct baseline for your hypothesis:
    "A weakly supervised multimodal representation learning approach will
     outperform late-fusion baselines in predicting prostate cancer severity."

The key difference from contrastive learning:
    - Contrastive: both encoders learn a SHARED latent space aligned by severity
    - Late fusion: each encoder learns independently, combined only at inference

Usage:
    python late_fusion_baseline.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import tifffile
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
IMAGE_DIR   = "/oscar/data/shared/ursa/kaggle_panda/train_images"
PANDA_CSV   = "/oscar/data/class/biol1595_2595/students/hgle/extracted/panda_balanced.csv"
EHR_MATRIX  = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
EHR_LABELS  = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
OUT_DIR     = "/oscar/data/class/biol1595_2595/students/hgle/outputs/late_fusion"

BATCH_SIZE  = 16
EPOCHS      = 25
PATCH_SIZE  = 256
NUM_PATCHES = 8
TIFF_LEVEL  = 2
EMB_DIM     = 128
NUM_CLASSES = 3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITY_MAP   = {"low": 0, "moderate": 1, "high": 2}
SEVERITY_NAMES = {0: "Low", 1: "Moderate", 2: "High"}

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# TIFF LOADING
# =====================
def load_tiff_patches(path):
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
        dummy[:min(H,PATCH_SIZE), :min(W,PATCH_SIZE)] = \
            img[:min(H,PATCH_SIZE), :min(W,PATCH_SIZE)].astype(np.float32)/255.0
        patch = np.transpose(dummy, (2, 0, 1))
        return torch.tensor(np.stack([patch]*NUM_PATCHES))

    patches = []
    for _ in range(NUM_PATCHES):
        x = np.random.randint(0, H - PATCH_SIZE)
        y = np.random.randint(0, W - PATCH_SIZE)
        patch = img[x:x+PATCH_SIZE, y:y+PATCH_SIZE].astype(np.float32)/255.0
        patches.append(np.transpose(patch, (2, 0, 1)))
    return torch.tensor(np.stack(patches))

# =====================
# DATASETS
# =====================
class ImageDataset(Dataset):
    def __init__(self, labels_df, image_dir):
        self.labels    = labels_df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row      = self.labels.iloc[idx]
        img_id   = row["image_id"]
        severity = int(row["severity_int"])
        patches  = load_tiff_patches(
            os.path.join(self.image_dir, f"{img_id}.tiff")
        )
        return patches, severity


class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================
# MODELS
# Same architecture as contrastive model but with classification heads
# =====================
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base          = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head     = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, EMB_DIM), nn.ReLU(),
            nn.Linear(EMB_DIM, num_classes)
        )

    def forward(self, patches_flat, batch_size):
        feats  = self.backbone(patches_flat).view(patches_flat.size(0), -1)
        pooled = feats.view(batch_size, NUM_PATCHES, -1).mean(1)
        return self.head(pooled)


class EHRClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),    nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# TRAINING
# =====================
def train_model(model, loader, optimizer, scheduler, scaler, epochs, is_image):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"    Epoch {epoch+1}/{epochs}", leave=False):
            if is_image:
                patches, labels = batch
                B            = patches.size(0)
                patches_flat = patches.view(
                    B*NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE
                ).to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(model(patches_flat, B), labels)
            else:
                X, labels = batch
                X, labels = X.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(model(X), labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"    Epoch {epoch+1:>2} | Loss: {total_loss/len(loader):.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()

# =====================
# EVALUATION
# =====================
def get_predictions(model, loader, is_image):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    Evaluating", leave=False):
            if is_image:
                patches, labels = batch
                B = patches.size(0)
                logits = model(
                    patches.view(B*NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE).to(DEVICE),
                    B
                )
            else:
                X, labels = batch
                logits = model(X.to(DEVICE))

            all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def report_metrics(y_true, y_pred, y_prob, name, out_dir):
    classes   = sorted(np.unique(y_true))
    label_str = [SEVERITY_NAMES[c] for c in classes]

    macro_f1  = f1_score(y_true, y_pred, average="macro")
    macro_pre = precision_score(y_true, y_pred, average="macro")
    macro_rec = recall_score(y_true, y_pred, average="macro")
    y_bin     = label_binarize(y_true, classes=classes)
    macro_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")

    print(f"\n  {name}")
    print(f"  {'Macro F1':<20}: {macro_f1:.3f}")
    print(f"  {'Macro Precision':<20}: {macro_pre:.3f}")
    print(f"  {'Macro Recall':<20}: {macro_rec:.3f}")
    print(f"  {'Macro AUROC':<20}: {macro_auc:.3f}")
    print()
    print(classification_report(y_true, y_pred, target_names=label_str, digits=3))

    cm  = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=label_str).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title(f"{name}\nF1={macro_f1:.3f}  AUROC={macro_auc:.3f}")
    plt.tight_layout()
    path = os.path.join(out_dir,
           f"cm_{name.lower().replace(' ','_').replace('/','_')}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix -> {path}")

    return {"name": name, "macro_f1": macro_f1,
            "macro_pre": macro_pre, "macro_rec": macro_rec,
            "macro_auc": macro_auc}

# =====================
# MAIN
# =====================
print("=" * 60)
print("  Late Fusion Baseline — True Separate Training")
print("  Image model: cross-entropy on PANDA only")
print("  EHR model  : cross-entropy on MIMIC IV EHR only")
print("  Fusion     : average softmax probabilities at inference")
print("=" * 60)

# --- Load ---
print("\nLoading data...")
panda_df = pd.read_csv(PANDA_CSV)
if "severity_int" not in panda_df.columns:
    panda_df["severity_int"] = panda_df["severity"].map(SEVERITY_MAP)

ehr_mat = pd.read_csv(EHR_MATRIX, dtype=str)
ehr_lab = pd.read_csv(EHR_LABELS, dtype=str)
norm_c  = [c for c in ehr_mat.columns if c != "subject_id"]
X_ehr   = ehr_mat[norm_c].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
y_ehr   = ehr_lab["severity_int"].astype(int).values

panda_train, panda_test = train_test_split(
    panda_df, test_size=0.2, stratify=panda_df["severity_int"], random_state=42
)
idx = np.arange(len(X_ehr))
ehr_tr, ehr_te = train_test_split(idx, test_size=0.2, stratify=y_ehr, random_state=42)

print(f"  Image  train={len(panda_train):,}  test={len(panda_test):,}")
print(f"  EHR    train={len(ehr_tr):,}  test={len(ehr_te):,}")

# --- Image model ---
print("\nStep 1/3 — Training ImageClassifier (cross-entropy, no EHR)...")
class_counts   = panda_train["severity_int"].value_counts().to_dict()
sample_weights = panda_train["severity_int"].map(lambda s: 1.0/class_counts.get(s,1)).values
sampler = WeightedRandomSampler(torch.tensor(sample_weights).double(),
                                 len(panda_train), replacement=True)

img_train_dl = DataLoader(ImageDataset(panda_train, IMAGE_DIR),
                           batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=4, pin_memory=True, prefetch_factor=2)
img_test_dl  = DataLoader(ImageDataset(panda_test, IMAGE_DIR),
                           batch_size=32, shuffle=False, num_workers=4)

img_model  = ImageClassifier().to(DEVICE)
img_optim  = torch.optim.Adam(img_model.parameters(), lr=1e-4, weight_decay=1e-5)
img_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(img_optim, T_max=EPOCHS)
img_scaler = torch.cuda.amp.GradScaler()
train_model(img_model, img_train_dl, img_optim, img_sched, img_scaler, EPOCHS, is_image=True)

# --- EHR model ---
print("\nStep 2/3 — Training EHRClassifier (cross-entropy, no images)...")
ehr_train_dl = DataLoader(EHRDataset(X_ehr[ehr_tr], y_ehr[ehr_tr]),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
ehr_test_dl  = DataLoader(EHRDataset(X_ehr[ehr_te], y_ehr[ehr_te]),
                           batch_size=256, shuffle=False, num_workers=2)

ehr_model  = EHRClassifier(X_ehr.shape[1]).to(DEVICE)
ehr_optim  = torch.optim.Adam(ehr_model.parameters(), lr=1e-4, weight_decay=1e-5)
ehr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(ehr_optim, T_max=EPOCHS)
ehr_scaler = torch.cuda.amp.GradScaler()
train_model(ehr_model, ehr_test_dl, ehr_optim, ehr_sched, ehr_scaler, EPOCHS, is_image=False)

# --- Evaluate ---
print("\nStep 3/3 — Evaluation...")
results = []

y_img_true, y_img_pred, y_img_prob = get_predictions(img_model, img_test_dl, is_image=True)
results.append(report_metrics(y_img_true, y_img_pred, y_img_prob, "Image Only", OUT_DIR))

y_ehr_true, y_ehr_pred, y_ehr_prob = get_predictions(ehr_model, ehr_test_dl, is_image=False)
results.append(report_metrics(y_ehr_true, y_ehr_pred, y_ehr_prob, "EHR Only", OUT_DIR))

# Late fusion: average probabilities — use minimum test set size
n     = min(len(y_img_prob), len(y_ehr_prob))
fused = (y_img_prob[:n] + y_ehr_prob[:n]) / 2.0
results.append(report_metrics(
    y_img_true[:n], fused.argmax(axis=1), fused, "Late Fusion", OUT_DIR
))

# --- Summary ---
print("\n" + "=" * 60)
print("  Comparison Table")
print("=" * 60)
print(f"  {'Model':<35} {'F1':>6} {'AUROC':>7}")
print(f"  {'-'*50}")
for r in results:
    print(f"  {r['name']:<35} {r['macro_f1']:>6.3f} {r['macro_auc']:>7.3f}")
print(f"  {'─'*50}")
print(f"  {'Contrastive retrieval F1':<35} {'0.664':>6} {'  —':>7}")
print(f"  {'Chance baseline':<35} {'0.333':>6} {'0.500':>7}")
print()
print("  NOTE: Late fusion F1 uses label supervision during training.")
print("  Contrastive retrieval F1 does not — making it a harder task.")

pd.DataFrame(results).to_csv(os.path.join(OUT_DIR,"late_fusion_results.csv"), index=False)
torch.save(img_model.state_dict(), os.path.join(OUT_DIR, "image_classifier.pt"))
torch.save(ehr_model.state_dict(), os.path.join(OUT_DIR, "ehr_classifier.pt"))
print(f"\n  All outputs saved to {OUT_DIR}")