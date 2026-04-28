"""
multimodal_contrastive.py

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
    - Cross-modal retrieval evaluation after training
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
from tqdm import tqdm

# =====================
# CONFIG
# =====================
IMAGE_DIR    = "/oscar/data/shared/ursa/kaggle_panda/train_images"
EHR_MATRIX   = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix_balanced.csv"
EHR_LABELS   = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_balanced.csv"
PANDA_CSV    = "/oscar/data/class/biol1595_2595/students/hgle/extracted/panda_balanced.csv"
OUT_DIR      = "/oscar/data/class/biol1595_2595/students/hgle/outputs"

BATCH_SIZE   = 16
EPOCHS       = 25
PATCH_SIZE   = 256
NUM_PATCHES  = 8
TIFF_LEVEL   = 2       # pyramid level 2 (~1/16 full res) — fast to load
EMB_DIM      = 128
TEMPERATURE  = 0.1    # raised from 0.07 — prevents logit overflow with AMP
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# SEVERITY MAPPING
# Shared 3-class scheme used by both modalities
# =====================
SEVERITY_MAP    = {"low": 0, "moderate": 1, "high": 2}
SEVERITY_LABELS = {0: "low", 1: "moderate", 2: "high"}

def isup_to_severity(isup: int) -> int:
    """Map PANDA ISUP grade (0-5) to 3-class severity integer."""
    if isup <= 1: return 0   # low
    if isup <= 3: return 1   # moderate
    return 2                  # high

# =====================
# EHR LOADING
# Reads the pre-built normalized feature matrix and severity labels.
# Both CSVs are outputs of build_ehr_features.py + downsampling step.
#
# ehr_feature_matrix_balanced.csv columns:
#     subject_id, psa_max_norm, psa_order_count_norm, procedure_count_norm,
#     distinct_med_count_norm, los_days_norm, anchor_age_norm
#
# ehr_severity_balanced.csv columns:
#     subject_id, severity_int, severity_class
# =====================
print("Loading EHR features...")

ehr_matrix_df = pd.read_csv(EHR_MATRIX)
ehr_labels_df = pd.read_csv(EHR_LABELS, dtype=str)

# Feature matrix: drop subject_id, keep 6 normalized columns
ehr_features  = ehr_matrix_df.drop(columns=["subject_id"]).values.astype(np.float32)
ehr_severity  = ehr_labels_df["severity_int"].astype(int).values
EHR_DIM       = ehr_features.shape[1]   # 6

print(f"  EHR patients  : {len(ehr_features):,}")
print(f"  Feature dim   : {EHR_DIM}")
print(f"  Severity dist : ", end="")
for sev, name in SEVERITY_LABELS.items():
    print(f"{name}={( ehr_severity == sev).sum()}", end="  ")
print()

# Group EHR feature vectors by severity for weak-supervision pairing
ehr_by_severity = {0: [], 1: [], 2: []}
for feat, sev in zip(ehr_features, ehr_severity):
    ehr_by_severity[sev].append(feat)

# =====================
# IMAGE LOADING
# Reads at pyramid level TIFF_LEVEL (~1/16 full res).
# PANDA .tiff files are multi-resolution; level 0 is the gigapixel scan.
# =====================
def load_tiff_patches(path: str) -> torch.Tensor:
    """
    Load NUM_PATCHES random patches from a TIFF pyramid level.

    Tries TIFF_LEVEL first; if the image is too small for one PATCH_SIZE patch,
    walks up the pyramid (lower level number = higher resolution) until a level
    large enough is found. Falls back to center-crop and zero-pad if all levels
    are too small.
    """
    with tifffile.TiffFile(path) as tif:
        n_levels = len(tif.pages)

        # Walk from TIFF_LEVEL toward level 0 until image is big enough
        img = None
        for lvl in range(min(TIFF_LEVEL, n_levels - 1), -1, -1):
            candidate = tif.pages[lvl].asarray()
            h = candidate.shape[0]
            w = candidate.shape[1] if candidate.ndim > 1 else 1
            if h > PATCH_SIZE and w > PATCH_SIZE:
                img = candidate
                break

        # If no level is large enough, use the highest resolution available
        if img is None:
            img = tif.pages[0].asarray()

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]    # drop alpha channel
    elif img.ndim == 3 and img.shape[0] in (1, 3):
        # Handle CHW format — convert to HWC
        img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

    H, W, _ = img.shape

    # If still smaller than patch size, zero-pad and return
    if H <= PATCH_SIZE or W <= PATCH_SIZE:
        dummy = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        dummy[:min(H, PATCH_SIZE), :min(W, PATCH_SIZE)] =             img[:min(H, PATCH_SIZE), :min(W, PATCH_SIZE)].astype(np.float32) / 255.0
        patch = np.transpose(dummy, (2, 0, 1))
        patches = [patch] * NUM_PATCHES
        return torch.tensor(np.stack(patches))

    patches = []
    for _ in range(NUM_PATCHES):
        x = np.random.randint(0, H - PATCH_SIZE)
        y = np.random.randint(0, W - PATCH_SIZE)
        patch = img[x:x+PATCH_SIZE, y:y+PATCH_SIZE].astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))   # HWC -> CHW
        patches.append(patch)

    return torch.tensor(np.stack(patches))        # [NUM_PATCHES, 3, H, W]

# =====================
# DATASET
# For each image, samples one EHR record from the SAME severity class
# to form a weak positive pair.
# Uses panda_balanced.csv which already has equal class sizes.
# =====================
print("\nLoading PANDA balanced labels...")
labels_df = pd.read_csv(PANDA_CSV)

# Use pre-computed severity_int column from panda_balanced.csv
# Falls back to mapping from isup_grade or severity string if not present
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
        # severity column is always int after labels_df preprocessing above
        severity = int(row["severity"])

        path    = os.path.join(self.image_dir, f"{img_id}.tiff")
        patches = load_tiff_patches(path)  # [NUM_PATCHES, 3, H, W]

        # Weak positive pairing: sample one EHR record from same severity class
        pool = self.ehr_by_severity.get(severity, [])
        if pool:
            ehr_vec = torch.tensor(random.choice(pool)).float()
        else:
            ehr_vec = torch.zeros(EHR_DIM)

        return patches, ehr_vec, severity

# =====================
# WEIGHTED SAMPLER
# Guarantees equal severity class representation per batch even if
# panda_balanced.csv has minor residual imbalance.
# =====================
class_counts    = labels_df["severity"].value_counts().to_dict()
sample_weights  = labels_df["severity"].map(lambda s: 1.0 / class_counts.get(s, 1)).values
sampler         = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights).double(),
    num_samples = len(labels_df),
    replacement = True
)

# =====================
# MODEL
#
# ImageEncoder: ResNet18 backbone, patches processed in one batched forward
#               pass then mean-pooled across patches → 128-d embedding.
#
# EHREncoder:   BatchNorm on input (stabilizes training across different
#               feature scales) → 3-layer MLP → 128-d embedding.
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
        # x: [B*NUM_PATCHES, 3, H, W]
        feats = self.backbone(x).view(x.size(0), -1)   # [B*P, 512]
        return self.proj(feats)                         # [B*P, emb_dim]


class EHREncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),   # stabilize across feature scales
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)            # [B, emb_dim]


class MultimodalModel(nn.Module):
    def __init__(self, ehr_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.img_enc = ImageEncoder(emb_dim)
        self.ehr_enc = EHREncoder(ehr_dim, emb_dim)

    def forward(self, patches_flat, ehr, batch_size):
        # Single batched image forward pass, then mean-pool across patches
        patch_embs = self.img_enc(patches_flat)                            # [B*P, D]
        img_embs   = patch_embs.view(batch_size, NUM_PATCHES, -1).mean(1)  # [B, D]
        ehr_embs   = self.ehr_enc(ehr)                                     # [B, D]
        return img_embs, ehr_embs

# =====================
# LOSS
# Supervised contrastive loss over a 2B-sample pool of image + EHR embeddings.
# Any (image_i, EHR_j) pair sharing a severity class is treated as a positive.
# This encodes the weak supervision: severity label is the only alignment signal.
# Same-index pairs are excluded (self-similarity).
# =====================
def supervised_contrastive_loss(img_embs, ehr_embs, severities, temp=TEMPERATURE):
    """
    Numerically stable supervised contrastive loss.
    Runs in float32 regardless of AMP context to prevent NaN from overflow.
    """
    B    = img_embs.size(0)

    # Cast to float32 — AMP float16 overflows with large logit values
    img_embs = img_embs.float()
    ehr_embs = ehr_embs.float()

    embs = F.normalize(torch.cat([img_embs, ehr_embs], dim=0), dim=1)  # [2B, D]
    sev2 = torch.cat([severities, severities], dim=0)                   # [2B]

    # Positive mask: same severity, different index
    pos_mask = (sev2.unsqueeze(0) == sev2.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    # Clamp embeddings to prevent extreme dot products
    logits = torch.clamp((embs @ embs.T) / temp, min=-50, max=50)  # [2B, 2B]

    # Mask self-similarity by setting diagonal to large negative (not -inf)
    # -inf causes NaN in logsumexp when all other values are also very negative
    eye_mask = torch.eye(2 * B, device=logits.device).bool()
    logits   = logits.masked_fill(eye_mask, -50)

    # Stable log-sum-exp
    log_denom = torch.logsumexp(logits, dim=1)                  # [2B]
    n_pos     = pos_mask.sum(dim=1).clamp(min=1)

    # Mean log-prob over positives
    numerator = (logits * pos_mask).sum(dim=1) / n_pos          # [2B]
    loss      = -numerator + log_denom

    # Guard: if any loss value is NaN, skip this batch
    if torch.isnan(loss).any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return loss.mean()

# =====================
# TRAIN
# =====================
dataset   = MultimodalDataset(ehr_by_severity, labels_df, IMAGE_DIR)
loader    = DataLoader(
    dataset,
    batch_size      = BATCH_SIZE,
    sampler         = sampler,
    num_workers     = 4,
    pin_memory      = True,
    prefetch_factor = 2,
)

model     = MultimodalModel(EHR_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = torch.cuda.amp.GradScaler()

print(f"\nTraining on {DEVICE} | {EPOCHS} epochs | batch {BATCH_SIZE}")
print(f"EHR dim: {EHR_DIM} | Embedding dim: {EMB_DIM}\n")

history = []

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0.0
    batch_bar  = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for patches, ehr, severities in batch_bar:
        B = patches.size(0)

        # Flatten: [B, P, 3, H, W] → [B*P, 3, H, W]
        patches_flat = patches.view(
            B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE
        ).to(DEVICE)
        ehr        = ehr.to(DEVICE)
        severities = severities.to(DEVICE)

        # autocast only for forward pass — loss runs in float32 to prevent NaN
        with torch.cuda.amp.autocast():
            img_embs, ehr_embs = model(patches_flat, ehr, B)
        loss = supervised_contrastive_loss(img_embs, ehr_embs, severities)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    scheduler.step()
    history.append(avg_loss)
    print(f"Epoch {epoch+1:>2} | Avg Loss: {avg_loss:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.2e}")

# Save model checkpoint
ckpt_path = os.path.join(OUT_DIR, "multimodal_model.pt")
torch.save({
    "epoch"       : EPOCHS,
    "model_state" : model.state_dict(),
    "optimizer"   : optimizer.state_dict(),
    "history"     : history,
    "ehr_dim"     : EHR_DIM,
    "emb_dim"     : EMB_DIM,
}, ckpt_path)
print(f"\nCheckpoint saved -> {ckpt_path}")

# =====================
# EVALUATION
# Cross-modal retrieval: for each image embedding, find the nearest EHR
# embedding by cosine similarity. Report top-1 severity accuracy.
# Chance = 33.3% for 3 balanced classes.
# =====================
print("\nRunning cross-modal retrieval evaluation...")

model.eval()
all_img_embs, all_ehr_embs, all_sev = [], [], []

eval_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=4, pin_memory=True)

with torch.no_grad():
    for patches, ehr, severities in tqdm(eval_loader, desc="Embedding"):
        B            = patches.size(0)
        patches_flat = patches.view(
            B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE
        ).to(DEVICE)
        ehr          = ehr.to(DEVICE)
        img_embs, ehr_embs = model(patches_flat, ehr, B)
        all_img_embs.append(F.normalize(img_embs, dim=1).cpu())
        all_ehr_embs.append(F.normalize(ehr_embs, dim=1).cpu())
        all_sev.append(severities)

all_img = torch.cat(all_img_embs)   # [N, D]
all_ehr = torch.cat(all_ehr_embs)   # [N, D]
all_sev = torch.cat(all_sev)        # [N]

# Cosine similarity: image rows vs EHR columns
sim      = all_img @ all_ehr.T       # [N, N]
top1_idx = sim.argmax(dim=1)
top1_sev = all_sev[top1_idx]
correct  = (top1_sev == all_sev).float()

retrieval_acc = correct.mean().item()
print(f"\nCross-modal retrieval top-1 accuracy : {retrieval_acc*100:.1f}%")
print("(chance = 33.3% for 3 balanced classes)\n")

print("Per-class retrieval accuracy:")
for sev, name in SEVERITY_LABELS.items():
    mask = (all_sev == sev)
    if mask.sum() > 0:
        acc = correct[mask].mean().item()
        print(f"  {name:<10}: {acc*100:.1f}%  (n={mask.sum()})")

# =====================
# SAVE EMBEDDINGS AS CSV
# For UMAP / t-SNE visualization in a notebook
# Columns: emb_0 ... emb_127, severity_int, modality (image/ehr)
# =====================
emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]

img_df          = pd.DataFrame(all_img.numpy(), columns=emb_cols)
img_df["severity_int"] = all_sev.numpy()
img_df["modality"]     = "image"

ehr_df_out          = pd.DataFrame(all_ehr.numpy(), columns=emb_cols)
ehr_df_out["severity_int"] = all_sev.numpy()
ehr_df_out["modality"]     = "ehr"

embeddings_df = pd.concat([img_df, ehr_df_out], ignore_index=True)
emb_path      = os.path.join(OUT_DIR, "embeddings.csv")
embeddings_df.to_csv(emb_path, index=False)

print(f"\nEmbeddings saved -> {emb_path}")
print("Columns: emb_0..emb_127, severity_int, modality")
print("Load in notebook: pd.read_csv(emb_path) then run UMAP on emb columns.")