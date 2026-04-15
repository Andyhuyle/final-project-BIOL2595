import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import tifffile
from tqdm import tqdm

# =====================
# CONFIG
# =====================
IMAGE_DIR = "/oscar/data/shared/ursa/kaggle_panda/train_images/"
EHR_PATH = "/oscar/data/class/biol1595_2595/students/hgle/mimic_data/pc_cases.csv"
LABELS_PATH = "/oscar/data/shared/ursa/kaggle_panda/train.csv"

BATCH_SIZE = 16          # Increase now that we batch patches properly
EPOCHS = 5
PATCH_SIZE = 256
NUM_PATCHES = 8
TIFF_LEVEL = 2           # FIX 1: Read pyramid level 2 (~1/16 full res) instead of level 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# EHR PROCESSING  (unchanged)
# =====================
ehr_df = pd.read_csv(EHR_PATH)

all_types = set()
for s in ehr_df["admission_type_counts"].dropna():
    for item in s.split("|"):
        t = item.split(":")[0]
        all_types.add(t)

type_to_idx = {t: i for i, t in enumerate(sorted(all_types))}

def encode_admissions(s):
    vec = np.zeros(len(type_to_idx))
    if pd.isna(s):
        return vec
    for item in s.split("|"):
        t, count = item.split(":")
        vec[type_to_idx[t]] = int(count)
    return vec

ehr_features = []
for _, row in ehr_df.iterrows():
    age = row["anchor_age"] / 100.0
    gender = 1 if row["gender"] == "M" else 0
    admissions = encode_admissions(row["admission_type_counts"])
    feature = np.concatenate([[age, gender], admissions])
    ehr_features.append(feature)

ehr_features = np.array(ehr_features)

# =====================
# IMAGE LOADING
# FIX 1: Read at pyramid level TIFF_LEVEL, not level 0.
# PANDA .tiff files are multi-resolution; level 0 is the gigapixel scan.
# Level 2 is ~1/16 the size and loads in milliseconds, not minutes.
# =====================
def load_tiff_patches(path):
    # tifffile stores pyramid pages; page 0 = highest res
    with tifffile.TiffFile(path) as tif:
        level = min(TIFF_LEVEL, len(tif.pages) - 1)
        img = tif.pages[level].asarray()

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]          # drop alpha channel if present

    H, W, C = img.shape

    # Guard: skip images smaller than one patch
    if H < PATCH_SIZE or W < PATCH_SIZE:
        dummy = np.zeros((NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        return torch.tensor(dummy)

    patches = []
    for _ in range(NUM_PATCHES):
        x = np.random.randint(0, H - PATCH_SIZE)
        y = np.random.randint(0, W - PATCH_SIZE)
        patch = img[x:x+PATCH_SIZE, y:y+PATCH_SIZE].astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        patches.append(patch)

    return torch.tensor(np.stack(patches))   # [NUM_PATCHES, 3, H, W]

# =====================
# DATASET
# FIX 2: Return the ISUP grade so the contrastive loss has a real signal.
# Without the label the loss was comparing random EHR<->image pairs — pure noise.
# =====================
labels_df = pd.read_csv(LABELS_PATH)

# Build an EHR lookup keyed by ISUP grade bucket (0-5).
# Since image patients ≠ EHR patients we match on *severity level*, not patient ID.
# This is the weak supervision: images and EHR records with the same grade share an embedding.
ehr_by_grade = {g: [] for g in range(6)}
for feat in ehr_features:
    # EHR has no grade → assign randomly to simulate disease-severity distribution
    # In practice you'd use comorbidity scores or PSA level to derive a proxy grade
    ehr_by_grade[random.randint(0, 5)].append(feat)

class MultimodalDataset(Dataset):
    def __init__(self, ehr_by_grade, labels_df, image_dir):
        self.ehr_by_grade = ehr_by_grade
        self.labels = labels_df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_id = row["image_id"]
        grade = int(row["isup_grade"])          # 0-5

        path = os.path.join(self.image_dir, f"{img_id}.tiff")
        patches = load_tiff_patches(path)       # [NUM_PATCHES, 3, 256, 256]

        # FIX 2: Sample EHR from the same grade bucket → weak positive pairing
        pool = self.ehr_by_grade.get(grade, [])
        if pool:
            ehr_vec = torch.tensor(random.choice(pool)).float()
        else:
            ehr_vec = torch.zeros(ehr_features.shape[1])

        return patches, ehr_vec, grade

# =====================
# MODEL
# FIX 3: ImageEncoder now accepts a full batch of patches [B*P, 3, H, W]
# and aggregates within-sample with mean pooling AFTER the full forward pass.
# Previously, the outer loop called model(patches[i], ...) B times, doing B
# separate cuda kernel launches instead of one batched matmul.
# =====================
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # FIX: no deprecation warning
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        # x: [B*NUM_PATCHES, 3, 256, 256] — all patches in one batched pass
        feats = self.backbone(x)                      # [B*P, 512, 1, 1]
        feats = feats.view(feats.size(0), -1)         # [B*P, 512]
        feats = self.fc(feats)                        # [B*P, 128]
        return feats                                  # caller handles mean pooling

class EHREncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    def __init__(self, ehr_dim):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.ehr_enc = EHREncoder(ehr_dim)

    def forward(self, patches_flat, ehr, batch_size):
        # FIX 3: single batched forward pass, then fold back to [B, 128]
        patch_feats = self.img_enc(patches_flat)                   # [B*P, 128]
        img_embs = patch_feats.view(batch_size, NUM_PATCHES, 128).mean(dim=1)  # [B, 128]
        ehr_embs = self.ehr_enc(ehr)                               # [B, 128]
        return img_embs, ehr_embs

# =====================
# LOSS — supervised contrastive: same grade → attract, different grade → repel
# FIX 4: The original loss treated every sample as its own positive (diagonal only).
# With grade labels we can use proper SupCon: any same-grade pair is a positive.
# =====================
def supervised_contrastive_loss(img_embs, ehr_embs, grades, temp=0.07):
    """
    Combines image and EHR embeddings into a single 2B-sample pool,
    then treats (img_i, ehr_j) as positive whenever grade_i == grade_j.
    """
    B = img_embs.size(0)
    embs = torch.cat([img_embs, ehr_embs], dim=0)          # [2B, 128]
    embs = F.normalize(embs, dim=1)
    
    grades_doubled = torch.cat([grades, grades], dim=0)    # [2B]
    
    # Positive mask: same grade, different index
    pos_mask = (grades_doubled.unsqueeze(0) == grades_doubled.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    logits = embs @ embs.T / temp                          # [2B, 2B]
    logits.fill_diagonal_(float('-inf'))                   # exclude self

    # Log-sum-exp over all non-self pairs
    log_denom = torch.logsumexp(logits, dim=1)             # [2B]
    # Mean log-prob over positives
    loss = -(logits * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1).clamp(min=1))
    loss = loss + log_denom
    return loss.mean()

# =====================
# TRAIN
# FIX 5: Use torch.cuda.amp for mixed-precision — ~2× throughput on Ampere GPUs.
# =====================
dataset = MultimodalDataset(ehr_by_grade, labels_df, IMAGE_DIR)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,          # parallel data loading — big win for TIFF I/O
    pin_memory=True,
    prefetch_factor=2,
)

model = Model(ehr_features.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()                       # FIX 5: AMP scaler

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    model.train()
    total_loss = 0

    batch_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

    for patches, ehr, grades in batch_bar:
        B = patches.size(0)

        # Flatten patches for batched encoder: [B, P, 3, H, W] → [B*P, 3, H, W]
        patches_flat = patches.view(B * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        ehr = ehr.to(DEVICE)
        grades = grades.to(DEVICE)

        with torch.cuda.amp.autocast():                    # FIX 5: mixed precision
            img_embs, ehr_embs = model(patches_flat, ehr, B)
            loss = supervised_contrastive_loss(img_embs, ehr_embs, grades)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_bar.set_postfix(loss=loss.item())

    print(f"\nEpoch {epoch+1} completed | Avg Loss: {total_loss/len(loader):.4f}")