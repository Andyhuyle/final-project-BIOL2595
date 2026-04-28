"""
downsample.py

Creates balanced CSVs for both PANDA and EHR datasets.

Downsampling logic:
    - High severity is the binding constraint (smallest class)
    - High  : keep all (688 EHR / up to 2473 PANDA)
    - Moderate and Low : sample 2x the high count

Inputs:
    PANDA_TRAIN_CSV          original train.csv from PANDA dataset
    EHR_FEATURE_MATRIX_CSV  full ehr_feature_matrix.csv (all 12,701 patients)
    EHR_LABELS_CSV           full ehr_severity_labels.csv (all 12,701 patients)

Outputs:
    panda_balanced.csv              3,440 PANDA images
    ehr_severity_balanced.csv       3,440 EHR severity labels
    ehr_feature_matrix_balanced.csv 3,440 EHR normalized features (model input)

Usage:
    python downsample.py
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — edit if needed
# ---------------------------------------------------------------------------
PANDA_TRAIN_CSV          = "/oscar/data/shared/ursa/kaggle_panda/train.csv"
EHR_FEATURE_MATRIX_CSV  = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_feature_matrix.csv"
EHR_LABELS_CSV           = "/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_severity_labels.csv"
OUT_DIR                  = "/oscar/data/class/biol1595_2595/students/hgle/extracted"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------
def isup_to_severity(isup):
    if isup <= 1: return "low"
    if isup <= 3: return "moderate"
    return "high"

# ---------------------------------------------------------------------------
# Step 1 — Load and audit PANDA
# ---------------------------------------------------------------------------
print("Loading PANDA train.csv...")
panda = pd.read_csv(PANDA_TRAIN_CSV)
panda["severity"] = panda["isup_grade"].apply(isup_to_severity)

print("Raw PANDA class distribution:")
print(panda["severity"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# Step 2 — Load and audit EHR
# ---------------------------------------------------------------------------
print("Loading EHR labels...")
ehr_labels  = pd.read_csv(EHR_LABELS_CSV,         dtype=str)
ehr_matrix  = pd.read_csv(EHR_FEATURE_MATRIX_CSV, dtype=str)

# subject_id stays string for merges; numeric feature cols cast back to float
ehr_labels["severity_int"] = ehr_labels["severity_int"].astype(int)
norm_cols = [c for c in ehr_matrix.columns if c != "subject_id"]
ehr_matrix[norm_cols] = ehr_matrix[norm_cols].apply(pd.to_numeric, errors="coerce")

print("Raw EHR class distribution:")
print(ehr_labels["severity_class"].value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# Step 3 — Determine target sizes
#
# Binding constraint: smallest class across BOTH modalities
#   EHR high  : 688
#   PANDA high : 2,473
#   → binding = 688 (EHR)
#
# Moderate and low = 2 * binding = 1,376
# ---------------------------------------------------------------------------
ehr_high_n   = (ehr_labels["severity_class"] == "high").sum()
panda_high_n = (panda["severity"] == "high").sum()

HIGH_N    = min(ehr_high_n, panda_high_n)   # 688
LOW_MOD_N = HIGH_N * 2                       # 1,376

print(f"Binding constraint (high severity) : {HIGH_N}")
print(f"Moderate and Low target            : {LOW_MOD_N}")
print(f"Total samples per modality         : {HIGH_N + LOW_MOD_N * 2}")
print()

# ---------------------------------------------------------------------------
# Step 4 — Downsample PANDA
# ---------------------------------------------------------------------------
print("Downsampling PANDA...")

panda_high     = panda[panda["severity"] == "high"]
panda_moderate = panda[panda["severity"] == "moderate"]
panda_low      = panda[panda["severity"] == "low"]

# Warn if a class doesn't have enough samples
for name, df, target in [
    ("high",     panda_high,     HIGH_N),
    ("moderate", panda_moderate, LOW_MOD_N),
    ("low",      panda_low,      LOW_MOD_N),
]:
    if len(df) < target:
        print(f"  WARNING: PANDA {name} has only {len(df)} samples "
              f"(target {target}) — using all available")

SEVERITY_MAP = {"low": 0, "moderate": 1, "high": 2}

panda_balanced = pd.concat([
    panda_high.sample(    n=min(HIGH_N,    len(panda_high)),     random_state=RANDOM_SEED),
    panda_moderate.sample(n=min(LOW_MOD_N, len(panda_moderate)), random_state=RANDOM_SEED),
    panda_low.sample(     n=min(LOW_MOD_N, len(panda_low)),      random_state=RANDOM_SEED),
]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Add integer severity column so model never needs to parse strings
panda_balanced["severity_int"] = panda_balanced["severity"].map(SEVERITY_MAP)

print("Balanced PANDA class distribution:")
print(panda_balanced["severity"].value_counts().to_string())
print(f"Total PANDA images : {len(panda_balanced):,}")
print()

# ---------------------------------------------------------------------------
# Step 5 — Downsample EHR
# ---------------------------------------------------------------------------
print("Downsampling EHR...")

ehr_high     = ehr_labels[ehr_labels["severity_class"] == "high"]
ehr_moderate = ehr_labels[ehr_labels["severity_class"] == "moderate"]
ehr_low      = ehr_labels[ehr_labels["severity_class"] == "low"]

for name, df, target in [
    ("high",     ehr_high,     HIGH_N),
    ("moderate", ehr_moderate, LOW_MOD_N),
    ("low",      ehr_low,      LOW_MOD_N),
]:
    if len(df) < target:
        print(f"  WARNING: EHR {name} has only {len(df)} samples "
              f"(target {target}) — using all available")

ehr_balanced = pd.concat([
    ehr_high.sample(    n=min(HIGH_N,    len(ehr_high)),     random_state=RANDOM_SEED),
    ehr_moderate.sample(n=min(LOW_MOD_N, len(ehr_moderate)), random_state=RANDOM_SEED),
    ehr_low.sample(     n=min(LOW_MOD_N, len(ehr_low)),      random_state=RANDOM_SEED),
]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Ensure severity_int is present and correct as integer
ehr_balanced["severity_int"] = ehr_balanced["severity_class"].map(SEVERITY_MAP)

print("Balanced EHR class distribution:")
print(ehr_balanced["severity_class"].value_counts().to_string())
print(f"Total EHR patients : {len(ehr_balanced):,}")
print()

# ---------------------------------------------------------------------------
# Step 6 — Match EHR feature matrix rows to balanced labels
# The feature matrix must stay aligned with the label file row-by-row
# ---------------------------------------------------------------------------
print("Aligning EHR feature matrix to balanced labels...")

# Cast subject_id to string in both dataframes to avoid int64 vs object mismatch
ehr_matrix["subject_id"]  = ehr_matrix["subject_id"].astype(str).str.strip()
ehr_balanced["subject_id"] = ehr_balanced["subject_id"].astype(str).str.strip()

ehr_matrix_balanced = ehr_matrix.merge(
    ehr_balanced[["subject_id"]],
    on="subject_id",
    how="inner"
).reset_index(drop=True)

# Reorder to match ehr_balanced row order exactly
ehr_matrix_balanced = (
    ehr_balanced[["subject_id"]]
    .merge(ehr_matrix_balanced, on="subject_id", how="left")
    .reset_index(drop=True)
)

print(f"Feature matrix rows : {len(ehr_matrix_balanced):,} "
      f"(should match {len(ehr_balanced):,})")
assert len(ehr_matrix_balanced) == len(ehr_balanced), \
    "Row count mismatch between feature matrix and labels after balancing"
print()

# ---------------------------------------------------------------------------
# Step 7 — Save
# ---------------------------------------------------------------------------
panda_out   = f"{OUT_DIR}/panda_balanced.csv"
labels_out  = f"{OUT_DIR}/ehr_severity_balanced.csv"
matrix_out  = f"{OUT_DIR}/ehr_feature_matrix_balanced.csv"

panda_balanced.to_csv(panda_out,  index=False)
ehr_balanced.to_csv(  labels_out, index=False)
ehr_matrix_balanced.to_csv(matrix_out, index=False)

print("=" * 50)
print("Output files:")
for path in [panda_out, labels_out, matrix_out]:
    import os
    size_kb = os.path.getsize(path) / 1024
    print(f"  {path}  ({size_kb:,.0f} KB)")

print()
print("Summary:")
print(f"  PANDA images per class : high={HIGH_N} | moderate={LOW_MOD_N} | low={LOW_MOD_N}")
print(f"  EHR patients per class : high={HIGH_N} | moderate={LOW_MOD_N} | low={LOW_MOD_N}")
print(f"  Total samples          : {HIGH_N + LOW_MOD_N * 2:,} per modality")
print()
print("These files are ready to pass into multimodal_contrastive.py")