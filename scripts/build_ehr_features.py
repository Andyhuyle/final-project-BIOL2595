"""
build_ehr_features.py

Single-input EHR feature pipeline. Given only psa_severity_classes.csv,
this script builds the full feature matrix by pulling the remaining features
directly from MIMIC IV source files.

Pipeline:
    psa_severity_classes.csv  (subject_id, psa_max, severity_class)
              |
              |-- labevents.csv        -> psa_order_count per subject
              |-- procedures_icd.csv   -> procedure_count per subject
              |-- prescriptions.csv    -> distinct_med_count per subject
              |-- icustays.csv         -> total_icu_los_days per subject
              |-- admissions.csv       -> los_days per subject
              |-- patients.csv         -> anchor_age, gender
              |
              v
    ehr_features.csv          (raw + normalized + labels)
    ehr_feature_matrix.csv    (normalized features only, model input)
    ehr_severity_labels.csv   (subject_id, severity_int, severity_class)

Usage:
    python build_ehr_features.py \
        --psa     /path/to/psa_severity_classes.csv \
        --mimic   /oscar/data/shared/ursa/mimic-iv \
        --version 3.1 \
        --out     /path/to/output/dir
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEVERITY_MAP = {"low": 0, "moderate": 1, "high": 2}

FEATURE_COLS = [
    "psa_max",
    "psa_order_count",
    "procedure_count",
    "distinct_med_count",
    "los_days",
    "anchor_age",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def hosp(mimic_dir, version, filename):
    path = os.path.join(mimic_dir, "hosp", version, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MIMIC file not found: {path}\n"
            f"Check --mimic and --version arguments."
        )
    return path


def icu(mimic_dir, version, filename):
    path = os.path.join(mimic_dir, "icu", version, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MIMIC file not found: {path}\n"
            f"Check --mimic and --version arguments."
        )
    return path


def read_cols(path, cols, chunksize=None):
    """Read only needed columns, lowercase headers, parse subject_id as str."""
    kwargs = dict(dtype=str, usecols=cols, low_memory=False)
    if chunksize:
        kwargs["chunksize"] = chunksize
    df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.str.strip().str.lower()
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Feature extractors — each returns a subject-level Series or DataFrame
# ---------------------------------------------------------------------------

def get_psa_order_count(labevents_path, psa_itemid, subject_ids):
    """Count PSA lab rows per subject across all encounters."""
    log("  Scanning labevents for PSA order counts (chunked)...")
    counts = {}
    for chunk in pd.read_csv(
        labevents_path, dtype=str,
        usecols=["subject_id", "itemid"],
        chunksize=500_000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip().str.lower()
        chunk["subject_id"] = chunk["subject_id"].astype(str).str.strip()
        chunk["itemid"]     = chunk["itemid"].astype(str).str.strip()
        psa = chunk[chunk["itemid"] == str(psa_itemid)]
        for sid, grp in psa.groupby("subject_id"):
            counts[sid] = counts.get(sid, 0) + len(grp)

    result = pd.Series(counts, name="psa_order_count")
    result.index.name = "subject_id"
    log(f"  PSA order counts for {len(result):,} subjects")
    return result.reset_index()


def get_procedure_count(procedures_path, subject_ids):
    """Count procedure rows per subject (summed across all admissions)."""
    log("  Counting procedures per subject...")
    df = read_cols(procedures_path, ["subject_id"])
    counts = df["subject_id"].value_counts().rename("procedure_count")
    counts.index.name = "subject_id"
    log(f"  Procedure counts for {len(counts):,} subjects")
    return counts.reset_index()


def get_med_count(prescriptions_path, subject_ids):
    """Count distinct medications per subject across all admissions."""
    log("  Counting distinct medications per subject...")
    df = read_cols(prescriptions_path, ["subject_id", "drug"])
    df["drug"] = df["drug"].astype(str).str.strip()
    df = df[df["drug"] != ""]
    counts = (
        df.drop_duplicates(subset=["subject_id", "drug"])
          .groupby("subject_id")
          .size()
          .rename("distinct_med_count")
          .reset_index()
    )
    log(f"  Medication counts for {len(counts):,} subjects")
    return counts


def get_icu_los(icustays_path, subject_ids):
    """Sum ICU LOS (days) across all ICU stays per subject."""
    log("  Aggregating ICU LOS per subject...")
    df = read_cols(icustays_path, ["subject_id", "los"])
    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    result = (
        df.groupby("subject_id")["los"]
          .sum()
          .rename("total_icu_los_days")
          .reset_index()
    )
    log(f"  ICU LOS for {len(result):,} subjects")
    return result


def get_admission_los(admissions_path, subject_ids):
    """Compute mean admission LOS in days per subject."""
    log("  Computing admission LOS per subject...")
    df = read_cols(admissions_path, ["subject_id", "admittime", "dischtime"])
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
    df["los_days"]  = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    result = (
        df.groupby("subject_id")["los_days"]
          .mean()
          .rename("los_days")
          .reset_index()
    )
    log(f"  Admission LOS for {len(result):,} subjects")
    return result


def get_demographics(patients_path, subject_ids):
    """Pull anchor_age and gender per subject."""
    log("  Loading patient demographics...")
    df = read_cols(patients_path, ["subject_id", "anchor_age", "gender"])
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    log(f"  Demographics for {len(df):,} subjects")
    return df[["subject_id", "anchor_age", "gender"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(psa_path, mimic_dir, version, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  EHR Feature Builder  (single-input pipeline)")
    print(f"  Input             : {psa_path}")
    print(f"  MIMIC root        : {mimic_dir}")
    print(f"  MIMIC version     : {version}")
    print(f"  Output dir        : {out_dir}")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1 — Load PSA severity classes (the only input file)
    # -----------------------------------------------------------------------
    log("[1/8] Loading psa_severity_classes.csv...")

    psa = pd.read_csv(psa_path, dtype=str)
    psa.columns       = psa.columns.str.strip().str.lower()
    psa["subject_id"] = psa["subject_id"].astype(str).str.strip()
    psa["psa_max"]    = pd.to_numeric(psa["psa_max"], errors="coerce")
    psa["severity_class"] = psa["severity_class"].str.strip().str.lower()

    n_before = len(psa)
    psa = psa[psa["severity_class"].isin(SEVERITY_MAP.keys())].copy()
    log(f"Patients with valid severity: {len(psa):,} "
        f"(dropped {n_before - len(psa):,} unknown)")

    subject_ids = set(psa["subject_id"].tolist())
    log(f"Subject IDs to extract features for: {len(subject_ids):,}")
    print()

    # -----------------------------------------------------------------------
    # Step 2 — Detect PSA itemid
    # -----------------------------------------------------------------------
    log("[2/8] Detecting PSA itemid from d_labitems.csv...")

    labitems = pd.read_csv(hosp(mimic_dir, version, "d_labitems.csv"),
                           dtype=str, usecols=["itemid", "label"])
    labitems.columns = labitems.columns.str.strip().str.lower()
    psa_rows = labitems[
        labitems["label"].str.lower().str.contains(
            r"prostate specific antigen|^psa$", regex=True, na=False
        )
    ]
    PSA_ITEMID = psa_rows.iloc[0]["itemid"].strip() if not psa_rows.empty else "50813"
    log(f"PSA itemid: {PSA_ITEMID}")
    print()

    # -----------------------------------------------------------------------
    # Step 3 — PSA order count per subject
    # -----------------------------------------------------------------------
    log("[3/8] PSA order count...")
    psa_counts = get_psa_order_count(
        hosp(mimic_dir, version, "labevents.csv"),
        PSA_ITEMID, subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Step 4 — Procedure count per subject
    # -----------------------------------------------------------------------
    log("[4/8] Procedure count...")
    proc_counts = get_procedure_count(
        hosp(mimic_dir, version, "procedures_icd.csv"),
        subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Step 5 — Distinct medication count per subject
    # -----------------------------------------------------------------------
    log("[5/8] Medication count...")
    med_counts = get_med_count(
        hosp(mimic_dir, version, "prescriptions.csv"),
        subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Step 6 — ICU LOS per subject
    # -----------------------------------------------------------------------
    log("[6/8] ICU LOS...")
    icu_los = get_icu_los(
        os.path.join(mimic_dir, "icu", version, "icustays.csv"),
        subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Step 7 — Admission LOS per subject
    # -----------------------------------------------------------------------
    log("[7/8] Admission LOS...")
    adm_los = get_admission_los(
        hosp(mimic_dir, version, "admissions.csv"),
        subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Step 8 — Demographics
    # -----------------------------------------------------------------------
    log("[8/8] Demographics...")
    demo = get_demographics(
        hosp(mimic_dir, version, "patients.csv"),
        subject_ids
    )
    print()

    # -----------------------------------------------------------------------
    # Join everything onto the PSA severity table
    # -----------------------------------------------------------------------
    log("Joining all features onto PSA cohort...")

    df = psa.copy()
    for feat_df, key in [
        (psa_counts,  "psa_order_count"),
        (proc_counts, "procedure_count"),
        (med_counts,  "distinct_med_count"),
        (icu_los,     "total_icu_los_days"),
        (adm_los,     "los_days"),
        (demo,        "anchor_age"),
    ]:
        df = df.merge(feat_df, on="subject_id", how="left")

    # Keep gender separately (not in FEATURE_COLS but useful for Table 1)
    if "gender" in df.columns:
        gender_col = df["gender"].copy()

    log(f"Final joined table: {len(df):,} rows x {len(df.columns)} columns")
    print()

    # -----------------------------------------------------------------------
    # Cohort filter: keep only patients with procedures AND medications
    # Patients missing both have too sparse EHR data to contribute signal
    # -----------------------------------------------------------------------
    log("Filtering cohort to patients with procedures AND medications recorded...")

    df["procedure_count"]    = pd.to_numeric(df["procedure_count"],    errors="coerce")
    df["distinct_med_count"] = pd.to_numeric(df["distinct_med_count"], errors="coerce")

    n_before  = len(df)
    df        = df[
        df["procedure_count"].notna() &
        df["distinct_med_count"].notna()
    ].copy().reset_index(drop=True)
    n_dropped = n_before - len(df)

    log(f"Before filter : {n_before:,} patients")
    log(f"After filter  : {len(df):,} patients  (dropped {n_dropped:,})")
    print()

    log("Severity distribution after filtering:")
    for cls in ["low", "moderate", "high"]:
        n   = (df["severity_class"] == cls).sum()
        pct = n / len(df) * 100
        log(f"  {cls:<10}: {n:>6,}  ({pct:.1f}%)")
    print()

    # -----------------------------------------------------------------------
    # Missing value report and median imputation
    # -----------------------------------------------------------------------
    log("Missing value report (before imputation):")
    print(f"  {'Feature':<22} {'Non-null':>10} {'Missing':>10} {'Median':>10}")
    print(f"  {'-'*56}")
    medians = {}
    for col in FEATURE_COLS:
        df[col]       = pd.to_numeric(df[col], errors="coerce")
        n_miss        = df[col].isna().sum()
        n_valid       = df[col].notna().sum()
        med           = df[col].median()
        medians[col]  = med if pd.notna(med) else 0.0
        print(f"  {col:<22} {n_valid:>10,} {n_miss:>10,} {medians[col]:>10.2f}")
    print()

    for col in FEATURE_COLS:
        df[col] = df[col].fillna(medians[col])

    # -----------------------------------------------------------------------
    # Min-max normalization
    # -----------------------------------------------------------------------
    log("Normalizing features to [0, 1]...")

    X_raw   = df[FEATURE_COLS].values.astype(np.float32)
    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    col_rng = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    X_norm  = (X_raw - col_min) / col_rng

    print(f"  {'Feature':<22} {'Min':>8} {'Max':>8} {'Range':>8}")
    print(f"  {'-'*50}")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<22} {col_min[i]:>8.2f} {col_max[i]:>8.2f} {col_rng[i]:>8.2f}")
    print()

    # -----------------------------------------------------------------------
    # Build output dataframes
    # -----------------------------------------------------------------------
    df["severity_int"] = df["severity_class"].map(SEVERITY_MAP).astype(int)
    subject_ids_arr    = df["subject_id"].values
    severity_labels    = df["severity_int"].values

    # 1. Full feature file (raw + normalized + labels)
    df_full = df[["subject_id", "severity_class", "severity_int"] + FEATURE_COLS].copy()
    for i, col in enumerate(FEATURE_COLS):
        df_full[f"{col}_norm"] = X_norm[:, i]

    # 2. Normalized matrix only (model input)
    norm_cols = [f"{col}_norm" for col in FEATURE_COLS]
    df_matrix = pd.DataFrame(X_norm, columns=norm_cols)
    df_matrix.insert(0, "subject_id", subject_ids_arr)

    # 3. Severity labels
    df_labels = pd.DataFrame({
        "subject_id"    : subject_ids_arr,
        "severity_int"  : severity_labels,
        "severity_class": df["severity_class"].values,
    })

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    paths = {
        "ehr_features.csv"        : df_full,
        "ehr_feature_matrix.csv"  : df_matrix,
        "ehr_severity_labels.csv" : df_labels,
    }
    for fname, frame in paths.items():
        fpath = os.path.join(out_dir, fname)
        frame.to_csv(fpath, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total patients     : {len(df):,}")
    print(f"  Feature dimensions : {X_norm.shape[1]}")
    print()
    print("  Severity class distribution:")
    for cls, idx in SEVERITY_MAP.items():
        n   = (severity_labels == idx).sum()
        pct = n / len(severity_labels) * 100
        print(f"    {cls:<10}: {n:>6,}  ({pct:.1f}%)")
    print()
    print("  Output files:")
    for fname in paths:
        fpath   = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fpath}  ({size_kb:,.0f} KB)")
    print()
    print("  Load in your model with:")
    print("    import pandas as pd, numpy as np")
    matrix_path = os.path.join(out_dir, "ehr_feature_matrix.csv")
    labels_path = os.path.join(out_dir, "ehr_severity_labels.csv")
    print(f"    X = pd.read_csv('{matrix_path}')"
          f".drop(columns='subject_id').values.astype('float32')")
    print(f"    y = pd.read_csv('{labels_path}')['severity_int'].values")
    print()
    log("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build normalized EHR feature matrix from PSA severity classes"
    )
    parser.add_argument(
        "--psa",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted/psa_severity_classes.csv",
        help="Path to psa_severity_classes.csv (the only required input)"
    )
    parser.add_argument(
        "--mimic",
        default="/oscar/data/shared/ursa/mimic-iv",
        help="MIMIC IV root directory"
    )
    parser.add_argument(
        "--version",
        default="3.1",
        help="MIMIC IV version subdirectory under hosp/ and icu/ (default: 3.1)"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted",
        help="Output directory"
    )
    args = parser.parse_args()
    main(args.psa, args.mimic, args.version, args.out)