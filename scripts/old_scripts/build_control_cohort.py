"""
build_control_cohort.py

Builds a balanced control cohort (no PCa) to match ~1,700 PCa cases.

Cohort universe : all patients with >= 1 PSA lab test in labevents.csv
Strategy 1      : PSA-tested patients with NO C61 / ICD-9 185 diagnosis
Strategy 2      : Urological encounter patients (BPH, prostatitis, screening)
                  with no PCa ICD — only used if Strategy 1 falls short

MIMIC IV path structure on Oscar:
    /oscar/data/shared/ursa/mimic-iv/hosp/3.1/labevents.csv
    i.e. <mimic_root>/hosp/<version>/<file>.csv

Usage:
    python build_control_cohort.py \
        --mimic   /oscar/data/shared/ursa/mimic-iv \
        --version 3.1 \
        --out     /oscar/data/class/biol1595_2595/students/hgle/extracted \
        --target  1700 \
        --seed    42

Outputs (all written to --out directory):
    psa_cohort.csv           All PSA-tested patients (universe)
    pca_cases.csv            Confirmed PCa subjects  (has_pca=1)
    controls_strategy1.csv   PSA-tested, no PCa ICD
    controls_strategy2.csv   Urology encounter, no PCa ICD (fallback)
    controls_final.csv       Downsampled to --target, ready for modelling
"""

import argparse
import os
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_hosp_path(mimic_dir: str, version: str, filename: str) -> str:
    """
    Construct a full path to a MIMIC IV hosp file.
    Matches Oscar structure: <mimic_dir>/hosp/<version>/<filename>
    Example: /oscar/data/shared/ursa/mimic-iv/hosp/3.1/labevents.csv
    """
    path = os.path.join(mimic_dir, "hosp", version, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Expected MIMIC file not found: {path}\n"
            f"Check --mimic and --version arguments."
        )
    return path


def read_csv_safe(path: str, usecols: list = None) -> pd.DataFrame:
    """Read a MIMIC CSV with consistent string dtypes and optional column subset."""
    log(f"  Reading: {path}")
    df = pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    for col in ("subject_id", "hadm_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
    return df


# ---------------------------------------------------------------------------
# ICD matching functions
# ---------------------------------------------------------------------------

def is_pca(code: str, version: str) -> bool:
    """ICD-10 C61 or ICD-9 185 = malignant neoplasm of prostate."""
    code    = str(code).strip().upper().replace(".", "")
    version = str(version).strip()
    if version == "10":
        return code.startswith("C61")
    if version == "9":
        return code.startswith("185")
    return False


def is_urology(code: str, version: str) -> bool:
    """
    ICD-10: N40 (BPH), N41 (prostatitis), Z125 (screening), Z8042 (FHx)
    ICD-9:  600 (BPH), 601 (prostatitis), V7644 (screening)
    """
    code    = str(code).strip().upper().replace(".", "")
    version = str(version).strip()
    if version == "10":
        return any(code.startswith(p) for p in
                   ("N40", "N41", "Z125", "Z8042"))
    if version == "9":
        return any(code.startswith(p) for p in
                   ("600", "601", "V7644"))
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mimic_dir: str, version: str, out_dir: str,
         target: int, seed: int) -> None:

    os.makedirs(out_dir, exist_ok=True)

    # Convenience wrapper: build path and verify existence
    def hosp(filename: str) -> str:
        return build_hosp_path(mimic_dir, version, filename)

    print("=" * 60)
    print("  Control Cohort Builder")
    print(f"  MIMIC root    : {mimic_dir}")
    print(f"  MIMIC version : {version}")
    print(f"  hosp path     : {os.path.join(mimic_dir, 'hosp', version)}")
    print(f"  Output dir    : {out_dir}")
    print(f"  Target N      : {target:,}")
    print(f"  Random seed   : {seed}")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1 — Detect PSA itemid from labevents.csv
    # -----------------------------------------------------------------------
    log("[1/6] Detecting PSA itemid from labevents.csv...")

    labitems = read_csv_safe(
        hosp("labevents.csv"),
        usecols=["itemid", "label"]
    )
    psa_mask = labitems["label"].str.lower().str.contains(
        r"prostate specific antigen|^psa$", regex=True, na=False
    )
    psa_rows = labitems[psa_mask]

    if psa_rows.empty:
        log("  WARNING: PSA itemid not found in labevents — falling back to 50813")
        psa_itemid = "50813"
    else:
        psa_itemid = str(psa_rows.iloc[0]["itemid"]).strip()
        log(f"  PSA label    : {psa_rows.iloc[0]['label']}")
        log(f"  PSA itemid   : {psa_itemid}")
    print()

    # -----------------------------------------------------------------------
    # Step 2 — Build PSA cohort from labevents.csv (chunked)
    # -----------------------------------------------------------------------
    log("[2/6] Scanning labevents.csv for patients with >= 1 PSA test...")
    log("  (reading in 500k-row chunks — this is the slowest step)")

    psa_subjects: dict = {}   # subject_id -> best hadm_id seen so far
    chunks_read = 0
    BLANK_VALS  = {"", "nan", "NA", "\\N", "<NA>"}

    for chunk in pd.read_csv(
        hosp("labevents.csv"),
        dtype=str,
        usecols=["subject_id", "hadm_id", "itemid"],
        chunksize=500_000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip().str.lower()
        chunk["subject_id"] = chunk["subject_id"].astype(str).str.strip()
        chunk["hadm_id"]    = chunk["hadm_id"].astype(str).str.strip()
        chunk["itemid"]     = chunk["itemid"].astype(str).str.strip()

        psa_chunk = chunk[chunk["itemid"] == psa_itemid]

        for _, row in psa_chunk.iterrows():
            subj = row["subject_id"]
            hadm = row["hadm_id"]
            if subj not in psa_subjects:
                psa_subjects[subj] = hadm
            elif hadm not in BLANK_VALS and psa_subjects[subj] in BLANK_VALS:
                psa_subjects[subj] = hadm   # upgrade to inpatient hadm_id

        chunks_read += 1
        if chunks_read % 10 == 0:
            log(f"  ...{chunks_read * 500_000:,} rows scanned, "
                f"{len(psa_subjects):,} PSA patients found so far")

    psa_cohort = pd.DataFrame(
        list(psa_subjects.items()), columns=["subject_id", "hadm_id"]
    )
    psa_cohort["hadm_id"] = psa_cohort["hadm_id"].replace(
        dict.fromkeys(BLANK_VALS, pd.NA)
    )
    psa_cohort.to_csv(os.path.join(out_dir, "psa_cohort.csv"), index=False)

    log(f"  PSA-tested universe : {len(psa_cohort):,} patients")
    log(f"  Written -> psa_cohort.csv")
    print()

    # -----------------------------------------------------------------------
    # Step 3 — Identify confirmed PCa subjects from diagnoses_icd.csv
    # -----------------------------------------------------------------------
    log("[3/6] Identifying confirmed PCa subjects from diagnoses_icd...")

    diag = read_csv_safe(
        hosp("diagnoses_icd.csv"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"]
    )
    diag["icd_code"]    = diag["icd_code"].astype(str).str.strip()
    diag["icd_version"] = diag["icd_version"].astype(str).str.strip()

    pca_mask        = diag.apply(lambda r: is_pca(r["icd_code"], r["icd_version"]), axis=1)
    pca_subject_ids = set(diag.loc[pca_mask, "subject_id"].unique())

    pca_cases = pd.DataFrame(sorted(pca_subject_ids), columns=["subject_id"])
    pca_cases["has_pca"] = 1
    pca_cases.to_csv(os.path.join(out_dir, "pca_cases.csv"), index=False)

    log(f"  Confirmed PCa subjects : {len(pca_subject_ids):,}")
    log(f"  Written -> pca_cases.csv")
    print()

    # -----------------------------------------------------------------------
    # Step 4 — Strategy 1: PSA-tested, no PCa ICD code
    # -----------------------------------------------------------------------
    log("[4/6] Strategy 1 — PSA-tested patients with no PCa ICD code...")

    s1 = psa_cohort[~psa_cohort["subject_id"].isin(pca_subject_ids)].copy()
    s1["control_source"] = "strategy1_psa_no_pca"
    s1.to_csv(os.path.join(out_dir, "controls_strategy1.csv"), index=False)

    log(f"  Strategy 1 controls : {len(s1):,}  (target: {target:,})")
    print()

    # -----------------------------------------------------------------------
    # Step 5 — Strategy 2 (fallback): urological encounter, no PCa ICD
    # -----------------------------------------------------------------------
    log("[5/6] Strategy 2 — Urological encounter patients (fallback)...")

    if len(s1) >= target:
        log("  Strategy 1 meets target — skipping Strategy 2 scan.")
        s2 = pd.DataFrame(columns=["subject_id", "hadm_id", "control_source"])
    else:
        shortfall = target - len(s1)
        log(f"  Strategy 1 is {shortfall:,} short — scanning for urology ICD codes...")

        uro_mask = diag.apply(
            lambda r: is_urology(r["icd_code"], r["icd_version"]), axis=1
        )
        uro = diag.loc[uro_mask, ["subject_id", "hadm_id"]].copy()

        s1_ids = set(s1["subject_id"])
        uro = uro[
            ~uro["subject_id"].isin(pca_subject_ids) &
            ~uro["subject_id"].isin(s1_ids)
        ]
        uro = uro.drop_duplicates(subset="subject_id", keep="first")
        uro["control_source"] = "strategy2_urology_no_pca"
        s2 = uro.reset_index(drop=True)

        log(f"  Strategy 2 additional controls : {len(s2):,}")

    s2.to_csv(os.path.join(out_dir, "controls_strategy2.csv"), index=False)
    print()

    # -----------------------------------------------------------------------
    # Step 6 — Merge and downsample to target
    # -----------------------------------------------------------------------
    log(f"[6/6] Merging and downsampling to {target:,} controls...")

    combined   = pd.concat([s1, s2], ignore_index=True)
    combined_n = len(combined)
    log(f"  Combined pool : {combined_n:,}")

    if combined_n < target:
        log(f"  WARNING: Pool ({combined_n:,}) < target ({target:,}). "
            f"Using all available controls.")
        final = combined
    else:
        final = combined.sample(n=target, random_state=seed).reset_index(drop=True)

    final.to_csv(os.path.join(out_dir, "controls_final.csv"), index=False)

    final_n  = len(final)
    s1_final = (final["control_source"] == "strategy1_psa_no_pca").sum()
    s2_final = (final["control_source"] == "strategy2_urology_no_pca").sum()
    ratio    = final_n / max(len(pca_subject_ids), 1)

    print()
    print("=" * 60)
    print("  Cohort Balance Summary")
    print("=" * 60)
    print(f"  PSA-tested universe            : {len(psa_cohort):>8,}")
    print(f"  PCa cases        (has_pca=1)   : {len(pca_subject_ids):>8,}")
    print(f"  Controls         (has_pca=0)   : {final_n:>8,}")
    print(f"    Strategy 1  (PSA, no ICD)    : {s1_final:>8,}")
    print(f"    Strategy 2  (Urology)        : {s2_final:>8,}")
    print(f"  Case:Control ratio             :     1:{ratio:.2f}")
    print()
    print("  Next step: run proxy score binning on")
    print("  controls_final.csv + pca_cases.csv to assign")
    print("  low / moderate / high severity labels")
    print("  before merging with PANDA image classes.")
    print("=" * 60)
    print()

    log("Output files:")
    for fname in ("psa_cohort.csv", "pca_cases.csv",
                  "controls_strategy1.csv", "controls_strategy2.csv",
                  "controls_final.csv"):
        fpath = os.path.join(out_dir, fname)
        size  = os.path.getsize(fpath) / 1024
        log(f"  {fpath}  ({size:,.0f} KB)")

    print()
    log("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced PCa / control cohort from MIMIC IV"
    )
    parser.add_argument(
        "--mimic",
        default="/oscar/data/shared/ursa/mimic-iv",
        help="MIMIC IV root directory (default: /oscar/data/shared/ursa/mimic-iv)"
    )
    parser.add_argument(
        "--version",
        default="3.1",
        help="MIMIC IV hosp version subdirectory (default: 3.1)"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted",
        help="Output directory for generated CSVs"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1700,
        help="Target number of control patients (default: 1700)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible downsampling (default: 42)"
    )
    args = parser.parse_args()
    main(args.mimic, args.version, args.out, args.target, args.seed)