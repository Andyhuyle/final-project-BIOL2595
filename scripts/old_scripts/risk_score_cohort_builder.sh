#!/usr/bin/env bash
# =============================================================================
# extract_mimic_pca.sh
# MIMIC IV - Prostate Cancer Cohort Feature Extraction
# Usage: bash extract_mimic_pca.sh /path/to/mimic-iv /path/to/output
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Arguments & directory setup
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <mimic_iv_dir> <output_dir>"
    echo "  mimic_iv_dir  Root directory containing hosp/ and icu/ subdirs"
    echo "  output_dir    Directory where extracted CSVs will be written"
    exit 1
fi

MIMIC="$1"
OUT="$2"
HOSP="$MIMIC/hosp/3.1"
ICU="$MIMIC/icu/3.1"

mkdir -p "$OUT"

echo "=============================================="
echo " MIMIC IV Prostate Cancer Feature Extraction"
echo "=============================================="
echo "MIMIC root : $MIMIC"
echo "Output dir : $OUT"
echo ""

# Verify required files exist before starting
REQUIRED=(
    "$HOSP/patients.csv"
    "$HOSP/admissions.csv"
    "$HOSP/diagnoses_icd.csv"
    "$HOSP/labevents.csv"
    "$HOSP/d_labitems.csv"
    "$HOSP/procedures_icd.csv"
    "$HOSP/prescriptions.csv"
    "$ICU/icustays.csv"
)
for f in "${REQUIRED[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
echo "All required input files found."
echo ""

# ---------------------------------------------------------------------------
# 1. Detect PSA itemid from d_labitems.csv
# ---------------------------------------------------------------------------
echo "[1/8] Detecting PSA itemid from d_labitems.csv..."

PSA_ITEMID=$(awk -F',' '
NR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    next
}
tolower($col["label"]) ~ /prostate specific antigen|^psa$/ {
    # Strip surrounding quotes if present
    gsub(/"/, "", $col["itemid"])
    print $col["itemid"]
    exit
}
' "$HOSP/d_labitems.csv")

if [[ -z "$PSA_ITEMID" ]]; then
    echo "  WARNING: Could not auto-detect PSA itemid. Falling back to default 50813."
    PSA_ITEMID=50813
else
    echo "  Found PSA itemid: $PSA_ITEMID"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Extract PCa patient cohort (ICD-9: 185, ICD-10: C61)
# ---------------------------------------------------------------------------
echo "[2/8] Filtering to prostate cancer admissions (ICD-9: 185, ICD-10: C61)..."

awk -F',' '
NR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "subject_id,hadm_id"
    next
}
{
    gsub(/"/, "", $col["icd_code"])
    gsub(/"/, "", $col["icd_version"])
    code    = $col["icd_code"]
    version = $col["icd_version"]

    if ((version == "10" && code ~ /^C61/) ||
        (version == "9"  && code ~ /^185/)) {
        key = $col["subject_id"] "," $col["hadm_id"]
        if (!seen[key]++) print key
    }
}
' "$HOSP/diagnoses_icd.csv" > "$OUT/pca_admissions.csv"

PCA_COUNT=$(awk 'NR>1' "$OUT/pca_admissions.csv" | wc -l | tr -d ' ')
echo "  Found $PCA_COUNT unique prostate cancer admissions."
echo ""

# Build lookup sets for downstream filters (subject_id and hadm_id)
# We write two temp files used as filter keys by later awk steps
awk -F',' 'NR>1 { print $1 }' "$OUT/pca_admissions.csv" | sort -u > "$OUT/.pca_subjects.tmp"
awk -F',' 'NR>1 { print $2 }' "$OUT/pca_admissions.csv" | sort -u > "$OUT/.pca_hadms.tmp"

# ---------------------------------------------------------------------------
# 3. Patient demographics (age, gender)
# ---------------------------------------------------------------------------
echo "[3/8] Extracting patient demographics (age, gender)..."

awk -F',' '
FNR==NR {
    pca_subj[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "subject_id,gender,anchor_age"
    next
}
{
    gsub(/"/, "", $col["subject_id"])
    if ($col["subject_id"] in pca_subj)
        print $col["subject_id"] "," $col["gender"] "," $col["anchor_age"]
}
' "$OUT/.pca_subjects.tmp" "$HOSP/patients.csv" > "$OUT/patient_demographics.csv"

echo "  Done -> patient_demographics.csv"
echo ""

# ---------------------------------------------------------------------------
# 4. Race (one row per subject; first recorded admission race)
# ---------------------------------------------------------------------------
echo "[4/8] Extracting race from admissions..."

awk -F',' '
FNR==NR {
    pca_subj[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "subject_id,race"
    next
}
{
    gsub(/"/, "", $col["subject_id"])
    subj = $col["subject_id"]
    if (subj in pca_subj && !seen[subj]++) {
        gsub(/"/, "", $col["race"])
        print subj "," $col["race"]
    }
}
' "$OUT/.pca_subjects.tmp" "$HOSP/admissions.csv" > "$OUT/patient_race.csv"

echo "  Done -> patient_race.csv"
echo ""

# ---------------------------------------------------------------------------
# 5. Admission LOS (raw timestamps; delta computed downstream in Python)
# ---------------------------------------------------------------------------
echo "[5/8] Extracting admission timestamps for LOS computation..."

awk -F',' '
FNR==NR {
    pca_hadm[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    print "subject_id,hadm_id,admittime,dischtime"
    next
}
{
    gsub(/"/, "", $col["hadm_id"])
    if ($col["hadm_id"] in pca_hadm)
        print $col["subject_id"] "," $col["hadm_id"] "," \
              $col["admittime"]  "," $col["dischtime"]
}
' "$OUT/.pca_hadms.tmp" "$HOSP/admissions.csv" > "$OUT/admissions_los_raw.csv"

echo "  Done -> admissions_los_raw.csv"
echo ""

# ---------------------------------------------------------------------------
# 6. ICU LOS per admission (summed across all ICU stays)
# ---------------------------------------------------------------------------
echo "[6/8] Extracting and aggregating ICU LOS..."

awk -F',' '
FNR==NR {
    pca_hadm[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    next
}
{
    gsub(/"/, "", $col["hadm_id"])
    hadm = $col["hadm_id"]
    if (hadm in pca_hadm) {
        key = $col["subject_id"] "," hadm
        icu_sum[key] += $col["los"]
        subj[key]     = $col["subject_id"]
        hadm_id[key]  = hadm
    }
}
END {
    print "subject_id,hadm_id,total_icu_los_days"
    for (k in icu_sum)
        print subj[k] "," hadm_id[k] "," icu_sum[k]
}
' "$OUT/.pca_hadms.tmp" "$ICU/icustays.csv" > "$OUT/icu_los_per_admission.csv"

echo "  Done -> icu_los_per_admission.csv"
echo ""

# ---------------------------------------------------------------------------
# 7. PSA lab order count per admission
# ---------------------------------------------------------------------------
echo "[7/8] Counting PSA lab orders per admission (itemid=$PSA_ITEMID)..."

awk -F',' -v psa_id="$PSA_ITEMID" '
FNR==NR {
    pca_hadm[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    next
}
{
    gsub(/"/, "", $col["hadm_id"])
    gsub(/"/, "", $col["itemid"])
    hadm = $col["hadm_id"]
    if ($col["itemid"] == psa_id && hadm in pca_hadm) {
        key = $col["subject_id"] "," hadm
        count[key]++
        subj[key]    = $col["subject_id"]
        hadm_id[key] = hadm
    }
}
END {
    print "subject_id,hadm_id,psa_order_count"
    for (k in count)
        print subj[k] "," hadm_id[k] "," count[k]
}
' "$OUT/.pca_hadms.tmp" "$HOSP/labevents.csv" > "$OUT/psa_counts.csv"

echo "  Done -> psa_counts.csv"
echo ""

# ---------------------------------------------------------------------------
# 8. Procedure count per admission
# ---------------------------------------------------------------------------
echo "[8/8] Counting procedures per admission..."

awk -F',' '
FNR==NR {
    pca_hadm[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    next
}
{
    gsub(/"/, "", $col["hadm_id"])
    hadm = $col["hadm_id"]
    if (hadm in pca_hadm) {
        key = $col["subject_id"] "," hadm
        count[key]++
        subj[key]    = $col["subject_id"]
        hadm_id[key] = hadm
    }
}
END {
    print "subject_id,hadm_id,procedure_count"
    for (k in count)
        print subj[k] "," hadm_id[k] "," count[k]
}
' "$OUT/.pca_hadms.tmp" "$HOSP/procedures_icd.csv" > "$OUT/procedure_counts.csv"

echo "  Done -> procedure_counts.csv"
echo ""

# ---------------------------------------------------------------------------
# 9. Distinct medication count per admission
# ---------------------------------------------------------------------------
echo "[9/9] Counting distinct medications per admission..."

awk -F',' '
FNR==NR {
    pca_hadm[$1] = 1
    next
}
FNR==1 {
    for (i=1; i<=NF; i++) col[$i]=i
    next
}
{
    gsub(/"/, "", $col["hadm_id"])
    gsub(/"/, "", $col["drug"])
    hadm = $col["hadm_id"]
    drug = $col["drug"]
    if (hadm in pca_hadm && drug != "") {
        dedup_key = $col["subject_id"] "," hadm "," drug
        if (!seen[dedup_key]++) {
            key = $col["subject_id"] "," hadm
            count[key]++
            subj[key]    = $col["subject_id"]
            hadm_id[key] = hadm
        }
    }
}
END {
    print "subject_id,hadm_id,distinct_med_count"
    for (k in count)
        print subj[k] "," hadm_id[k] "," count[k]
}
' "$OUT/.pca_hadms.tmp" "$HOSP/prescriptions.csv" > "$OUT/med_counts.csv"

echo "  Done -> med_counts.csv"
echo ""

# ---------------------------------------------------------------------------
# 10. Merge all features into a single cohort CSV using Python
# ---------------------------------------------------------------------------
echo "Merging all feature files into final cohort CSV..."

python3 - "$OUT" << 'PYEOF'
import sys
import pandas as pd

out = sys.argv[1]

base    = pd.read_csv(f"{out}/pca_admissions.csv",      dtype=str)
demo    = pd.read_csv(f"{out}/patient_demographics.csv", dtype=str)
race    = pd.read_csv(f"{out}/patient_race.csv",         dtype=str)
icu     = pd.read_csv(f"{out}/icu_los_per_admission.csv")
psa     = pd.read_csv(f"{out}/psa_counts.csv")
procs   = pd.read_csv(f"{out}/procedure_counts.csv")
meds    = pd.read_csv(f"{out}/med_counts.csv")

adm_raw = pd.read_csv(
    f"{out}/admissions_los_raw.csv",
    parse_dates=["admittime", "dischtime"]
)
adm_raw["los_days"] = (
    (adm_raw["dischtime"] - adm_raw["admittime"]).dt.total_seconds() / 86400
)

df = (base
      .merge(demo,                                  on="subject_id",              how="left")
      .merge(race,                                  on="subject_id",              how="left")
      .merge(adm_raw[["subject_id","hadm_id","los_days"]],
                                                    on=["subject_id","hadm_id"],  how="left")
      .merge(icu,                                   on=["subject_id","hadm_id"],  how="left")
      .merge(psa,                                   on=["subject_id","hadm_id"],  how="left")
      .merge(procs,                                 on=["subject_id","hadm_id"],  how="left")
      .merge(meds,                                  on=["subject_id","hadm_id"],  how="left"))

fill_zero = ["total_icu_los_days","psa_order_count","procedure_count","distinct_med_count"]
df[fill_zero] = df[fill_zero].fillna(0)

df["anchor_age"]   = pd.to_numeric(df["anchor_age"],   errors="coerce")
df["los_days"]     = pd.to_numeric(df["los_days"],     errors="coerce")

out_path = f"{out}/mimic_pca_cohort_features.csv"
df.to_csv(out_path, index=False)

print(f"  Admissions : {len(df)}")
print(f"  Patients   : {df['subject_id'].nunique()}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Saved to   : {out_path}")
PYEOF

# ---------------------------------------------------------------------------
# Cleanup temp files
# ---------------------------------------------------------------------------
rm -f "$OUT/.pca_subjects.tmp" "$OUT/.pca_hadms.tmp"

echo ""
echo "=============================================="
echo " Extraction complete."
echo " Final file: $OUT/mimic_pca_cohort_features.csv"
echo "=============================================="