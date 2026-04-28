import pandas as pd

# ---------------------------------------------------------------------------
# Paths — edit these
# ---------------------------------------------------------------------------
PSA_CSV = "/oscar/data/class/biol1595_2595/students/hgle/cleaned_lab.csv"
OUT_DIR = "/oscar/data/class/biol1595_2595/students/hgle/extracted"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(PSA_CSV, header=None,
                 names=["subject_id", "charttime", "valuenum", "valueuom"])

df["subject_id"] = df["subject_id"].astype(str).str.strip()
df["valuenum"]   = pd.to_numeric(df["valuenum"], errors="coerce")

print(f"Total PSA rows     : {len(df):,}")
print(f"Unique patients    : {df['subject_id'].nunique():,}")
print(f"Missing valuenum   : {df['valuenum'].isna().sum():,}")
print()

# ---------------------------------------------------------------------------
# Aggregate — max PSA per patient
# ---------------------------------------------------------------------------
agg = df.groupby("subject_id")["valuenum"].max().reset_index()
agg.columns = ["subject_id", "psa_max"]

# ---------------------------------------------------------------------------
# Assign severity class
# ---------------------------------------------------------------------------
def assign_severity(psa):
    if pd.isna(psa):  return "unknown"
    if psa < 4.0:     return "low"
    if psa <= 20.0:   return "moderate"
    return "high"

agg["severity_class"] = agg["psa_max"].apply(assign_severity)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
counts = agg["severity_class"].value_counts()
total  = len(agg)

print("PSA Severity Distribution (max PSA per patient)")
print("-" * 45)
for cls, threshold in [("low",      "PSA < 4.0"),
                        ("moderate", "4.0 <= PSA <= 20.0"),
                        ("high",     "PSA > 20.0"),
                        ("unknown",  "no numeric value")]:
    n   = counts.get(cls, 0)
    pct = n / total * 100
    print(f"  {cls:<10} {threshold:<22} {n:>6,}  ({pct:.1f}%)")
print("-" * 45)
print(f"  {'TOTAL':<10} {'':<22} {total:>6,}")
print()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = f"{OUT_DIR}/psa_severity_classes.csv"
agg.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")