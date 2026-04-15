import pandas as pd

# -----------------------------
# Load datasets
# -----------------------------
adm = pd.read_csv("patient_admission_features.csv")
demo = pd.read_csv("total_cohort_demographics.csv")

# -----------------------------
# Inner join (keep only matching patients)
# -----------------------------
merged = demo.merge(adm, on="subject_id", how="inner")

# -----------------------------
# Save output
# -----------------------------
merged.to_csv("final_cohort.csv", index=False)

print("Final cohort size:", len(merged))