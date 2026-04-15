import pandas as pd

# -----------------------------
# Load cohort
# -----------------------------
cohort = pd.read_csv("final_cohort.csv")

# -----------------------------
# Load prostate cancer IDs (1 column, no header)
# -----------------------------
pc_ids = pd.read_csv(
    "prostate_cancer_icd910_subject_ids.csv",
    header=None,
    names=["subject_id"]
)

# -----------------------------
# Convert to set (fast lookup)
# -----------------------------
pc_set = set(pc_ids["subject_id"])

# -----------------------------
# CASES: prostate cancer patients
# -----------------------------
pc_cohort = cohort[cohort["subject_id"].isin(pc_set)].copy()

# -----------------------------
# CONTROLS: NOT prostate cancer
# -----------------------------
control_cohort = cohort[~cohort["subject_id"].isin(pc_set)].copy()

# -----------------------------
# Save outputs
# -----------------------------
pc_cohort.to_csv("pc_cases.csv", index=False)
control_cohort.to_csv("pc_controls.csv", index=False)

# -----------------------------
# Summary
# -----------------------------
print("Total cohort:", len(cohort))
print("Cases (PCa):", len(pc_cohort))
print("Controls:", len(control_cohort))