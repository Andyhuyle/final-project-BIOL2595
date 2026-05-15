import pandas as pd

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(
    "total_cohort_admissions.csv",
    usecols=["subject_id", "admission_type", "race"]
)

# -----------------------------
# Clean strings
# -----------------------------
df["admission_type"] = df["admission_type"].astype(str).str.strip()
df["race"] = df["race"].astype(str).str.strip()

# -----------------------------
# STEP 1: count admission types per patient
# -----------------------------
counts = (
    df.groupby(["subject_id", "admission_type"])
    .size()
    .reset_index(name="count")
)

# -----------------------------
# STEP 2: collapse counts into one column
# -----------------------------
adm_str = (
    counts.groupby("subject_id")
    .apply(lambda x: "|".join(
        f"{row['admission_type']}:{row['count']}"
        for _, row in x.iterrows()
    ))
    .reset_index(name="admission_type_counts")
)

# -----------------------------
# STEP 3: get race per patient
# (assumes race is consistent per subject)
# -----------------------------
race_df = (
    df.groupby("subject_id")["race"]
    .first()
    .reset_index()
)

# -----------------------------
# STEP 4: merge everything
# -----------------------------
final_df = adm_str.merge(race_df, on="subject_id", how="left")

# -----------------------------
# Save output
# -----------------------------
final_df.to_csv("patient_admission_features.csv", index=False)

print("Saved:", len(final_df), "patients")