import pandas as pd
import numpy as np

# Load EHR
ehr_df = pd.read_csv("/oscar/data/class/biol1595_2595/students/hgle/mimic_data/final_cohort.csv")

# Build admission type vocabulary
all_types = set()
for s in ehr_df["admission_type_counts"].dropna():
    for item in s.split("|"):
        t = item.split(":")[0]
        all_types.add(t)

type_to_idx = {t:i for i,t in enumerate(sorted(all_types))}

def encode_admissions(s):
    vec = np.zeros(len(type_to_idx))
    if pd.isna(s):
        return vec
    for item in s.split("|"):
        t, count = item.split(":")
        vec[type_to_idx[t]] = int(count)
    return vec

# Encode EHR features
ehr_features = []
for _, row in ehr_df.iterrows():
    age = row["anchor_age"] / 100.0
    gender = 1 if row["gender"] == "M" else 0
    
    admissions_vec = encode_admissions(row["admission_type_counts"])
    
    feature = np.concatenate([[age, gender], admissions_vec])
    ehr_features.append(feature)

ehr_features = np.array(ehr_features)

print("EHR features shape:", ehr_features[0])