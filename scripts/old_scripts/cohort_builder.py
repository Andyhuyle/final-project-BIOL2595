# too memory intensive

import pandas as pd

def main():

    lab_file_name          = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/labevents.csv"
    lab_subset_output_file = "/oscar/data/class/biol1595_2595/students/hgle/pts_with_psa_labs.csv"

    total_pt_cohort = set()

    print("loading labs csv----------")
    labs_df = pd.read_csv(lab_file_name)

    print("filtering rows")
    df = labs_df.iloc[:, [1, 4, 14]]
    df.columns = ["subject_id", "itemid", "priority"]

    TARGET_ITEMID = 50974 # psa lab test

    df_filtered = df[df["itemid"] == TARGET_ITEMID].copy()

    df_filtered.to_csv(lab_subset_output_file, index=False)

    total_pt_cohort = set(df_filtered["subject_id"].unique())
    pd.DataFrame({"subject_id": list(total_pt_cohort)}).to_csv(
        "total_pt_cohort_ids.csv",
        index=False
    )

    print("Filtered rows:", len(df_filtered))
    print("Unique patients:", len(total_pt_cohort))
    print("Outputs saved to:", lab_subset_output_file)
    
main()
    
    



        



