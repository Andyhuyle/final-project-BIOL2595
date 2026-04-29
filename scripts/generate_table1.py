"""
generate_table1.py

Generates Table 1 (cohort characteristics) for the prostate cancer
severity study. Reports the full 12,701-patient cohort stratified by
PSA-derived severity class (low / moderate / high).

Follows TRIPOD/JAMIA reporting standards:
    - Continuous variables: median (IQR)
    - Categorical variables: n (%)
    - Statistical comparisons: Kruskal-Wallis for continuous,
      Chi-square for categorical
    - p-values reported but not used for selection

Inputs:
    ehr_features.csv        full cohort with raw + normalized features
                            (output of build_ehr_features.py)

Output:
    table1.csv              machine-readable table
    table1_formatted.txt    formatted for copy-paste into paper

Usage:
    python generate_table1.py \
        --ehr   /oscar/.../extracted/ehr_features.csv \
        --out   /oscar/.../outputs
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

SEVERITY_ORDER  = ["low", "moderate", "high"]
SEVERITY_LABELS = {"low": "Low\n(PSA < 4.0)",
                   "moderate": "Moderate\n(PSA 4-20)",
                   "high": "High\n(PSA > 20)"}

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def median_iqr(series):
    """Return median (Q1-Q3) string."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    med = s.median()
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


def n_pct(series, total):
    """Return n (%) string."""
    n   = series.sum()
    pct = n / total * 100 if total > 0 else 0
    return f"{int(n):,} ({pct:.1f}%)"


def kruskal_p(df, col, group_col):
    """Kruskal-Wallis p-value for continuous variable across 3 groups."""
    groups = [df.loc[df[group_col] == g, col].dropna().values
              for g in SEVERITY_ORDER]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    try:
        _, p = stats.kruskal(*groups)
        return p
    except Exception:
        return np.nan


def chisq_p(df, col, group_col):
    """Chi-square p-value for categorical variable across 3 groups."""
    try:
        ct  = pd.crosstab(df[col], df[group_col])
        _, p, _, _ = stats.chi2_contingency(ct)
        return p
    except Exception:
        return np.nan


def fmt_p(p):
    """Format p-value for display."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(ehr_path, panda_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading cohort data...")
    df = pd.read_csv(ehr_path)
    df.columns = df.columns.str.strip().str.lower()

    # Ensure severity_class is present and lowercase
    df["severity_class"] = df["severity_class"].str.strip().str.lower()
    df = df[df["severity_class"].isin(SEVERITY_ORDER)].copy()

    # Parse numeric columns
    numeric_cols = ["psa_max", "psa_order_count", "procedure_count",
                    "distinct_med_count", "los_days", "anchor_age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse gender if present
    if "gender" in df.columns:
        df["is_male"] = df["gender"].str.upper().str.strip() == "M"

    # Parse race if present
    if "race" in df.columns:
        df["race_clean"] = df["race"].str.strip().str.title()

    total = len(df)
    groups = {s: df[df["severity_class"] == s] for s in SEVERITY_ORDER}
    n_groups = {s: len(g) for s, g in groups.items()}

    print(f"Total cohort: {total:,} patients")
    for s in SEVERITY_ORDER:
        print(f"  {s:<10}: {n_groups[s]:,} ({n_groups[s]/total*100:.1f}%)")
    print()

    # -----------------------------------------------------------------------
    # Load PANDA data
    # -----------------------------------------------------------------------
    print("Loading PANDA labels...")
    panda_df = pd.read_csv(panda_path)
    panda_df.columns = panda_df.columns.str.strip().str.lower()

    def isup_to_severity(isup):
        if isup <= 1: return "low"
        if isup <= 3: return "moderate"
        return "high"

    panda_df["severity_class"] = panda_df["isup_grade"].apply(isup_to_severity)

    panda_total   = len(panda_df)
    panda_groups  = {s: panda_df[panda_df["severity_class"] == s]
                     for s in SEVERITY_ORDER}
    panda_n       = {s: len(g) for s, g in panda_groups.items()}

    print(f"PANDA total: {panda_total:,} images")
    for s in SEVERITY_ORDER:
        print(f"  {s:<10}: {panda_n[s]:,} ({panda_n[s]/panda_total*100:.1f}%)")
    print()

    # -----------------------------------------------------------------------
    # Build rows
    # -----------------------------------------------------------------------
    rows = []

    def add_row(label, values_by_group, p_val, indent=False):
        prefix = "  " if indent else ""
        rows.append({
            "Characteristic"  : prefix + label,
            "Overall"         : values_by_group.get("overall", ""),
            "Low (n)"         : values_by_group.get("low", ""),
            "Moderate (n)"    : values_by_group.get("moderate", ""),
            "High (n)"        : values_by_group.get("high", ""),
            "p-value"         : fmt_p(p_val),
        })

    # --- N ---
    add_row("N (%)", {
        "overall" : f"{total:,}",
        "low"     : f"{n_groups['low']:,} ({n_groups['low']/total*100:.1f}%)",
        "moderate": f"{n_groups['moderate']:,} ({n_groups['moderate']/total*100:.1f}%)",
        "high"    : f"{n_groups['high']:,} ({n_groups['high']/total*100:.1f}%)",
    }, p_val=np.nan)

    # --- Age ---
    if "anchor_age" in df.columns:
        p = kruskal_p(df, "anchor_age", "severity_class")
        add_row("Age, years — median (IQR)", {
            "overall" : median_iqr(df["anchor_age"]),
            **{s: median_iqr(groups[s]["anchor_age"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- Gender ---
    if "is_male" in df.columns:
        p = chisq_p(df, "is_male", "severity_class")
        add_row("Male sex — n (%)", {
            "overall" : n_pct(df["is_male"], total),
            **{s: n_pct(groups[s]["is_male"], n_groups[s]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- Race ---
    if "race_clean" in df.columns:
        top_races = df["race_clean"].value_counts().head(5).index.tolist()
        add_row("Race — n (%)", {}, p_val=chisq_p(df, "race_clean", "severity_class"))
        for race in top_races:
            mask = df["race_clean"] == race
            p_r  = np.nan
            add_row(race, {
                "overall" : n_pct(mask, total),
                **{s: n_pct(groups[s]["race_clean"] == race, n_groups[s])
                   for s in SEVERITY_ORDER}
            }, p_val=p_r, indent=True)

    # --- PSA max ---
    if "psa_max" in df.columns:
        p = kruskal_p(df, "psa_max", "severity_class")
        add_row("PSA max, ng/mL — median (IQR)", {
            "overall" : median_iqr(df["psa_max"]),
            **{s: median_iqr(groups[s]["psa_max"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- PSA order count ---
    if "psa_order_count" in df.columns:
        p = kruskal_p(df, "psa_order_count", "severity_class")
        add_row("PSA tests ordered — median (IQR)", {
            "overall" : median_iqr(df["psa_order_count"]),
            **{s: median_iqr(groups[s]["psa_order_count"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- Procedure count ---
    if "procedure_count" in df.columns:
        p = kruskal_p(df, "procedure_count", "severity_class")
        add_row("Procedures per admission — median (IQR)", {
            "overall" : median_iqr(df["procedure_count"]),
            **{s: median_iqr(groups[s]["procedure_count"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- Medication count ---
    if "distinct_med_count" in df.columns:
        p = kruskal_p(df, "distinct_med_count", "severity_class")
        add_row("Distinct medications — median (IQR)", {
            "overall" : median_iqr(df["distinct_med_count"]),
            **{s: median_iqr(groups[s]["distinct_med_count"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # --- LOS ---
    if "los_days" in df.columns:
        p = kruskal_p(df, "los_days", "severity_class")
        add_row("Admission LOS, days — median (IQR)", {
            "overall" : median_iqr(df["los_days"]),
            **{s: median_iqr(groups[s]["los_days"]) for s in SEVERITY_ORDER}
        }, p_val=p)

    # -----------------------------------------------------------------------
    # PANDA image cohort rows
    # -----------------------------------------------------------------------

    # Section header
    rows.append({"Characteristic": "", "Overall": "", "Low (n)": "",
                 "Moderate (n)": "", "High (n)": "", "p-value": ""})
    rows.append({"Characteristic": "IMAGE COHORT (PANDA)",
                 "Overall": f"n={panda_total:,}",
                 "Low (n)": f"n={panda_n['low']:,}",
                 "Moderate (n)": f"n={panda_n['moderate']:,}",
                 "High (n)": f"n={panda_n['high']:,}",
                 "p-value": ""})

    # ISUP grade — one summary row per severity class showing grade range
    # (not broken out by severity column — would be all zeros by definition
    # since ISUP grades map exclusively to severity classes)
    isup_ranges = {
        "low"     : "ISUP 0–1 (benign / 3+3)",
        "moderate": "ISUP 2–3 (3+4 / 4+3)",
        "high"    : "ISUP 4–5 (4+4 and above)",
    }
    rows.append({
        "Characteristic": "  ISUP grade range",
        "Overall"       : "0–5",
        "Low (n)"       : isup_ranges["low"],
        "Moderate (n)"  : isup_ranges["moderate"],
        "High (n)"      : isup_ranges["high"],
        "p-value"       : "—",
    })

    # ISUP grade counts — overall only, not broken out by severity
    for grade in sorted(panda_df["isup_grade"].unique()):
        mask = panda_df["isup_grade"] == grade
        rows.append({
            "Characteristic": f"  ISUP grade {grade} — n (%)",
            "Overall"       : n_pct(mask, panda_total),
            "Low (n)"       : "—" if grade <= 1 else "",
            "Moderate (n)"  : "—" if grade in [2, 3] else "",
            "High (n)"      : "—" if grade >= 4 else "",
            "p-value"       : "",
        })

    # Gleason score — show within-class distribution (adds real info)
    # Groups: benign (0+0/negative), low (3+3), moderate (3+4/4+3), high (4+4+)
    gleason_groups = {
        "Benign/negative (0+0, negative)": ["0+0", "negative"],
        "3+3"  : ["3+3"],
        "3+4"  : ["3+4"],
        "4+3"  : ["4+3"],
        "4+4"  : ["4+4"],
        "4+5 or higher": ["4+5", "5+4", "5+5", "3+5", "5+3"],
    }
    for label, gs_list in gleason_groups.items():
        mask     = panda_df["gleason_score"].isin(gs_list)
        low_mask = panda_groups["low"]["gleason_score"].isin(gs_list)
        mod_mask = panda_groups["moderate"]["gleason_score"].isin(gs_list)
        hi_mask  = panda_groups["high"]["gleason_score"].isin(gs_list)
        rows.append({
            "Characteristic": f"  {label} — n (%)",
            "Overall"       : n_pct(mask, panda_total),
            "Low (n)"       : n_pct(low_mask, panda_n["low"]),
            "Moderate (n)"  : n_pct(mod_mask, panda_n["moderate"]),
            "High (n)"      : n_pct(hi_mask,  panda_n["high"]),
            "p-value"       : "",
        })

    # Data source (Karolinska vs Radboud) if column exists
    if "data_provider" in panda_df.columns:
        for provider in panda_df["data_provider"].unique():
            mask = panda_df["data_provider"] == provider
            rows.append({
                "Characteristic": f"  {provider} — n (%)",
                "Overall"       : n_pct(mask, panda_total),
                "Low (n)"       : n_pct(panda_groups["low"]["data_provider"] == provider,
                                        panda_n["low"]),
                "Moderate (n)"  : n_pct(panda_groups["moderate"]["data_provider"] == provider,
                                        panda_n["moderate"]),
                "High (n)"      : n_pct(panda_groups["high"]["data_provider"] == provider,
                                        panda_n["high"]),
                "p-value"       : "",
            })

    # Balanced training set summary
    rows.append({"Characteristic": "", "Overall": "", "Low (n)": "",
                 "Moderate (n)": "", "High (n)": "", "p-value": ""})
    rows.append({
        "Characteristic": "BALANCED TRAINING SET",
        "Overall"       : "n=3,440 per modality",
        "Low (n)"       : "n=1,376",
        "Moderate (n)"  : "n=1,376",
        "High (n)"      : "n=688",
        "p-value"       : "",
    })
    rows.append({
        "Characteristic": "  EHR patients — n",
        "Overall": "3,440", "Low (n)": "1,376",
        "Moderate (n)": "1,376", "High (n)": "688", "p-value": "",
    })
    rows.append({
        "Characteristic": "  PANDA images — n",
        "Overall": "3,440", "Low (n)": "1,376",
        "Moderate (n)": "1,376", "High (n)": "688", "p-value": "",
    })

    # -----------------------------------------------------------------------
    # Build dataframe
    # -----------------------------------------------------------------------
    table = pd.DataFrame(rows)

    # Rename columns to final labels
    table = table.rename(columns={
        "Low (n)"     : f"Low (n={n_groups['low']:,})",
        "Moderate (n)": f"Moderate (n={n_groups['moderate']:,})",
        "High (n)"    : f"High (n={n_groups['high']:,})",
        "Overall"     : f"Overall (n={total:,})",
    })

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "table1.csv")
    table.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    # -----------------------------------------------------------------------
    # Print formatted table
    # -----------------------------------------------------------------------
    txt_path = os.path.join(out_dir, "table1_formatted.txt")
    with open(txt_path, "w") as f:

        header = (
            f"Table 1. Cohort Characteristics by PSA-Derived Severity Class\n"
            f"{'='*100}\n"
            f"Data source: MIMIC IV (v3.1). Cohort defined as patients with ≥1 PSA lab value\n"
            f"and at least one procedure and medication record (n={total:,}).\n"
            f"Continuous variables reported as median (IQR). Categorical as n (%).\n"
            f"p-values from Kruskal-Wallis (continuous) or chi-square (categorical).\n"
            f"{'='*100}\n\n"
        )
        f.write(header)
        print(header, end="")

        col_w = [45, 18, 18, 18, 18, 10]
        cols  = list(table.columns)
        header_row = "".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
        sep = "-" * sum(col_w)

        f.write(header_row + "\n")
        f.write(sep + "\n")
        print(header_row)
        print(sep)

        for _, row in table.iterrows():
            line = "".join(
                f"{str(v):<{w}}" for v, w in zip(row.values, col_w)
            )
            f.write(line + "\n")
            print(line)

        f.write(sep + "\n")
        print(sep)

        footer = (
            f"\nIQR = interquartile range; LOS = length of stay; PSA = prostate-specific antigen.\n"
            f"PSA severity thresholds: Low <4.0 ng/mL, Moderate 4.0–20.0 ng/mL, High >20.0 ng/mL\n"
            f"(American Urological Association guidelines).\n"
        )
        f.write(footer)
        print(footer)

    print(f"Formatted table saved -> {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Table 1 for PCa cohort")
    parser.add_argument(
        "--ehr",
        default="/oscar/data/class/biol1595_2595/students/hgle/extracted/ehr_features.csv",
        help="Path to ehr_features.csv (full cohort, output of build_ehr_features.py)"
    )
    parser.add_argument(
        "--panda",
        default="/oscar/data/shared/ursa/kaggle_panda/train.csv",
        help="Path to PANDA train.csv (full dataset before downsampling)"
    )
    parser.add_argument(
        "--out",
        default="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/outputs/table1",
        help="Output directory"
    )
    args = parser.parse_args()
    main(args.ehr, args.panda, args.out)