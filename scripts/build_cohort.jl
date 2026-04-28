"""
build_control_cohort.jl

Builds a balanced control cohort (no PCa) to match ~1,700 PCa cases.

Cohort universe : all patients with >= 1 PSA lab test in labevents.csv
Strategy 1      : PSA-tested patients with NO C61 / ICD-9 185 diagnosis
Strategy 2      : Urological encounter patients (BPH, prostatitis, screening)
                  with no PCa ICD — only used if Strategy 1 falls short

MIMIC IV path structure on Oscar:
    /oscar/data/shared/ursa/mimic-iv/hosp/3.1/labevents.csv

Usage:
    julia build_control_cohort.jl \\
        --mimic   /oscar/data/shared/ursa/mimic-iv \\
        --version 3.1 \\
        --out     /oscar/data/class/biol1595_2595/students/hgle/extracted \\
        --target  1700 \\
        --seed    42

Outputs (all written to --out directory):
    psa_cohort.csv           All PSA-tested patients (universe)
    pca_cases.csv            Confirmed PCa subjects  (has_pca=1)
    controls_strategy1.csv   PSA-tested, no PCa ICD
    controls_strategy2.csv   Urology encounter, no PCa ICD (fallback)
    controls_final.csv       Downsampled to --target, ready for modelling
"""

using CSV
using DataFrames
using ArgParse
using Random
using Dates
using Base: GC

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log(msg::String) = println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")
flush_log() = flush(stdout)

# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------
function hosp_path(mimic_dir::String, version::String, filename::String)::String
    path = joinpath(mimic_dir, "hosp", version, filename)
    if !isfile(path)
        error("Expected MIMIC file not found: $path\nCheck --mimic and --version arguments.")
    end
    return path
end

# ---------------------------------------------------------------------------
# Safe CSV reader — consistent string types, lowercase column names
# ---------------------------------------------------------------------------
function read_csv_safe(path::String; select=nothing)::DataFrame
    log("  Reading: $path")
    flush_log()
    df = if isnothing(select)
        CSV.read(path, DataFrame; stringtype=String, missingstring=["", "NA", "\\N", "nan", "NULL"])
    else
        CSV.read(path, DataFrame; stringtype=String, missingstring=["", "NA", "\\N", "nan", "NULL"], select=select)
    end
    rename!(df, Symbol.(lowercase.(string.(names(df)))))
    # Convert any SubString columns to String
    for col in propertynames(df)
        if eltype(df[!, col]) <: Union{SubString, Missing} || eltype(df[!, col]) == SubString{String}
            df[!, col] = String.(coalesce.(df[!, col], ""))
        end
    end
    for col in (:subject_id, :hadm_id)
        if col in propertynames(df)
            df[!, col] = strip.(coalesce.(string.(df[!, col]), ""))
            df[!, col] = replace(df[!, col], "missing" => missing)
        end
    end
    return df
end

# ---------------------------------------------------------------------------
# ICD matching
# ---------------------------------------------------------------------------
function is_pca(code::Union{AbstractString, Missing}, version::Union{AbstractString, Missing})::Bool
    (ismissing(code) || ismissing(version)) && return false
    c = uppercase(replace(strip(String(code)), "." => ""))
    v = strip(String(version))
    v == "10" && return startswith(c, "C61")
    v == "9"  && return startswith(c, "185")
    return false
end

function is_urology(code::Union{AbstractString, Missing}, version::Union{AbstractString, Missing})::Bool
    (ismissing(code) || ismissing(version)) && return false
    c = uppercase(replace(strip(String(code)), "." => ""))
    v = strip(String(version))
    if v == "10"
        return any(startswith(c, p) for p in ("N40", "N41", "Z125", "Z8042"))
    elseif v == "9"
        return any(startswith(c, p) for p in ("600", "601", "V7644"))
    end
    return false
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main(mimic_dir::String, version::String, out_dir::String,
              target::Int, seed::Int)

    mkpath(out_dir)
    hp(f) = hosp_path(mimic_dir, version, f)   # convenience alias

    println("=" ^ 60)
    println("  Control Cohort Builder")
    println("  MIMIC root    : $mimic_dir")
    println("  MIMIC version : $version")
    println("  hosp path     : $(joinpath(mimic_dir, "hosp", version))")
    println("  Output dir    : $out_dir")
    println("  Target N      : $target")
    println("  Random seed   : $seed")
    println("=" ^ 60)
    println()

    # -----------------------------------------------------------------------
    # Step 1 — Detect PSA itemid from d_labitems.csv
    # -----------------------------------------------------------------------
    log("[1/6] Detecting PSA itemid from d_labitems.csv...")

    labitems = read_csv_safe(hp("d_labitems.csv"); select=["itemid", "label"])

    psa_mask = [occursin(r"prostate specific antigen|^psa$"i,
                         coalesce(l, "")) for l in labitems.label]
    psa_rows = labitems[psa_mask, :]

    psa_itemid::String = if nrow(psa_rows) == 0
        log("  WARNING: PSA itemid not found — falling back to 50813")
        "50813"
    else
        log("  PSA label  : $(psa_rows[1, :label])")
        id = string(psa_rows[1, :itemid])
        log("  PSA itemid : $id")
        id
    end
    println()

    # -----------------------------------------------------------------------
    # Step 2 — Build PSA cohort from labevents.csv
    #
    # MIMIC IV 3.1 labevents.csv columns:
    #   labevent_id, subject_id, hadm_id, specimen_id, itemid,
    #   order_provider_id, charttime, storetime, value, valuenum,
    #   valueuom, ref_range_lower, ref_range_upper, flag, priority, comments
    #
    # hadm_id is NULL for outpatient draws — handled explicitly below.
    # We select only subject_id, hadm_id, itemid by name to minimize memory.
    # CSV.Chunks is avoided — use CSV.File which is stable across versions.
    # -----------------------------------------------------------------------
    log("[2/6] Scanning labevents.csv for patients with >= 1 PSA test...")
    log("  Reading 3 columns only (subject_id, hadm_id, itemid)...")
    flush_log()

    BLANK = Set(["", "missing", "NA", "\\N", "<NA>", "nothing", "NULL"])

    # Read only the three needed columns — keeps ~64G job within memory
    lab_df = CSV.read(
        hp("labevents.csv"),
        DataFrame;
        stringtype    = String,
        missingstring = ["", "NA", "\\N", "nan", "NULL"],
        select        = ["subject_id", "hadm_id", "itemid"],
        silencewarnings = true
    )
    rename!(lab_df, Symbol.(strip.(lowercase.(string.(names(lab_df))))))
    log("  Loaded $(nrow(lab_df)) rows — filtering to PSA itemid $psa_itemid...")
    flush_log()

    # Filter to PSA rows only
    psa_df = filter(row -> coalesce(string(row.itemid), "") == psa_itemid, lab_df)
    lab_df = nothing   # free memory immediately
    GC.gc()
    log("  Found $(nrow(psa_df)) PSA lab rows across all patients")

    # Build subject -> best hadm_id dict
    # Prefer inpatient hadm_id over blank (outpatient draw)
    psa_subjects = Dict{String, Union{String, Missing}}()

    for row in eachrow(psa_df)
        subj = strip(coalesce(string(row.subject_id), ""))
        hadm = strip(coalesce(string(row.hadm_id),    ""))

        (isempty(subj) || subj == "missing") && continue

        if !haskey(psa_subjects, subj)
            psa_subjects[subj] = hadm ∈ BLANK ? missing : hadm
        elseif hadm ∉ BLANK && ismissing(psa_subjects[subj])
            psa_subjects[subj] = hadm
        end
    end
    psa_df = nothing
    GC.gc()

    log("  Scan complete: $(length(psa_subjects)) unique PSA patients found")

    psa_cohort = DataFrame(
        subject_id = collect(keys(psa_subjects)),
        hadm_id    = collect(values(psa_subjects))
    )

    CSV.write(joinpath(out_dir, "psa_cohort.csv"), psa_cohort)
    log("  PSA-tested universe : $(nrow(psa_cohort)) patients")
    log("  Written -> psa_cohort.csv")
    println()

    # -----------------------------------------------------------------------
    # Step 3 — Identify confirmed PCa subjects from diagnoses_icd.csv
    # -----------------------------------------------------------------------
    log("[3/6] Identifying confirmed PCa subjects from diagnoses_icd...")

    diag = read_csv_safe(hp("diagnoses_icd.csv");
                          select=["subject_id", "hadm_id", "icd_code", "icd_version"])

    diag[!, :icd_code]    = strip.(coalesce.(string.(diag.icd_code), ""))
    diag[!, :icd_version] = strip.(coalesce.(string.(diag.icd_version), ""))

    pca_mask_diag = [is_pca(r.icd_code, r.icd_version) for r in eachrow(diag)]
    pca_subject_ids = Set(diag[pca_mask_diag, :subject_id])

    pca_cases = DataFrame(
        subject_id = collect(pca_subject_ids),
        has_pca    = ones(Int, length(pca_subject_ids))
    )
    sort!(pca_cases, :subject_id)
    CSV.write(joinpath(out_dir, "pca_cases.csv"), pca_cases)

    log("  Confirmed PCa subjects : $(length(pca_subject_ids))")
    log("  Written -> pca_cases.csv")
    println()

    # -----------------------------------------------------------------------
    # Step 4 — Strategy 1: PSA-tested, no PCa ICD code
    # -----------------------------------------------------------------------
    log("[4/6] Strategy 1 — PSA-tested patients with no PCa ICD code...")

    s1 = filter(row -> !(row.subject_id in pca_subject_ids), psa_cohort)
    s1 = copy(s1)
    s1[!, :control_source] .= "strategy1_psa_no_pca"

    CSV.write(joinpath(out_dir, "controls_strategy1.csv"), s1)
    log("  Strategy 1 controls : $(nrow(s1))  (target: $target)")
    println()

    # -----------------------------------------------------------------------
    # Step 5 — Strategy 2 (fallback): urological encounter, no PCa ICD
    # -----------------------------------------------------------------------
    log("[5/6] Strategy 2 — Urological encounter patients (fallback)...")

    s2 = if nrow(s1) >= target
        log("  Strategy 1 meets target — skipping Strategy 2 scan.")
        DataFrame(subject_id=String[], hadm_id=Union{String,Missing}[],
                  control_source=String[])
    else
        shortfall = target - nrow(s1)
        log("  Strategy 1 is $shortfall short — scanning for urology ICD codes...")
        flush_log()

        s1_ids  = Set(s1.subject_id)
        uro_mask = [is_urology(r.icd_code, r.icd_version) for r in eachrow(diag)]
        uro = diag[uro_mask, [:subject_id, :hadm_id]]

        # Exclude PCa patients and Strategy 1 subjects
        uro = filter(row ->
            !(row.subject_id in pca_subject_ids) &&
            !(row.subject_id in s1_ids), uro)

        # One row per subject
        uro = combine(groupby(uro, :subject_id), :hadm_id => first => :hadm_id)
        uro[!, :control_source] .= "strategy2_urology_no_pca"

        log("  Strategy 2 additional controls : $(nrow(uro))")
        uro
    end

    CSV.write(joinpath(out_dir, "controls_strategy2.csv"), s2)
    println()

    # -----------------------------------------------------------------------
    # Step 6 — Merge and downsample to target
    # -----------------------------------------------------------------------
    log("[6/6] Merging and downsampling to $target controls...")

    combined   = vcat(s1, s2; cols=:union)
    combined_n = nrow(combined)
    log("  Combined pool : $combined_n")

    Random.seed!(seed)
    final = if combined_n < target
        log("  WARNING: Pool ($combined_n) < target ($target). Using all available controls.")
        combined
    else
        combined[shuffle(1:combined_n)[1:target], :]
    end

    CSV.write(joinpath(out_dir, "controls_final.csv"), final)

    final_n  = nrow(final)
    s1_final = sum(final.control_source .== "strategy1_psa_no_pca")
    s2_final = sum(final.control_source .== "strategy2_urology_no_pca")
    ratio    = round(final_n / max(length(pca_subject_ids), 1); digits=2)

    println()
    println("=" ^ 60)
    println("  Cohort Balance Summary")
    println("=" ^ 60)
    println("  PSA-tested universe            : $(lpad(nrow(psa_cohort), 8))")
    println("  PCa cases        (has_pca=1)   : $(lpad(length(pca_subject_ids), 8))")
    println("  Controls         (has_pca=0)   : $(lpad(final_n, 8))")
    println("    Strategy 1  (PSA, no ICD)    : $(lpad(s1_final, 8))")
    println("    Strategy 2  (Urology)        : $(lpad(s2_final, 8))")
    println("  Case:Control ratio             :     1:$ratio")
    println()
    println("  Next step: run proxy score binning on")
    println("  controls_final.csv + pca_cases.csv to assign")
    println("  low / moderate / high severity labels")
    println("  before merging with PANDA image classes.")
    println("=" ^ 60)
    println()

    log("Output files:")
    for fname in ("psa_cohort.csv", "pca_cases.csv",
                  "controls_strategy1.csv", "controls_strategy2.csv",
                  "controls_final.csv")
        fpath = joinpath(out_dir, fname)
        size_kb = round(filesize(fpath) / 1024; digits=1)
        log("  $fpath  ($size_kb KB)")
    end

    println()
    log("Done.")
end

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
function parse_args_custom()
    s = ArgParseSettings(description="Build balanced PCa / control cohort from MIMIC IV")
    @add_arg_table! s begin
        "--mimic"
            help    = "MIMIC IV root directory"
            default = "/oscar/data/shared/ursa/mimic-iv"
        "--version"
            help    = "MIMIC IV hosp version subdirectory"
            default = "3.1"
        "--out"
            help    = "Output directory for generated CSVs"
            default = "/oscar/data/class/biol1595_2595/students/hgle/extracted"
        "--target"
            help    = "Target number of control patients"
            arg_type = Int
            default  = 1700
        "--seed"
            help    = "Random seed for reproducible downsampling"
            arg_type = Int
            default  = 42
    end
    return parse_args(s)
end

args = parse_args_custom()
main(args["mimic"], args["version"], args["out"], args["target"], args["seed"])