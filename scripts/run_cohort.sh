#!/usr/bin/env bash
#SBATCH --job-name=mimic_cohort
#SBATCH --output=logs/cohort_%j.out
#SBATCH --error=logs/cohort_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# =============================================================================
# sbatch_build_cohort.sh
# SLURM wrapper for build_cohort.jl (Oscar / Brown HPC)
#
# Memory note: labevents.csv is read in full (3 cols only) then freed.
# Peak usage is roughly 10-15 GB for the file + working DataFrames.
# 128G gives comfortable headroom.
#
# Usage:
#   sbatch sbatch_build_cohort.sh
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these if your paths differ
# ---------------------------------------------------------------------------
MIMIC_DIR="/oscar/data/shared/ursa/mimic-iv"
MIMIC_VERSION="3.1"
OUT_DIR="/oscar/data/class/biol1595_2595/students/hgle/extracted"
SCRIPT_DIR="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/scripts"
TARGET=1700
SEED=42

# Julia package cache — keeps packages between jobs, avoids re-downloading
export JULIA_DEPOT_PATH="/oscar/data/class/biol1595_2595/students/hgle/.julia"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Job ID        : $SLURM_JOB_ID"
echo " Node          : $SLURMD_NODENAME"
echo " CPUs          : $SLURM_CPUS_PER_TASK"
echo " Memory        : 128G"
echo " MIMIC root    : $MIMIC_DIR"
echo " MIMIC version : $MIMIC_VERSION"
echo " hosp path     : $MIMIC_DIR/hosp/$MIMIC_VERSION"
echo " Script        : $SCRIPT_DIR/build_cohort.jl"
echo " Output dir    : $OUT_DIR"
echo " Julia depot   : $JULIA_DEPOT_PATH"
echo " Started       : $(date)"
echo "=============================================="
echo ""

mkdir -p logs
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Load Julia — check what is available if this fails
# Run: module avail julia
# ---------------------------------------------------------------------------
module load julia 2>/dev/null || {
    echo "ERROR: Could not load julia module."
    echo "Run 'module avail julia' on Oscar to find the correct module name"
    echo "and update this script accordingly."
    exit 1
}

echo "Julia   : $(which julia)"
echo "Version : $(julia --version)"
echo ""

# ---------------------------------------------------------------------------
# Verify the Julia script exists
# ---------------------------------------------------------------------------
if [[ ! -f "$SCRIPT_DIR/build_cohort.jl" ]]; then
    echo "ERROR: Julia script not found at $SCRIPT_DIR/build_cohort.jl"
    exit 1
fi

# ---------------------------------------------------------------------------
# Install / verify required packages into the depot
# This step is fast (no-op) if packages are already installed
# Required: CSV, DataFrames, ArgParse
# ---------------------------------------------------------------------------
# echo "Checking Julia packages (CSV, DataFrames, ArgParse)..."

# julia --project="$SCRIPT_DIR" -e '
#     import Pkg
#     Pkg.activate(".")
#     needed = ["CSV", "DataFrames", "ArgParse"]
#     installed = keys(Pkg.project().dependencies)
#     to_add = filter(p -> !(p in installed), needed)
#     if !isempty(to_add)
#         println("Installing: ", join(to_add, ", "))
#         Pkg.add(to_add)
#     else
#         println("All packages already installed.")
#     end
#     Pkg.instantiate()
#     # Quick import check
#     using CSV, DataFrames, ArgParse
#     println("Package check passed.")
# ' || {
#     echo "ERROR: Julia package setup failed."
#     echo "Try running interactively first:"
#     echo "  module load julia"
#     echo "  export JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
#     echo "  julia --project=$SCRIPT_DIR -e 'import Pkg; Pkg.add([\"CSV\",\"DataFrames\",\"ArgParse\"]); Pkg.instantiate()'"
#     exit 1
# }
# echo ""

# ---------------------------------------------------------------------------
# Run the cohort builder
# --threads=4 matches --cpus-per-task=4
# ---------------------------------------------------------------------------
echo "Starting cohort extraction..."
echo ""

julia --project="$SCRIPT_DIR" \
      --threads=4 \
      "$SCRIPT_DIR/build_cohort.jl" \
      --mimic   "$MIMIC_DIR" \
      --version "$MIMIC_VERSION" \
      --out     "$OUT_DIR" \
      --target  "$TARGET" \
      --seed    "$SEED"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo " Finished  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "=============================================="

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Job failed — check error log:"
    echo "  logs/cohort_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi

echo ""
echo "Output files:"
ls -lh "$OUT_DIR"/*.csv 2>/dev/null || echo "  No CSV files found in $OUT_DIR"