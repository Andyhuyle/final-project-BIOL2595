#!/bin/bash
#SBATCH --job-name=multimodal_pca
#SBATCH --mail-user=huy_le@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =========================
# CONFIG
# =========================

SCRIPT="/oscar/data/class/biol1595_2595/students/hgle/multimodal_model/train.py"
RUN_NAME="multimodal_pca"
LOG_DIR="/oscar/data/class/biol1595_2595/students/hgle/logs"

mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.err"

# =========================
# RUN
# =========================

echo "Starting training run: $RUN_NAME"
echo "Logs: $LOG_FILE"
echo "Errors: $ERR_FILE"

python -u $SCRIPT > $LOG_FILE 2> $ERR_FILE

EXIT_CODE=$?

# =========================
# RESULT
# =========================

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training FAILED. Check error log: $ERR_FILE"
fi