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
SCRIPT="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/multimodal_model/shared_embedding.py"
VENV="/oscar/data/class/biol1595_2595/students/hgle/ai_heathcare_venv"
RUN_NAME="multimodal_pca"
LOG_DIR="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/multimodal_model/logs"

mkdir -p "$LOG_DIR"
mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.err"

# =========================
# ENVIRONMENT
# =========================
module load python/3.9.0
module load cuda 2>/dev/null || true

source "$VENV/bin/activate"

echo "Python  : $(which python)"
echo "Version : $(python --version)"
echo ""

# Quick sanity checks
python -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU      : {torch.cuda.get_device_name(0)}')
"

# =========================
# RUN
# =========================
echo ""
echo "Starting training run : $RUN_NAME"
echo "Log  : $LOG_FILE"
echo "Error: $ERR_FILE"
echo ""

python -u "$SCRIPT" > "$LOG_FILE" 2> "$ERR_FILE"
EXIT_CODE=$?

# =========================
# RESULT
# =========================
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
    echo "Log saved to: $LOG_FILE"
else
    echo "Training FAILED (exit code $EXIT_CODE)"
    echo "Check error log: $ERR_FILE"
    echo ""
    echo "Last 20 lines of error log:"
    tail -20 "$ERR_FILE"
    exit $EXIT_CODE
fi