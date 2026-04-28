#!/bin/bash
#SBATCH --job-name=late_fusion_pca
#SBATCH --mail-user=huy_le@brown.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

SCRIPT="/oscar/data/class/biol1595_2595/students/hgle/final-project-BIOL2595/multimodal_model/late_fusion_baseline.py"
VENV="/oscar/data/class/biol1595_2595/students/hgle/ai_heathcare_venv"
LOG_DIR="/oscar/data/class/biol1595_2595/students/hgle/logs"

mkdir -p "$LOG_DIR" logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/late_fusion_${TIMESTAMP}.log"
ERR_FILE="${LOG_DIR}/late_fusion_${TIMESTAMP}.err"

module load python/3.9.0
module load cuda 2>/dev/null || true
source "$VENV/bin/activate"

echo "Python  : $(which python)"
echo "Version : $(python --version)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo ""

python -u "$SCRIPT" > "$LOG_FILE" 2> "$ERR_FILE"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Late fusion training completed."
    echo "Log: $LOG_FILE"
else
    echo "FAILED (exit $EXIT_CODE) — check: $ERR_FILE"
    tail -20 "$ERR_FILE"
    exit $EXIT_CODE
fi