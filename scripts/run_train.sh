#!/bin/bash -l
#SBATCH --job-name=train_polo_%j
#SBATCH --chdir=/bigdata/rhome/tfrw2023/Code/PoloTagger
#SBATCH --output=scripts/train_%j.log
#SBATCH --error=scripts/train_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Shell‐level debug
date
hostname
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] Job name: $SLURM_JOB_NAME"

# Use the correct Python interpreter
PYTHON_PATH="/bigdata/rhome/tfrw2023/.conda/envs/PoloTagger/bin/python3.10"
if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "[ERROR] Python not found at $PYTHON_PATH"
    exit 1
fi
echo "[DEBUG] Using Python: $PYTHON_PATH"
echo "[DEBUG] Python version: $($PYTHON_PATH --version)"

# Sanity‐check PyTorch/CUDA
$PYTHON_PATH - << 'PYCODE'
import torch
print(f"[DEBUG] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
PYCODE

# Paths (relative to repo root)
METADATA="data/metadata/clip_index.csv"
FEATURE_DIR="features"
OUTPUT_MODEL="models/r3d_18_final.pth"

# Verify inputs
[[ -f "$METADATA" ]]    || { echo "[ERROR] Missing CSV: $METADATA";    exit 1; }
[[ -d "$FEATURE_DIR" ]] || { echo "[ERROR] Missing features dir: $FEATURE_DIR"; exit 1; }

# Ensure output folder exists
mkdir -p "$(dirname "$OUTPUT_MODEL")"

# Launch training
export PYTHONPATH="/bigdata/rhome/tfrw2023/Code/PoloTagger:${PYTHONPATH:-}"
echo "[DEBUG] CWD: $(pwd)"

$PYTHON_PATH pipeline/train.py \
  --csv        "$METADATA" \
  --features   "$FEATURE_DIR" \
  --out        "$OUTPUT_MODEL" \
  --epochs     10 \
  --batch_size 4 \
  # --benchmark-data    # remove this if you don’t want the quick 10-batch timing

echo "[INFO] Training complete; model saved to $OUTPUT_MODEL"
