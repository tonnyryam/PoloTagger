#!/bin/bash -l

#SBATCH --job-name="train_only"\
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=16
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --err=train_%j.log
#SBATCH --out=train_%j.log

# End the script if any command fails
set -euo pipefail

# Debug info
date
hostname

PYTHON_PATH="/bigdata/rhome/tfrw2023/.conda/envs/PoloTagger/bin/python3.10"
if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "[ERROR] Python not found at $PYTHON_PATH"
    exit 1
fi

echo "[DEBUG] Using Python: $PYTHON_PATH"
echo "[DEBUG] Python version: $($PYTHON_PATH --version)"

# Quick PyTorch sanity check
$PYTHON_PATH - << 'PYCODE'
import torch
print(f"[DEBUG] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
PYCODE

# Fail fast on errors
set -euo pipefail

# Models (positional args, defaults shown)
YOLO_MODEL="${1:-yolov5s.pt}"
DIGIT_MODEL="${2:-mnist-onnx}"

echo "[INFO] YOLO model:  $YOLO_MODEL"
echo "[INFO] Digit model: $DIGIT_MODEL"

# Paths
METADATA="data/metadata/clip_index.csv"
FEATURE_DIR="features"
OUTPUT_MODEL="models/r3d_18_final.pth"

# 7. Verify inputs
[[ -f "$METADATA" ]]    || { echo "[ERROR] Missing CSV: $METADATA";    exit 1; }
[[ -d "$FEATURE_DIR" ]] || { echo "[ERROR] Missing features dir: $FEATURE_DIR"; exit 1; }

# Ensure output folder exists
mkdir -p "$(dirname "$OUTPUT_MODEL")"

# Run training (add --benchmark-data if you want the 10-batch timing)
export PYTHONPATH="/bigdata/rhome/tfrw2023/Code/PoloTagger:${PYTHONPATH:-}"
echo "[DEBUG] CWD: $(pwd)"
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"

$PYTHON_PATH pipeline/train.py \
  --csv    "$METADATA" \
  --features "$FEATURE_DIR" \
  --out    "$OUTPUT_MODEL" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL" \
  "${@:3}"

echo "[INFO] Training complete; model saved to $OUTPUT_MODEL"