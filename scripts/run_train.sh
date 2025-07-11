#!/bin/bash -l

#SBATCH --job-name=train_only
#SBATCH --output=scripts/%x_%j.log
#SBATCH --error=scripts/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --chdir=/bigdata/rhome/tfrw2023/Code/PoloTagger

# Debug info
date
hostname

# Load & initialize Conda
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate PoloTagger

echo "[DEBUG] Python path: $(which python)"
echo "[DEBUG] Python version: $(python --version)"
echo "[DEBUG] Conda environment: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "[ERROR] PyTorch not available"

# Bail on errors or unset vars
set -euo pipefail

# Accept optional args for pretrained models (default to yolov5s & mnist-onnx)
YOLO_MODEL="${1:-yolov5s.pt}"
DIGIT_MODEL="${2:-mnist-onnx}"

echo "[INFO] Training only – pulling weights as needed"
echo "[INFO]   YOLO model specifier:  $YOLO_MODEL"
echo "[INFO]   Digit model specifier: $DIGIT_MODEL"

# Paths
METADATA="data/metadata/clip_index.csv"
FEATURE_DIR="features"
OUTPUT_MODEL="models/r3d_18_final.pth"

# Verify inputs exist
if [[ ! -f "$METADATA" ]]; then
  echo "[ERROR] Metadata CSV not found: $METADATA"
  exit 1
fi
if [[ ! -d "$FEATURE_DIR" ]]; then
  echo "[ERROR] Features directory not found: $FEATURE_DIR"
  exit 1
fi

# Create output dir
mkdir -p "$(dirname "$OUTPUT_MODEL")"

# Run training
python pipeline/train.py \
  --csv "$METADATA" \
  --features "$FEATURE_DIR" \
  --out "$OUTPUT_MODEL" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL"

echo "[INFO] Training complete, model saved to $OUTPUT_MODEL"