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

# 1. Debug info
date
hostname

# 2. Set direct path to Python in the PoloTagger environment
PYTHON_PATH="/bigdata/rhome/tfrw2023/.conda/envs/PoloTagger/bin/python"

# 3. Verify the Python path exists and works
if [[ ! -f "$PYTHON_PATH" ]]; then
    echo "[ERROR] Python not found at: $PYTHON_PATH"
    echo "[INFO] Checking conda environments..."
    ls -la /bigdata/rhome/tfrw2023/.conda/envs/
    exit 1
fi

echo "[DEBUG] Using Python: $PYTHON_PATH"
echo "[DEBUG] Python version: $($PYTHON_PATH --version)"

# Test PyTorch import
$PYTHON_PATH -c "import torch; print(f'[DEBUG] PyTorch version: {torch.__version__}'); print(f'[DEBUG] CUDA available: {torch.cuda.is_available()}')" || {
    echo "[ERROR] PyTorch import failed"
    echo "[INFO] Checking installed packages..."
    $PYTHON_PATH -c "import sys; print('Python executable:', sys.executable)"
    $PYTHON_PATH -m pip list | grep torch || echo "[INFO] No torch packages found"
    exit 1
}

# 4. Bail on errors or unset vars (after setup)
set -euo pipefail

# 5. Accept optional args for pretrained models (default to yolov5s & mnist-onnx)
YOLO_MODEL="${1:-yolov5s.pt}"
DIGIT_MODEL="${2:-mnist-onnx}"

echo "[INFO] Training only – pulling weights as needed"
echo "[INFO]   YOLO model specifier:  $YOLO_MODEL"
echo "[INFO]   Digit model specifier: $DIGIT_MODEL"

# 6. Paths
METADATA="data/metadata/clip_index.csv"
FEATURE_DIR="features"
OUTPUT_MODEL="models/r3d_18_final.pth"

# 7. Verify inputs exist
if [[ ! -f "$METADATA" ]]; then
  echo "[ERROR] Metadata CSV not found: $METADATA"
  exit 1
fi
if [[ ! -d "$FEATURE_DIR" ]]; then
  echo "[ERROR] Features directory not found: $FEATURE_DIR"
  exit 1
fi

# 8. Create output dir
mkdir -p "$(dirname "$OUTPUT_MODEL")"

# 9. Run training with direct Python path
$PYTHON_PATH pipeline/train.py \
  --csv "$METADATA" \
  --features "$FEATURE_DIR" \
  --out "$OUTPUT_MODEL" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL"

echo "[INFO] Training complete, model saved to $OUTPUT_MODEL"