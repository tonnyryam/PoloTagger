# scripts/run_train.sh
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

# 2. Load & initialize Conda
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate PoloTagger

# 3. Bail on errors or unset vars
set -euo pipefail

# 4. Accept optional args for pretrained models (default to yolov5s & mnist-onnx)
YOLO_MODEL="${1:-yolov5s.pt}"
DIGIT_MODEL="${2:-mnist-onnx}"

echo "[INFO] Training only – pulling weights as needed"
echo "[INFO]   YOLO model specifier:  $YOLO_MODEL"
echo "[INFO]   Digit model specifier: $DIGIT_MODEL"

# 5. Paths
METADATA="data/metadata/clip_index.csv"
FEATURE_DIR="features"
OUTPUT_MODEL="models/r3d_18_final.pth"

# 6. Verify inputs exist
if [[ ! -f "$METADATA" ]]; then
  echo "[ERROR] Metadata CSV not found: $METADATA"
  exit 1
fi
if [[ ! -d "$FEATURE_DIR" ]]; then
  echo "[ERROR] Features directory not found: $FEATURE_DIR"
  exit 1
fi

# 7. Create output dir
mkdir -p "$(dirname "$OUTPUT_MODEL")"

# 8. Run training
python train.py \
  --csv "$METADATA" \
  --features "$FEATURE_DIR" \
  --out "$OUTPUT_MODEL" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL"

echo "[INFO] Training complete, model saved to $OUTPUT_MODEL"