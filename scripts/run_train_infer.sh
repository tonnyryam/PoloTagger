#!/bin/bash -l

#SBATCH --job-name=train_infer
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

# 4. Parse arguments
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <yolo_model.pt> <digit_model.pth>"
  exit 1
fi
YOLO_MODEL=$1
DIGIT_MODEL=$2

# 5. Define paths
MODEL_OUT="models/r3d_18_final.pth"
METADATA="data/metadata/clip_index.csv"
EXPORT_XML="results/predictions.xml"

# 6. Create output directories
mkdir -p models results

# 7. Train the model
echo "[INFO] === TRAINING MODEL ==="
python train.py \
  --csv "$METADATA" \
  --features features \
  --out "$MODEL_OUT" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL"

# 8. Run inference
echo "[INFO] === RUNNING INFERENCE ==="
python infer.py \
  --csv "$METADATA" \
  --model "$MODEL_OUT" \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL" \
  --show_caps \
  --export_xml "$EXPORT_XML"

# 9. Completion
echo "[INFO] Completed training and inference"