#!/bin/bash
# Usage: ./run_train_infer.sh best_yolo_cap_model.pt digit_classifier.pth

YOLO_MODEL=$1
DIGIT_MODEL=$2

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/train_infer_$TIMESTAMP.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

MODEL_OUT="models/r3d_18_final.pth"
METADATA="data/metadata/clip_index.csv"
EXPORT_XML="results/predictions.xml"

mkdir -p models results

echo "[INFO] === TRAINING MODEL ==="
python train.py \\
  --csv "$METADATA" \\
  --features features \\
  --out "$MODEL_OUT" \\
  --epochs 10 \\
  --batch_size 8 \\
  --yolo_model "$YOLO_MODEL" \\
  --digit_model "$DIGIT_MODEL"

echo "[INFO] === RUNNING INFERENCE ==="
python infer.py \\
  --csv "$METADATA" \\
  --model "$MODEL_OUT" \\
  --batch_size 8 \\
  --yolo_model "$YOLO_MODEL" \\
  --digit_model "$DIGIT_MODEL" \\
  --show_caps \\
  --export_xml "$EXPORT_XML"

echo "[INFO] 📄 Full log saved to $LOG_FILE"