#!/bin/bash
# Usage: ./run_train_infer.sh best_yolo_cap_model.pt digit_classifier.pth

YOLO_MODEL=$1
DIGIT_MODEL=$2

MODEL_OUT="models/r3d_18_final.pth"
METADATA="data/metadata/clip_index.csv"
EXPORT_XML="results/predictions.xml"

mkdir -p models results

echo "=== TRAINING MODEL ==="
python train.py \
  --csv "$METADATA" \
  --features features \
  --out "$MODEL_OUT" \
  --epochs 10 \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL"

echo "=== RUNNING INFERENCE ==="
python infer.py \
  --csv "$METADATA" \
  --model "$MODEL_OUT" \
  --batch_size 8 \
  --yolo_model "$YOLO_MODEL" \
  --digit_model "$DIGIT_MODEL" \
  --show_caps \
  --export_xml "$EXPORT_XML"
