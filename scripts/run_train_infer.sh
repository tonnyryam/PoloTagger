#!/bin/bash

YOLO_MODEL=$1
DIGIT_MODEL=$2

MODEL_OUT="models/r3d_18_final.pth"
METADATA="data/metadata/clip_index.csv"
RAW_XML="results/predictions.xml"
FINAL_XML="results/predictions_sportscode.xml"

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
  --export_xml "$RAW_XML"

echo "=== POSTPROCESSING TO SPORTSCODE XML ==="
python postprocess.py \
  --xml "$RAW_XML" \
  --out "$FINAL_XML" \
  --fps 30 \
  --min_duration 30 \
  --min_conf 0.6 \
  --merge_gap 45

echo "✅ Done. Final output: $FINAL_XML"
