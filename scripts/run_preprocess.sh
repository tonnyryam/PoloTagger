#!/bin/bash

# Resolve project root (the directory containing this script)
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")

INPUT_DIR="$1"
OUT_CLIPS="$PROJECT_ROOT/data/clips"
OUT_CSV="$PROJECT_ROOT/data/metadata/clip_index.csv"

mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")"

python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len 5 \
  --fps 30
