#!/bin/bash
# Usage: ./run_preprocess.sh /path/to/raw_data

INPUT_DIR=$1
OUT_CLIPS="data/clips"
OUT_CSV="data/metadata/clip_index.csv"

mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")"

python preprocess.py \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len 5 \
  --fps 30
