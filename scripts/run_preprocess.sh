#!/bin/bash
# Usage: ./run_preprocess.sh /path/to/videos /path/to/xmls

VIDEO_DIR=$1
XML_DIR=$2

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/preprocess_$TIMESTAMP.log"
mkdir -p "$LOG_DIR"

# Log both stdout and stderr
exec > >(tee -a "$LOG_FILE") 2>&1

OUT_CLIPS="data/clips"
OUT_CSV="data/metadata/clip_index.csv"
mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")"

echo "[INFO] Creating header for metadata CSV: $OUT_CSV"
echo "clip_path,label,start_frame,end_frame,source_video" > "$OUT_CSV"

for video in "$VIDEO_DIR"/*.mp4; do
    base=$(basename "$video" .mp4)
    xml="$XML_DIR/$base.xml"
    if [ -f "$xml" ]; then
        echo "[INFO] 📦 Processing: $base"
        python preprocess.py \\
          --video "$video" \\
          --xml "$xml" \\
          --out_dir "$OUT_CLIPS" \\
          --metadata_csv temp_meta.csv \\
          --clip_len 5 \\
          --fps 30

        if [ -f temp_meta.csv ]; then
            echo "[INFO] ✅ Appending metadata from temp_meta.csv"
            tail -n +2 temp_meta.csv >> "$OUT_CSV"
            rm temp_meta.csv
        else
            echo "[WARN] ❌ temp_meta.csv not found after processing $base"
        fi
    else
        echo "[WARN] ⚠️ No XML found for $video"
    fi
done

echo "[INFO] ✅ All preprocessing complete. Combined metadata: $OUT_CSV"
echo "[INFO] 📄 Full log saved to $LOG_FILE"