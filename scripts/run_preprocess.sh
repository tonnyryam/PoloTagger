#!/bin/bash
# Usage: ./run_preprocess_all.sh /path/to/videos /path/to/xmls

VIDEO_DIR=$1
XML_DIR=$2

OUT_CLIPS="data/clips"
OUT_CSV="data/metadata/clip_index.csv"
mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")"

# Create/reset metadata CSV
echo "clip_path,label,start_frame,end_frame,source_video" > "$OUT_CSV"

# Match mp4 and xml files by base name (e.g. Game1.mp4 ↔ Game1.xml)
for video in "$VIDEO_DIR"/*.mp4; do
    base=$(basename "$video" .mp4)
    xml="$XML_DIR/$base.xml"
    if [ -f "$xml" ]; then
        echo "📦 Processing: $base"
        python preprocess.py \
          --video "$video" \
          --xml "$xml" \
          --out_dir "$OUT_CLIPS" \
          --metadata_csv temp_meta.csv \
          --clip_len 5 \
          --fps 30

        # Append to main CSV
        tail -n +2 temp_meta.csv >> "$OUT_CSV"
        rm temp_meta.csv
    else
        echo "⚠️ No XML found for $video"
    fi
done

echo "✅ All preprocessing complete. Combined metadata: $OUT_CSV"