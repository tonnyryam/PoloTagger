#!/bin/bash
#SBATCH --job-name=preprocess_fast
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --chdir=/bigdata/rhome/tfrw2023/Code/PoloTagger/scripts

# Backup: force directory change in case --chdir fails
cd /bigdata/rhome/tfrw2023/Code/PoloTagger/scripts

# Load environment
source ~/.bashrc
conda activate PoloTagger

# Define paths
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
INPUT_DIR="$1"
OUT_CLIPS="$PROJECT_ROOT/data/clips"
OUT_CSV="$PROJECT_ROOT/data/metadata/clip_index.csv"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/preprocess_$TIMESTAMP.log"

mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")" "$LOG_DIR"

# Logging
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="
echo "[INFO] Starting preprocessing at $TIMESTAMP"
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output clips: $OUT_CLIPS"
echo "[INFO] Output CSV: $OUT_CSV"
echo "[INFO] Logging to $LOG_FILE"

# Run Python preprocessing
python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len 5 \
  --fps 30 | tee -a "$LOG_FILE"

# Confirm output
if [ -f "$OUT_CSV" ]; then
  echo "[INFO] ✅ Metadata CSV successfully created: $OUT_CSV"
else
  echo "[ERROR] ❌ Metadata CSV was not created. Check logs for issues."
fi

echo "[INFO] 📄 Log saved to $LOG_FILE"
