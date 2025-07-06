#!/bin/bash
#SBATCH --job-name=preprocess_fast
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@mymail.pomona.edu
#SBATCH --export=ALL

# Load environment modules (adjust as needed)
module load miniconda3
conda activate PoloTagger

# Check inputs and set defaults
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_dir> [clip_len] [fps]"
  exit 1
fi
INPUT_DIR="$1"
CLIP_LEN="${2:-5}"
FPS="${3:-30}"

# Define project directories
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
OUT_CLIPS="$PROJECT_ROOT/data/clips"
OUT_CSV="$PROJECT_ROOT/data/metadata/clip_index.csv"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")" "$LOG_DIR"

# Timestamped logging
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/preprocess_${TIMESTAMP}.log"

echo "[INFO] Starting preprocessing: $(date '+%F %T')"
echo "[INFO] Input dir: $INPUT_DIR"
echo "[INFO] Clip length: $CLIP_LEN"
echo "[INFO] FPS: $FPS"
echo "[INFO] Output clips: $OUT_CLIPS"
echo "[INFO] Output CSV: $OUT_CSV"
echo "[INFO] Log file: $LOG_FILE"

# Run with srun to utilize allocated CPUs
srun python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len "$CLIP_LEN" \
  --fps "$FPS" 2>&1 | tee -a "$LOG_FILE"

# Verify CSV output
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV"
else
  echo "[ERROR] ❌ Metadata CSV not found or empty"
  exit 1
fi

echo "[INFO] Completed preprocessing: $(date '+%F %T')"
