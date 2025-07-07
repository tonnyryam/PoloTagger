#!/bin/bash -l

#SBATCH --job-name=preprocess_fast
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL

# Print timestamp and node for debugging
date
hostname

# Load conda and activate environment
module load miniconda3
conda activate PoloTagger

# Fail on errors and undefined variables
set -euo pipefail

# Change to the directory where sbatch was invoked and set project root
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="$PROJECT_ROOT/data"

# Parse arguments
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_dir> [clip_len] [fps]"
  exit 1
fi
INPUT_DIR="$1"
CLIP_LEN="${2:-5}"
FPS="${3:-30}"

# Define output paths
OUT_CLIPS="$DATA_DIR/clips"
OUT_METADATA="$DATA_DIR/metadata"
OUT_CSV="$OUT_METADATA/clip_index.csv"
LOG_DIR="$PROJECT_ROOT/logs"

# Create necessary directories
mkdir -p "$OUT_CLIPS" "$OUT_METADATA" "$LOG_DIR"

# Prepare log file
timestamp=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/preprocess_${timestamp}.log"

# Log configuration
echo "[INFO] Starting preprocessing: $(date '+%F %T')" | tee -a "$LOG_FILE"
echo "[INFO] Input dir: $INPUT_DIR"       | tee -a "$LOG_FILE"
echo "[INFO] Clip length: $CLIP_LEN"      | tee -a "$LOG_FILE"
echo "[INFO] FPS: $FPS"                   | tee -a "$LOG_FILE"
echo "[INFO] Output clips: $OUT_CLIPS"     | tee -a "$LOG_FILE"
echo "[INFO] Output CSV: $OUT_CSV"         | tee -a "$LOG_FILE"
echo "[INFO] Log file: $LOG_FILE"          | tee -a "$LOG_FILE"

# Run preprocessing script via srun
srun python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len "$CLIP_LEN" \
  --fps "$FPS" 2>&1 | tee -a "$LOG_FILE"

# Verify output CSV
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV" | tee -a "$LOG_FILE"
else
  echo "[ERROR] ❌ Metadata CSV not found or empty" | tee -a "$LOG_FILE"
  exit 1
fi

# Completion timestamp
echo "[INFO] Completed preprocessing: $(date '+%F %T')" | tee -a "$LOG_FILE"
