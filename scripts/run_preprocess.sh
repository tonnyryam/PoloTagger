#!/bin/bash -l

#SBATCH --job-name=preprocess_fast
#SBATCH --output=scripts/%x_%j.log
#SBATCH --error=scripts/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --chdir=/bigdata/rhome/tfrw2023/Code/PoloTagger

# Debug info
date
hostname

# Load environment
module load miniconda3
conda activate PoloTagger

# Bail on any error or undefined variable
set -euo pipefail

# 1. Define PROJECT_ROOT as the current working directory (set by --chdir)
PROJECT_ROOT="$(pwd)"

# 2. Verify data directory exists at $PROJECT_ROOT/data
DATA_DIR="$PROJECT_ROOT/data"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] Data directory not found: $DATA_DIR"
  exit 1
fi

# 3. Parse arguments
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <relative/path/to/raw_data> [clip_len] [fps]"
  exit 1
fi
REL_INPUT_DIR="$1"
INPUT_DIR="$PROJECT_ROOT/$REL_INPUT_DIR"
CLIP_LEN="${2:-5}"
FPS="${3:-30}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Input directory not found: $INPUT_DIR"
  exit 1
fi

# 4. Set up outputs
OUT_CLIPS="$DATA_DIR/clips"
OUT_METADATA="$DATA_DIR/metadata"
OUT_CSV="$OUT_METADATA/clip_index.csv"

# Ensure clips & metadata dirs exist
mkdir -p "$OUT_CLIPS" "$OUT_METADATA"

# 5. Logs go into scripts/
LOG_DIR="$PROJECT_ROOT/scripts"
timestamp=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/preprocess_${timestamp}.log"

# Write header to log
{
  echo "[INFO] Starting preprocessing: $(date '+%F %T')"
  echo "[INFO] PROJECT_ROOT:       $PROJECT_ROOT"
  echo "[INFO] Input directory:    $INPUT_DIR"
  echo "[INFO] Clip length (s):    $CLIP_LEN"
  echo "[INFO] FPS:                $FPS"
  echo "[INFO] Output clips dir:   $OUT_CLIPS"
  echo "[INFO] Metadata CSV path:  $OUT_CSV"
  echo "[INFO] Combined log file:  $LOG_FILE"
} | tee -a "$LOG_FILE"

# 6. Run the Python pipeline
srun python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir    "$INPUT_DIR" \
  --out_dir      "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len     "$CLIP_LEN" \
  --fps          "$FPS" 2>&1 | tee -a "$LOG_FILE"

# 7. Verify that metadata CSV was created
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV" | tee -a "$LOG_FILE"
else
  echo "[ERROR] ❌ Metadata CSV not found or is empty" | tee -a "$LOG_FILE"
  exit 1
fi

# 8. Final timestamp
echo "[INFO] Completed preprocessing: $(date '+%F %T')" | tee -a "$LOG_FILE"
