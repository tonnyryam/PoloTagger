#!/bin/bash -l

#SBATCH --job-name=preprocess_fast
#SBATCH --output=scripts/logs/%x_%j.log
#SBATCH --error=scripts/logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --chdir=/bigdata/rhome/tfrw2023/Code/PoloTagger

# Print timestamp and node for debugging
date
hostname

# Load conda and activate environment
module load miniconda3
conda activate PoloTagger

# Fail on errors and undefined variables
set -euo pipefail

# Determine script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$( realpath "$SCRIPT_DIR/.." )"
fi
cd "$PROJECT_ROOT"

# Locate your data directory
DATA_DIR="$PROJECT_ROOT/data"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] Data directory not found: $DATA_DIR"
  exit 1
fi

# Parse arguments
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

# Define output paths
OUT_CLIPS="$DATA_DIR/clips"
OUT_METADATA="$DATA_DIR/metadata"
OUT_CSV="$OUT_METADATA/clip_index.csv"

# Put logs inside the scripts/ folder
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_CLIPS" "$OUT_METADATA" "$LOG_DIR"

# Prepare log file with timestamp
timestamp=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/preprocess_${timestamp}.log"

# Log configuration
{
  echo "[INFO] Starting preprocessing: $(date '+%F %T')"
  echo "[INFO] Input dir:        $INPUT_DIR"
  echo "[INFO] Clip length:      $CLIP_LEN"
  echo "[INFO] FPS:              $FPS"
  echo "[INFO] Output clips:     $OUT_CLIPS"
  echo "[INFO] Output CSV:       $OUT_CSV"
  echo "[INFO] Log file:         $LOG_FILE"
} | tee -a "$LOG_FILE"

# Execute preprocessing via srun
srun python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir    "$INPUT_DIR" \
  --out_dir      "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len     "$CLIP_LEN" \
  --fps          "$FPS" 2>&1 | tee -a "$LOG_FILE"

# Verify output CSV
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV" | tee -a "$LOG_FILE"
else
  echo "[ERROR] ❌ Metadata CSV not found or empty" | tee -a "$LOG_FILE"
  exit 1
fi

# Completion timestamp
echo "[INFO] Completed preprocessing: $(date '+%F %T')" | tee -a "$LOG_FILE"
