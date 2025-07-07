#!/bin/bash -l

#SBATCH --job-name=preprocess_fast
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00         # adjust as needed
#SBATCH --mail-user=tfrw2023@mymail.pomona.edu
#SBATCH --mail-type=END,FAIL

# Print date and node for debugging
date
hostname

# Load conda and activate environment
module load miniconda3
conda activate PoloTagger

# Robust error handling
set -euo pipefail

# Ensure working directory is where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(pwd)"

# Usage and defaults
if [[ $# -lt 1 ]]; then
  echo "Usage: \$0 <input_dir> [clip_len] [fps]"
  exit 1
fi
INPUT_DIR="$1"
CLIP_LEN="\${2:-5}"
FPS="\${3:-30}"

# Output locations under project root
OUT_CLIPS="./bigdata/rhome/tfrw2023/Code/PoloTagger/data/clips"
OUT_CSV="./bigdata/rhome/tfrw2023/Code/PoloTagger/data/metadata/clip_index.csv"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$OUT_CLIPS" "$(dirname "$OUT_CSV")" "$LOG_DIR"

# Timestamp for log
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/preprocess_${TIMESTAMP}.log"

echo "[INFO] Starting preprocessing: \$(date '+%F %T')"
echo "[INFO] Input dir: \$INPUT_DIR"
echo "[INFO] Clip length: \$CLIP_LEN"
echo "[INFO] FPS: \$FPS"
echo "[INFO] Output clips: \$OUT_CLIPS"
echo "[INFO] Output CSV: \$OUT_CSV"
echo "[INFO] Log file: \$LOG_FILE"

# Launch preprocessing with srun
srun --nodes=1 --ntasks=1 \
  python "$PROJECT_ROOT/pipeline/preprocess.py" \
    --input_dir "$INPUT_DIR" \
    --out_dir "$OUT_CLIPS" \
    --metadata_csv "$OUT_CSV" \
    --clip_len "$CLIP_LEN" \
    --fps "$FPS" 2>&1 | tee -a "$LOG_FILE"

# Verify output CSV
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV"
else
  echo "[ERROR] ❌ Metadata CSV not found or empty"
  exit 1
fi

# Done
echo "[INFO] Completed preprocessing: \$(date '+%F %T')"
