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

# 1. Debug info
date
hostname

# 2. Load & initialize Conda
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate PoloTagger

# 3. Bail on errors or unset vars
set -euo pipefail

# 4. PROJECT_ROOT is forced by --chdir
PROJECT_ROOT="$(pwd)"

# 5. Verify data directory
DATA_DIR="$PROJECT_ROOT/data"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] Data directory not found: $DATA_DIR"
  exit 1
fi

# 6. Parse arguments
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

# 7. Prepare output dirs
OUT_CLIPS="$DATA_DIR/clips"
OUT_METADATA="$DATA_DIR/metadata"
OUT_CSV="$OUT_METADATA/clip_index.csv"
mkdir -p "$OUT_CLIPS" "$OUT_METADATA"

# 8. Header info (goes into the single SLURM log)
echo "[INFO] Starting preprocessing: $(date '+%F %T')"
echo "[INFO] PROJECT_ROOT:       $PROJECT_ROOT"
echo "[INFO] Input directory:    $INPUT_DIR"
echo "[INFO] Clip length (s):    $CLIP_LEN"
echo "[INFO] FPS:                $FPS"
echo "[INFO] Output clips dir:   $OUT_CLIPS"
echo "[INFO] Metadata CSV path:  $OUT_CSV"

# 9. Verify Python environment
echo "[INFO] Python path: $(which python)"
echo "[INFO] Testing pandas import..."
python - << 'EOF'
import pandas
print("Pandas version:", pandas.__version__)
EOF

# 10. Run the preprocessing pipeline
python "$PROJECT_ROOT/pipeline/preprocess.py" \
  --input_dir    "$INPUT_DIR" \
  --out_dir      "$OUT_CLIPS" \
  --metadata_csv "$OUT_CSV" \
  --clip_len     "$CLIP_LEN" \
  --fps          "$FPS"

# 11. Verify success
if [[ -s "$OUT_CSV" ]]; then
  echo "[INFO] ✅ Metadata CSV created: $OUT_CSV"
else
  echo "[ERROR] ❌ Metadata CSV not found or is empty"
  exit 1
fi

# 12. Completion timestamp
echo "[INFO] Completed preprocessing: $(date '+%F %T')"
