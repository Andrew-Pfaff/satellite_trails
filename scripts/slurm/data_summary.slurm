#!/usr/bin/env bash
#SBATCH -J h5_summary
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=results/summary/summary-%j.out

set -euo pipefail

module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"

SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/utils/h5summary.py"

DATA_PATH_1="$PROJECT_ROOT/data/h5s/small_dataset.h5"
DATA_PATH_2="$PROJECT_ROOT/data/h5s/dataset.h5"

CSV_PATH_1="$PROJECT_ROOT/data/h5s/small_dataset_summary.csv"
CSV_PATH_2="$PROJECT_ROOT/data/h5s/dataset_summary.csv"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

echo "=== Summary: small_dataset ==="
python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH_1" \
    --csv-path="$CSV_PATH_1"

echo "=== Summary: large_dataset ==="
python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH_2" \
    --csv-path="$CSV_PATH_2"

echo "Done. CSVs written to $(dirname "$CSV_PATH_1")"