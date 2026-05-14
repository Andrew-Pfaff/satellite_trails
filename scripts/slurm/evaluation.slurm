#!/usr/bin/env bash
#SBATCH -J evaluate
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/evaluate/evaluate-%j.out

set -euo pipefail

module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/evaluation.py"
DATA_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"

MODEL_PATH="$PROJECT_ROOT/results/mini_train/small_train.pt"
FULL_FIELD_INDEX=37
BATCH_SIZE=32
SUBSAMPLE_FRACTION=0.01
PATCH_DIM=528

PLOT_SAVE_PATH_FF="$PROJECT_ROOT/results/mini_train/evaluate/full_field.png"
PLOT_SAVE_PATH_ROC="$PROJECT_ROOT/results/mini_train/evaluate/roc_curve.png"
PLOT_SAVE_PATH_RESIDUAL="$PROJECT_ROOT/results/mini_train/evaluate/residual.png"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"

LOCAL_DATA_PATH="$TMPDIR/small_set.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
DATA_PATH="$LOCAL_DATA_PATH"

python3 "$SCRIPT_PATH" \
    --model-path="$MODEL_PATH" \
    --data-path="$DATA_PATH" \
    --full-field-index="$FULL_FIELD_INDEX" \
    --batch-size="$BATCH_SIZE" \
    --subsample-fraction="$SUBSAMPLE_FRACTION" \
    --patch-dim="$PATCH_DIM" \
    --plot-save-path-ff="$PLOT_SAVE_PATH_FF" \
    --plot-save-path-roc="$PLOT_SAVE_PATH_ROC" \
    --plot-save-path-residual="$PLOT_SAVE_PATH_RESIDUAL" \
    --verbose