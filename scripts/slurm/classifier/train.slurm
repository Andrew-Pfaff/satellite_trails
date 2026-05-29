#!/usr/bin/env bash
#SBATCH -J classifier_train
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/train/classifier_train-%j.out

set -euo pipefail

# Module handling
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/train/train_classifier.py"

cd "$PROJECT_ROOT"

source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


DATA_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"
EPOCHS=25
BATCH_SIZE=
LEARNING_RATE=
SAMPLER_FRACTION=
WARMUP_EPOCHS=
ETA_MIN=
P_SHIFT=
POS_WEIGHT=
FN_PENALTY_WEIGHT=
PRED_THRESHOLD=
MIN_RECALL=0.95
RECALL_PENALTY=1.0
NUM_WORKERS=0
CHECKPOINT_PATH="$PROJECT_ROOT/results/train/classifier_train.pt"
WEIGHT_SAVE_PATH="$PROJECT_ROOT/results/train/classifier_weights.pt"
PLOT_PATH="$PROJECT_ROOT/results/train/classifier_plots"


mkdir -p "$TMPDIR"
LOCAL_DATA_PATH="$TMPDIR/small_dataset.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
# Use local path from now on.
DATA_PATH="$LOCAL_DATA_PATH"


#Running the script:
python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH" \
    --epochs="$EPOCHS" \
    --batch-size="$BATCH_SIZE" \
    --learning-rate="$LEARNING_RATE" \
    --sampler-fraction="$SAMPLER_FRACTION" \
    --warmup-epochs="$WARMUP_EPOCHS" \
    --eta-min="$ETA_MIN" \
    --p-shift="$P_SHIFT" \
    --pos-weight="$POS_WEIGHT" \
    --fn-penalty-weight="$FN_PENALTY_WEIGHT" \
    --pred-threshold="$PRED_THRESHOLD" \
    --min-recall="$MIN_RECALL" \
    --recall-penalty="$RECALL_PENALTY" \
    --num-workers="$NUM_WORKERS" \
    --full-save-path="$CHECKPOINT_PATH" \
    --weight-save-path="$WEIGHT_SAVE_PATH" \
    --plot-path="$PLOT_PATH" \
    --verbose