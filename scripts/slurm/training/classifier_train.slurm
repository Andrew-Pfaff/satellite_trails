#!/usr/bin/env bash
#SBATCH -J classifier_train
#SBATCH -p ampere
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=classifier_train-%j.out

set -euo pipefail

module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/train/train_classifier.py"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"

DATA_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SAMPLER_FRACTION="${SAMPLER_FRACTION:-0.1}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
DROPOUT_RATE="${DROPOUT_RATE:-0.2}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
ETA_MIN="${ETA_MIN:-1e-6}"
POS_WEIGHT="${POS_WEIGHT:-3.0}"
FN_PENALTY_WEIGHT="${FN_PENALTY_WEIGHT:-3.0}"
MIN_RECALL="${MIN_RECALL:-0.99}"
RECALL_PENALTY="${RECALL_PENALTY:-3.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-800}"
NORMALIZATION="${NORMALIZATION:-source_zscore}"
GRAD_CLIP_MAX_NORM="${GRAD_CLIP_MAX_NORM:-1.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
P_SHIFT="${P_SHIFT:-0.25}"
MIN_SHIFT="${MIN_SHIFT:-15}"
MAX_SHIFT="${MAX_SHIFT:-100}"
SEED="${SEED:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/results/models/classifier/classifier_train.pt}"
WEIGHT_SAVE_PATH="${WEIGHT_SAVE_PATH:-$PROJECT_ROOT/results/models/classifier/classifier_weights.pt}"
PLOT_PATH="${PLOT_PATH:-$PROJECT_ROOT/results/models/classifier/classifier_plots.png}"

mkdir -p "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$WEIGHT_SAVE_PATH")" "$(dirname "$PLOT_PATH")"

python3 "$SCRIPT_PATH" \
    --data-path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --sampler-fraction "$SAMPLER_FRACTION" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --base-channels "$BASE_CHANNELS" \
    --dropout-rate "$DROPOUT_RATE" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --eta-min "$ETA_MIN" \
    --pos-weight "$POS_WEIGHT" \
    --fn-penalty-weight "$FN_PENALTY_WEIGHT" \
    --min-recall "$MIN_RECALL" \
    --recall-penalty "$RECALL_PENALTY" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-workers "$NUM_WORKERS" \
    --normalization "$NORMALIZATION" \
    --grad-clip-max-norm "$GRAD_CLIP_MAX_NORM" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
    --p-shift "$P_SHIFT" \
    --min-shift "$MIN_SHIFT" \
    --max-shift "$MAX_SHIFT" \
    --full-save-path "$CHECKPOINT_PATH" \
    --weight-save-path "$WEIGHT_SAVE_PATH" \
    --plot-path "$PLOT_PATH" \
    --seed "$SEED" \
    --verbose
