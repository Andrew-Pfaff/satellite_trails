#!/usr/bin/env bash
#SBATCH -J unet_tuning
#SBATCH -p ampere
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/hyperparameter_tuning/logs/unet/unet_tuning-%j.out

set -euo pipefail

module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/parameter_search/unet_optuna_tuning.py"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"

DATA_PATH="$PROJECT_ROOT/data/h5s/half_dataset.h5"
EPOCHS="${EPOCHS:-20}"
TRIALS="${TRIALS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-4}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
SEED="${SEED:-1}"
STUDY_NAME="${STUDY_NAME:-unet_tuning}"
DB_DIR="${DB_DIR:-$PROJECT_ROOT/results/hyperparameter_tuning/dbs}"

mkdir -p "$PROJECT_ROOT/results/hyperparameter_tuning/logs/unet" "$DB_DIR" "$(dirname "$DB_DIR")/summaries"

python3 "$SCRIPT_PATH" \
    --data-path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --trials "$TRIALS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --seed "$SEED" \
    --study-name "$STUDY_NAME" \
    --db-dir "$DB_DIR" \
    --verbose
