#!/usr/bin/env bash
#SBATCH -J classifier_train
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/models/classifier/classifier_train-%j.out

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


DATA_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
EPOCHS=40
BATCH_SIZE=128
LEARNING_RATE=0.0016
SAMPLER_FRACTION=0.25
POS_WEIGHT=4.0
FN_PENALTY_WEIGHT=2.1
PRED_THRESHOLD=0.67
NUM_WORKERS=12
CHECKPOINT_PATH="$PROJECT_ROOT/results/models/classifier/classifier_train.pt"
WEIGHT_SAVE_PATH="$PROJECT_ROOT/results/models/classifier/classifier_weights.pt"
PLOT_PATH="$PROJECT_ROOT/results/models/classifier/classifier_plots"


#Running the script:
python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH" \
    --epochs="$EPOCHS" \
    --batch-size="$BATCH_SIZE" \
    --learning-rate="$LEARNING_RATE" \
    --sampler-fraction="$SAMPLER_FRACTION" \
    --pos-weight="$POS_WEIGHT" \
    --fn-penalty-weight="$FN_PENALTY_WEIGHT" \
    --pred-threshold="$PRED_THRESHOLD" \
    --num-workers="$NUM_WORKERS" \
    --full-save-path="$CHECKPOINT_PATH" \
    --weight-save-path="$WEIGHT_SAVE_PATH" \
    --plot-path="$PLOT_PATH" \
    --verbose