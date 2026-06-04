#!/usr/bin/env bash
#SBATCH -J unet_train
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:20:00
#SBATCH --output=results/models/unet/unet_train-%j.out

set -euo pipefail

# Module handling
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/train/train_unet.py"

cd "$PROJECT_ROOT"

source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


DATA_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=0.003
DROPOUT_RATE=0.014
POS_WEIGHT=6.7
BCE_LOSS_FACTOR=1
DICE_LOSS_FACTOR=8
SAMPLER_FRACTION=0.26
WARMUP_EPOCHS=5
NUM_WORKERS=12
CHECKPOINT_PATH="$PROJECT_ROOT/results/models/unet/unet_train.pt"
WEIGHT_SAVE_PATH="$PROJECT_ROOT/results/models/unet/unet_weights.pt"
PLOT_PATH="$PROJECT_ROOT/results/models/unet/unet_plots"


#Running the script:
python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH" \
    --epochs="$EPOCHS" \
    --batch-size="$BATCH_SIZE" \
    --learning-rate="$LEARNING_RATE" \
    --dropout-rate="$DROPOUT_RATE" \
    --pos-weight="$POS_WEIGHT" \
    --bce-loss-factor="$BCE_LOSS_FACTOR" \
    --dice-loss-factor="$DICE_LOSS_FACTOR" \
    --num-workers="$NUM_WORKERS" \
    --warmup-epochs="$WARMUP_EPOCHS" \
    --sampler-fraction="$SAMPLER_FRACTION" \
    --full-save-path="$CHECKPOINT_PATH" \
    --weight-save-path="$WEIGHT_SAVE_PATH" \
    --plot-path="$PLOT_PATH" \
    --verbose