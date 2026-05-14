#!/usr/bin/env bash
#SBATCH -J mini_test
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/mini_train/mini_train-%j.out



# Stop immediately on errors, unset variables, or failed pipelines so the job does not continue after a broken training run.
set -euo pipefail


# Module handling matches the rest of your Slurm scripts so the job gets a predictable environment on the cluster node.
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6


# Fixed project paths keep the script simple to submit. Edit these in one place when you want to benchmark a different dataset or output location.
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/model/train.py"


DATA_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"
EPOCHS=15
BATCH_SIZE=32
LEARNING_RATE=0.0015
LR_DECAY=50
DROPOUT_RATE=0.08
P_AUG=0.4
NUM_WORKERS=4
CHECKPOINT_PATH="$PROJECT_ROOT/results/mini_train/small_train.pt"
PLOT_PATH="$PROJECT_ROOT/results/mini_train/plots"


cd "$PROJECT_ROOT"
source "$VENV_PATH"

#fail fast if torch cannot load to CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


LOCAL_DATA_PATH="$TMPDIR/small_set.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
# Use local path from now on
DATA_PATH="$LOCAL_DATA_PATH"


python3 "$SCRIPT_PATH" \
    --data-path="$DATA_PATH" \
    --epochs="$EPOCHS" \
    --batch-size="$BATCH_SIZE" \
    --learning-rate="$LEARNING_RATE" \
    --lr-decay="$LR_DECAY" \
    --dropout-rate="$DROPOUT_RATE" \
    --augmentation-prob="$P_AUG" \
    --save-path="$CHECKPOINT_PATH" \
    --plot-path="$PLOT_PATH" \
    --num-workers="$NUM_WORKERS"
