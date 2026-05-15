#!/usr/bin/env bash
#SBATCH -J optuna_study
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/optuna_study-%j.out

set -euo pipefail

#Module handling
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/optuna/hyperparam_tuning.py"


cd "$PROJECT_ROOT"

source "$VENV_PATH"

# Fail fast if torch cannot load to CUDA.
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


#Arguments:
DATA_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"
	
mkdir -p "$TMPDIR"
LOCAL_DATA_PATH="$TMPDIR/small_dataset.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
# Use local path from now on.
DATA_PATH="$LOCAL_DATA_PATH"


#Running the script:
python3 "$SCRIPT_PATH" --data-path="$DATA_PATH"
