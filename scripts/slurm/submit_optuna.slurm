#!/usr/bin/env bash
#SBATCH -J trail-pre
#SBATCH -p icelake
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out

#Module handling
module purge
module load rhel8/default-icl


#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/optuna/hyperparam_tuning.py"


cd "$PROJECT_ROOT"

source "$VENV_PATH"


#Arguments:
DATA_PATH="$PROJECT_ROOT/data/dataset_pngs"
OUTPUT_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
VAL_SPLIT=0.15
TEST_SPLIT=0.15
	

#Running the script:
python3 "$SCRIPT_PATH" --data-path="$DATA_PATH" --output-path="$OUTPUT_PATH" --val-split="$VAL_SPLIT" --test-split="$TEST_SPLIT"

