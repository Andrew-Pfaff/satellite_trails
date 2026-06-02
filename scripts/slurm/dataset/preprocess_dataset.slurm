#!/usr/bin/env bash
#SBATCH -J trail-pre
#SBATCH -p icelake
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

#Module handling
module purge
module load rhel8/default-icl


#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/data/preprocessing.py"


cd "$PROJECT_ROOT"

source "$VENV_PATH"


#Arguments:
DATA_PATH="$PROJECT_ROOT/data/png"
OUTPUT_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
VAL_SPLIT=0.15
TEST_SPLIT=0.15

SMALL_OUTPUT_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"
SMALL_VAL_SPLIT=0.25
SMALL_TEST_SPLIT=0.0
NUM_IMAGES=40
	

#Full dataset
python3 "$SCRIPT_PATH" --data-path="$DATA_PATH" --output-path="$OUTPUT_PATH" --val-split="$VAL_SPLIT" --test-split="$TEST_SPLIT" --verbose

#Small dataset
python3 "$SCRIPT_PATH" --data-path="$DATA_PATH" --output-path="$SMALL_OUTPUT_PATH" --val-split="$SMALL_VAL_SPLIT" --test-split="$SMALL_TEST_SPLIT" --num-images="$NUM_IMAGES" --verbose





#Dataset summary
CSV_PATH="$PROJECT_ROOT/data/h5s/dataset_summary.csv"
SMALL_CSV_PATH="$PROJECT_ROOT/data/h5s/small_dataset_summary.csv"

SUMMARY_SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/utils/h5summary.py"

#Full dataset
python3 "$SUMMARY_SCRIPT_PATH" --data-path="$OUTPUT_PATH" --csv-path="$CSV_PATH"

#Small dataset
python3 "$SUMMARY_SCRIPT_PATH" --data-path="$SMALL_OUTPUT_PATH" --csv-path="$SMALL_CSV_PATH"