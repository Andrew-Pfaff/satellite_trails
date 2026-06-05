#!/usr/bin/env bash
#SBATCH -J optuna_study
#SBATCH -p icelake
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=results/optuna/analyze_optuna-%j.out

set -euo pipefail

#Module handling
module purge
module load rhel8/default-icl


#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/optuna/results_analysis.py"


cd "$PROJECT_ROOT"

source "$VENV_PATH"

#Running the script:
python3 "$SCRIPT_PATH" \
    --model-type="unet" \
    --db-path="sqlite:///$PROJECT_ROOT/results/optuna/unet_full_tuning.db" \
    --csv-path="$PROJECT_ROOT/results/optuna/analysis/unet_summary.csv" \
    --plot-path="$PROJECT_ROOT/results/optuna/analysis/unet_plots"

python3 "$SCRIPT_PATH" \
    --model-type="classifier" \
    --db-path="sqlite:///$PROJECT_ROOT/results/optuna/classifier_full_tuning.db" \
    --csv-path="$PROJECT_ROOT/results/optuna/analysis/classifier_summary.csv" \
    --plot-path="$PROJECT_ROOT/results/optuna/analysis/classifier_plots"