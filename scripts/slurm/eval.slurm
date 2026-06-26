#!/usr/bin/env bash
#SBATCH -J model_eval
#SBATCH -p ampere
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=model_eval-%j.out

set -euo pipefail

module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/scripts/python/evaluate_models.py"

cd "$PROJECT_ROOT"
source "$VENV_PATH"

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"

MODEL_TYPE="${MODEL_TYPE:-classifier}"
H5_PATH="$PROJECT_ROOT/data/h5s/dataset.h5"
SPLIT="${SPLIT:-val}"
NORMALIZATION="${NORMALIZATION:-source_zscore}"
BATCH_SIZE="${BATCH_SIZE:-}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.05}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.95}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-19}"
MIN_RECALL="${MIN_RECALL:-0.99}"
RECALL_PENALTY="${RECALL_PENALTY:-3.0}"

if [[ "$MODEL_TYPE" == "unet" ]]; then
    MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/results/models/unet/unet_weights.pt}"
    THRESHOLD_PLOT="${THRESHOLD_PLOT:-$PROJECT_ROOT/results/models/unet/unet_${SPLIT}_threshold_metrics.png}"
    ROC_PLOT="${ROC_PLOT:-$PROJECT_ROOT/results/models/unet/unet_${SPLIT}_roc.png}"
    SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/results/models/unet/unet_${SPLIT}_summary.csv}"
elif [[ "$MODEL_TYPE" == "attention_unet" ]]; then
    MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/results/models/attention_unet/attention_unet_weights.pt}"
    THRESHOLD_PLOT="${THRESHOLD_PLOT:-$PROJECT_ROOT/results/models/attention_unet/attention_unet_${SPLIT}_threshold_metrics.png}"
    ROC_PLOT="${ROC_PLOT:-$PROJECT_ROOT/results/models/attention_unet/attention_unet_${SPLIT}_roc.png}"
    SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/results/models/attention_unet/attention_unet_${SPLIT}_summary.csv}"
elif [[ "$MODEL_TYPE" == "classifier" ]]; then
    MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/results/models/classifier/classifier_weights.pt}"
    THRESHOLD_PLOT="${THRESHOLD_PLOT:-$PROJECT_ROOT/results/models/classifier/classifier_${SPLIT}_threshold_metrics.png}"
    ROC_PLOT="${ROC_PLOT:-}"
    SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_ROOT/results/models/classifier/classifier_${SPLIT}_summary.csv}"
else
    echo "MODEL_TYPE must be 'unet', 'attention_unet', or 'classifier', got '$MODEL_TYPE'" >&2
    exit 1
fi

mkdir -p "$(dirname "$THRESHOLD_PLOT")" "$(dirname "$SUMMARY_CSV")"
if [[ -n "$ROC_PLOT" ]]; then
    mkdir -p "$(dirname "$ROC_PLOT")"
fi

args=(
    "$SCRIPT_PATH"
    --model-type "$MODEL_TYPE"
    --model-path "$MODEL_PATH"
    --h5-path "$H5_PATH"
    --split "$SPLIT"
    --normalization "$NORMALIZATION"
    --threshold-min "$THRESHOLD_MIN"
    --threshold-max "$THRESHOLD_MAX"
    --threshold-count "$THRESHOLD_COUNT"
    --min-recall "$MIN_RECALL"
    --recall-penalty "$RECALL_PENALTY"
    --threshold-metrics-save-path "$THRESHOLD_PLOT"
    --summary-csv-path "$SUMMARY_CSV"
)

if [[ -n "$BATCH_SIZE" ]]; then
    args+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "$ROC_PLOT" ]]; then
    args+=(--roc-save-path "$ROC_PLOT")
fi

python3 "${args[@]}"
