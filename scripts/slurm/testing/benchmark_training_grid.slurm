#!/usr/bin/env bash
#SBATCH -J train-grid
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail


#Module handling
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6


#Paths
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
UNET_SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/model/train.py"
CLASSIFIER_SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/trail_class_model/train.py"


#Benchmark configuration
DATA_PATH="$PROJECT_ROOT/data/h5s/small_dataset.h5"
OUTPUT_DIR="$PROJECT_ROOT/results/benchmark_runs"
BATCH_SIZES=(8 16 32 64)
NUM_WORKERS_LIST=(0 2 4 8)
EPOCHS=3
LEARNING_RATE=0.001
LR_DECAY=1e4
DROPOUT_RATE=0.0
P_AUG=0.0
POS_WEIGHT=10.0
BASE_CHANNELS=16
GPU_LOG_INTERVAL_SECONDS=1


cd "$PROJECT_ROOT"
source "$VENV_PATH"
mkdir -p "$OUTPUT_DIR"


#fail fast if torch cannot load to CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


LOCAL_DATA_PATH="$TMPDIR/small_dataset.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
DATA_PATH="$LOCAL_DATA_PATH"


UNET_OUTPUT_DIR="$OUTPUT_DIR/unet"
CLASSIFIER_OUTPUT_DIR="$OUTPUT_DIR/trail_class_model"
UNET_SUMMARY_CSV="$UNET_OUTPUT_DIR/summary.csv"
CLASSIFIER_SUMMARY_CSV="$CLASSIFIER_OUTPUT_DIR/summary.csv"

mkdir -p "$UNET_OUTPUT_DIR" "$CLASSIFIER_OUTPUT_DIR"

echo "model,batch_size,num_workers,epochs,learning_rate,elapsed_seconds,exit_code,avg_gpu_utilization_percent,avg_gpu_memory_utilization_percent,max_gpu_memory_used_mib,run_dir" > "$UNET_SUMMARY_CSV"
echo "model,batch_size,num_workers,epochs,learning_rate,elapsed_seconds,exit_code,avg_gpu_utilization_percent,avg_gpu_memory_utilization_percent,max_gpu_memory_used_mib,run_dir" > "$CLASSIFIER_SUMMARY_CSV"


run_benchmark() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local BATCH_SIZE="$3"
    local NUM_WORKERS="$4"
    local MODEL_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_NAME}"
    local SUMMARY_CSV="$MODEL_OUTPUT_DIR/summary.csv"

    local RUN_DIR="$MODEL_OUTPUT_DIR/bs_${BATCH_SIZE}/workers_${NUM_WORKERS}"
    local TRAIN_LOG="$RUN_DIR/train.log"
    local GPU_LOG="$RUN_DIR/gpu_metrics.csv"
    local RUN_EXIT_CODE=0

    mkdir -p "$RUN_DIR"

    echo "Starting ${MODEL_NAME} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} at $(date -Iseconds)"

    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv \
        -l "$GPU_LOG_INTERVAL_SECONDS" \
        > "$GPU_LOG" &
    SMI_PID=$!

    START_TIME=$(date +%s)

    set +e
    if [[ "$MODEL_NAME" == "unet" ]]; then
        python3 "$SCRIPT_PATH" \
            --data-path="$DATA_PATH" \
            --epochs="$EPOCHS" \
            --batch-size="$BATCH_SIZE" \
            --learning-rate="$LEARNING_RATE" \
            --lr-decay="$LR_DECAY" \
            --dropout-rate="$DROPOUT_RATE" \
            --augmentation-prob="$P_AUG" \
            --num-workers="$NUM_WORKERS" \
            2>&1 | tee "$TRAIN_LOG"
        RUN_EXIT_CODE=${PIPESTATUS[0]}
    else
        python3 "$SCRIPT_PATH" \
            --data-path="$DATA_PATH" \
            --epochs="$EPOCHS" \
            --batch-size="$BATCH_SIZE" \
            --learning-rate="$LEARNING_RATE" \
            --lr-decay="$LR_DECAY" \
            --dropout-rate="$DROPOUT_RATE" \
            --pos-weight="$POS_WEIGHT" \
            --base-channels="$BASE_CHANNELS" \
            --num-workers="$NUM_WORKERS" \
            2>&1 | tee "$TRAIN_LOG"
        RUN_EXIT_CODE=${PIPESTATUS[0]}
    fi
    set -e

    END_TIME=$(date +%s)

    kill "$SMI_PID" || true
    wait "$SMI_PID" 2>/dev/null || true

    ELAPSED_SECONDS=$((END_TIME - START_TIME))
    AVG_GPU_UTIL=$(awk -F, '
        NR > 1 { gsub(/[^0-9.]/, "", $4); if ($4 ~ /^[0-9]+(\.[0-9]+)?$/) { sum += $4; count++ } }
        END { if (count > 0) printf "%.2f", sum/count; else printf "NA" }
    ' "$GPU_LOG")

    AVG_GPU_MEMORY_UTIL=$(awk -F, '
        NR > 1 { gsub(/[^0-9.]/, "", $5); if ($5 ~ /^[0-9]+(\.[0-9]+)?$/) { sum += $5; count++ } }
        END { if (count > 0) printf "%.2f", sum/count; else printf "NA" }
    ' "$GPU_LOG")

    MAX_GPU_MEMORY_USED=$(awk -F, '
        NR > 1 { gsub(/[^0-9.]/, "", $6); if ($6 ~ /^[0-9]+(\.[0-9]+)?$/ && $6 > max) max = $6 }
        END { if (max != "") printf "%.0f", max; else printf "NA" }
    ' "$GPU_LOG")
    echo "${MODEL_NAME},${BATCH_SIZE},${NUM_WORKERS},${EPOCHS},${LEARNING_RATE},${ELAPSED_SECONDS},${RUN_EXIT_CODE},${AVG_GPU_UTIL},${AVG_GPU_MEMORY_UTIL},${MAX_GPU_MEMORY_USED},${RUN_DIR}" >> "$SUMMARY_CSV"


    if [[ "$RUN_EXIT_CODE" -eq 0 ]]; then
        echo "Finished ${MODEL_NAME} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} in ${ELAPSED_SECONDS} seconds"
    else
        echo "Failed ${MODEL_NAME} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} with exit code ${RUN_EXIT_CODE} after ${ELAPSED_SECONDS} seconds"
    fi
}


for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for NUM_WORKERS in "${NUM_WORKERS_LIST[@]}"; do
        run_benchmark "unet" "$UNET_SCRIPT_PATH" "$BATCH_SIZE" "$NUM_WORKERS"
    done
done


for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for NUM_WORKERS in "${NUM_WORKERS_LIST[@]}"; do
        run_benchmark "trail_class_model" "$CLASSIFIER_SCRIPT_PATH" "$BATCH_SIZE" "$NUM_WORKERS"
    done
done


echo "Benchmark complete."
echo "UNet summary written to $UNET_SUMMARY_CSV"
echo "Trail classifier summary written to $CLASSIFIER_SUMMARY_CSV"
