#!/usr/bin/env bash
#SBATCH -J trail-bench
#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# Stop immediately on errors, unset variables, or failed pipelines so the job does not
# continue after a broken training run.
set -euo pipefail


# Module handling matches the rest of your Slurm scripts so the job gets a predictable
# environment on the cluster node.
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6


# Fixed project paths keep the script simple to submit. Edit these in one place when you
# want to benchmark a different dataset or output location.
PROJECT_ROOT="/home/anp50/rds/hpc-work/satellite_trails"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"
SCRIPT_PATH="$PROJECT_ROOT/src/satellite_trail_segmentation/model/train.py"


# Training configuration is defined here rather than passed as job arguments so the
# benchmark is easy to rerun and compare across submissions.
DATA_PATH="$PROJECT_ROOT/data/h5s/small_set.h5"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_runs/a100_batch_sizes"
BATCH_SIZES=(8 16 32 64)
EPOCHS=3
LEARNING_RATE=0.001
LR_DECAY=1e4
DROPOUT_RATE=0.0


# GPU metrics are sampled once per second and written to CSV so you can inspect
# utilization and memory after the run instead of watching interactively.
GPU_LOG_INTERVAL_SECONDS=1


cd "$PROJECT_ROOT"
source "$VENV_PATH"
mkdir -p "$OUTPUT_DIR"


#fail fast if GPU does not load
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA device found'"


LOCAL_DATA_PATH="$TMPDIR/small_set.h5"
echo "Copying dataset to local scratch..."
cp "$DATA_PATH" "$LOCAL_DATA_PATH"
# Use local path from now on
DATA_PATH="$LOCAL_DATA_PATH"


for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    # Each batch size gets its own output folder so logs, checkpoints, and plots do not
    # overwrite one another.
    RUN_DIR="$OUTPUT_DIR/bs_${BATCH_SIZE}"
    CHECKPOINT_PATH="$RUN_DIR/save_model.pt"
    PLOT_PATH="$RUN_DIR/loss.png"
    TRAIN_LOG="$RUN_DIR/train.log"
    GPU_LOG="$RUN_DIR/gpu_metrics.csv"
    SUMMARY_PATH="$RUN_DIR/summary.txt"

    mkdir -p "$RUN_DIR"

    echo "Starting batch size ${BATCH_SIZE} at $(date -Iseconds)"

    # Background nvidia-smi logging captures GPU utilization and memory use over time.
    # The CSV is easy to inspect later in a spreadsheet or pandas.
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv \
        -l "$GPU_LOG_INTERVAL_SECONDS" \
        > "$GPU_LOG" &
    SMI_PID=$!

    # Wall-clock timing is recorded per batch-size run so you can compare throughput
    # directly from the summary files.
    START_TIME=$(date +%s)

    # Training output is streamed to the Slurm log and also saved per run with tee.
    python3 "$SCRIPT_PATH" \
        --data-path="$DATA_PATH" \
        --epochs="$EPOCHS" \
        --batch-size="$BATCH_SIZE" \
        --learning-rate="$LEARNING_RATE" \
        --lr-decay="$LR_DECAY" \
        --dropout-rate="$DROPOUT_RATE" \
        --save-path="$CHECKPOINT_PATH" \
        --plot-path="$PLOT_PATH" \
        2>&1 | tee "$TRAIN_LOG"

    END_TIME=$(date +%s)

    # Stop the background GPU logger cleanly so it does not continue into the next run.
    kill "$SMI_PID" || true
    wait "$SMI_PID" 2>/dev/null || true

    # The summary file gives a quick run-level comparison without opening the full logs.
    {
        echo "batch_size=${BATCH_SIZE}"
        echo "epochs=${EPOCHS}"
        echo "learning_rate=${LEARNING_RATE}"
        echo "elapsed_seconds=$((END_TIME - START_TIME))"
    } > "$SUMMARY_PATH"

    echo "Finished batch size ${BATCH_SIZE} in $((END_TIME - START_TIME)) seconds"
done
