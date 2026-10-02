#!/usr/bin/env bash
# Usage: bash scripts/run_n1_7_baseline.sh train [Hydra overrides...]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
export RLINF_PATH="$EMBODIED_TEMPLATE_ROOT/tmp/RLinf-n1.7-baseline"
export REPO_PATH="$RLINF_PATH"
export PYTHONPATH="$RLINF_PATH:$PYTHONPATH"

RUN_NAME="${1:-train}"
if (( $# > 0 )); then shift; fi
RUN_DIR="${EMBODIED_TEMPLATE_ROOT}/runs/n1.7-libero-baseline/${RUN_NAME}"
mkdir -p "$RUN_DIR"

export EMBODIED_PATH="$RLINF_PATH/examples/embodiment"
export LIBERO_CONFIG_PATH="$EMBODIED_TEMPLATE_ROOT/.venv-n1.7-libero/libero-config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MALLOC_TRIM_THRESHOLD_=0
export HYDRA_FULL_ERROR=1
unset DISPLAY

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv -lms 200 > "$RUN_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
vmstat -w -S M 1 > "$RUN_DIR/memory.log" &
MEMORY_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID" 2>/dev/null || true' EXIT

.venv-n1.7-libero/bin/python -u \
    "$RLINF_PATH/examples/embodiment/train_embodied_agent.py" \
    --config-path "$EMBODIED_TEMPLATE_ROOT/experiments/libero/ppo" \
    --config-name n1_7_smoke \
    "runner.logger.log_path=$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/console.log"

kill "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID"
trap - EXIT
.venv-n1.7-libero/bin/python scripts/verify_isaaclab_baseline.py \
    "$RUN_DIR" --baseline n1_7_libero
