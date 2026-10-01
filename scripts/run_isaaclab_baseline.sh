#!/usr/bin/env bash
# Usage: bash scripts/run_isaaclab_baseline.sh train [Hydra overrides...]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"

RUN_NAME="${1:-train}"
if (( $# > 0 )); then shift; fi
RUN_DIR="${EMBODIED_TEMPLATE_ROOT}/runs/isaaclab-n1.5-baseline/${RUN_NAME}"
mkdir -p "$RUN_DIR"

export OMNI_KIT_ACCEPT_EULA=YES
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
unset DISPLAY

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv -lms 200 > "$RUN_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

.venv-isaaclab-n1.5/bin/python -u \
    third_party/RLinf/examples/embodiment/train_embodied_agent.py \
    --config-path "$EMBODIED_TEMPLATE_ROOT/experiments/stack_cube/ppo" \
    --config-name isaaclab_n1_5_smoke \
    "runner.logger.log_path=$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/console.log"

kill "$GPU_MONITOR_PID"
trap - EXIT
.venv-isaaclab-n1.5/bin/python scripts/verify_isaaclab_baseline.py "$RUN_DIR"
