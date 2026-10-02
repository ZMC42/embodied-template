#!/usr/bin/env bash
# Usage: bash scripts/run_stack_cube_ppo_smoke.sh train [Hydra overrides...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
RUN_NAME="${1:-train}"
if (( $# > 0 )); then shift; fi
RUN_DIR="$EMBODIED_TEMPLATE_ROOT/runs/stack-cube/ppo-smoke/$RUN_NAME"
mkdir -p "$RUN_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_TRIM_THRESHOLD_=0
export OMNI_KIT_ACCEPT_EULA=YES HYDRA_FULL_ERROR=1
unset DISPLAY LIVESTREAM
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop-ms=200 > "$RUN_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
vmstat -w -S M 1 > "$RUN_DIR/memory.log" &
MEMORY_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID"' EXIT
.venvs/ppo-isaaclab/bin/python -u third_party/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path "$EMBODIED_TEMPLATE_ROOT/experiments/stack_cube/ppo" \
  --config-name isaaclab_n1_7_ppo_smoke "runner.logger.log_path=$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/console.log"
kill "$GPU_MONITOR_PID" "$MEMORY_MONITOR_PID"
trap - EXIT
.venvs/ppo-isaaclab/bin/python scripts/verify_isaaclab_baseline.py "$RUN_DIR" --baseline stack_cube_n1_7
