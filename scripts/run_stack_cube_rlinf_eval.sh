#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_TRIM_THRESHOLD_=0
export OMNI_KIT_ACCEPT_EULA=YES HYDRA_FULL_ERROR=1
unset DISPLAY LIVESTREAM
REPORT="$EMBODIED_TEMPLATE_ROOT/runs/stack-cube/integration/rlinf-eval"
mkdir -p "$REPORT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop-ms=200 > "$REPORT/gpu.csv" &
EVAL_MONITOR_PID=$!
trap 'kill "$EVAL_MONITOR_PID"' EXIT
.venvs/ppo-isaaclab/bin/python -u third_party/RLinf/evaluations/eval_embodied_agent.py \
  --config-path "$EMBODIED_TEMPLATE_ROOT/experiments/stack_cube/ppo" \
  --config-name isaaclab_n1_7_smoke "runner.logger.log_path=$REPORT" "$@" 2>&1 | tee "$REPORT/console.log"
.venvs/ppo-isaaclab/bin/python scripts/verify_stack_cube_integration.py "$REPORT" --ray
