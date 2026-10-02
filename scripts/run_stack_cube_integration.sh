#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MALLOC_TRIM_THRESHOLD_=0
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
MODE="${1:-closed-loop}"
if (( $# > 0 )); then shift; fi
REPORT="$EMBODIED_TEMPLATE_ROOT/runs/stack-cube/integration/$MODE"
mkdir -p "$REPORT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv --loop-ms=200 > "$REPORT/gpu.csv" &
INTEGRATION_MONITOR_PID=$!
trap 'kill "$INTEGRATION_MONITOR_PID"' EXIT
PYTHON="$EMBODIED_TEMPLATE_ROOT/.venvs/ppo-isaaclab/bin/python"
if [[ "$MODE" == offline ]]; then
    export INTEGRATION_HOST_NETWORK_NAMESPACE="$(readlink /proc/self/ns/net)"
    unshare --user --map-root-user --net "$PYTHON" scripts/run_interactive_eval.py --offline-load --output "$REPORT" "$@" 2>&1 | tee "$REPORT/console.log"
else
    "$PYTHON" scripts/run_interactive_eval.py --output "$REPORT" "$@" 2>&1 | tee "$REPORT/console.log"
    "$PYTHON" scripts/verify_stack_cube_integration.py "$REPORT"
fi
