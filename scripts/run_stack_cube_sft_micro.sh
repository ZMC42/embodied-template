#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
export PYTHONPATH="$EMBODIED_TEMPLATE_ROOT/src:$GR00T_PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# For hosts without python3.10-dev; see docs/STACK_CUBE_SFT_MICRO.md.
export CPATH="$EMBODIED_TEMPLATE_ROOT/tmp/sft-python-headers/usr/include/python3.10:$EMBODIED_TEMPLATE_ROOT/tmp/sft-python-headers/usr/include:${CPATH:-}"
PYTHON="$EMBODIED_TEMPLATE_ROOT/.venvs/sft-n1.7/bin/python"
REPORT="$EMBODIED_TEMPLATE_ROOT/runs/stack-cube/sft-micro"
mkdir -p "$REPORT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv --loop-ms=500 > "$REPORT/gpu.csv" &
SFT_GPU_MONITOR_PID=$!
trap 'kill "$SFT_GPU_MONITOR_PID"' EXIT
"$PYTHON" scripts/train_stack_cube_sft_micro.py 2>&1 | tee "$REPORT/training.log"
export SFT_HOST_NETWORK_NAMESPACE="$(readlink /proc/self/ns/net)"
unshare --user --map-root-user --net \
    "$PYTHON" scripts/verify_stack_cube_sft_micro.py 2>&1 | tee "$REPORT/offline.log"
