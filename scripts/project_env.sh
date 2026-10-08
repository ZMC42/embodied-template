#!/usr/bin/env bash

# Source this file from any directory:
#   source scripts/project_env.sh

EMBODIED_TEMPLATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EMBODIED_TEMPLATE_ROOT
if [[ -z "${EMBODIED_ASSET_ROOT:-}" ]]; then
    if [[ -L "$EMBODIED_TEMPLATE_ROOT/models" ]]; then
        EMBODIED_ASSET_ROOT="$(dirname "$(readlink -f "$EMBODIED_TEMPLATE_ROOT/models")")"
    else
        EMBODIED_ASSET_ROOT=/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets
    fi
fi
export EMBODIED_ASSET_ROOT
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    export CUDA_HOME
fi
export RLINF_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/RLinf"
export GR00T_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/Isaac-GR00T"
export ISAAC_LAB_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/IsaacLab"
export ISAAC_SIM_PATH="${EMBODIED_TEMPLATE_ROOT}/isaac-sim"
export REPO_PATH="$RLINF_PATH"
export PYTHONPATH="${EMBODIED_TEMPLATE_ROOT}/src:${RLINF_PATH}:${PYTHONPATH:-}"
