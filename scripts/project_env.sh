#!/usr/bin/env bash

# Source this file from any directory:
#   source scripts/project_env.sh

export EMBODIED_TEMPLATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EMBODIED_ASSET_ROOT="${EMBODIED_ASSET_ROOT:-/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets}"
export RLINF_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/RLinf"
export GR00T_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/Isaac-GR00T"
export ISAAC_LAB_PATH="${EMBODIED_TEMPLATE_ROOT}/third_party/IsaacLab"
export ISAAC_SIM_PATH="${EMBODIED_TEMPLATE_ROOT}/isaac-sim"
export REPO_PATH="$RLINF_PATH"
export PYTHONPATH="${EMBODIED_TEMPLATE_ROOT}/src:${RLINF_PATH}:${PYTHONPATH:-}"
