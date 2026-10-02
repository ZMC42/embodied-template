#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UV_PROJECT_ENVIRONMENT="${ROOT}/.venvs/sft-n1.7" \
    uv sync --frozen --project "${ROOT}/third_party/Isaac-GR00T"
"${ROOT}/.venvs/sft-n1.7/bin/python" \
    "${ROOT}/scripts/preflight.py" runtime sft \
    --output "${ROOT}/runs/preflight/sft-environment.json"

(
    cd "${ROOT}/third_party/RLinf"
    GR00T_PATH="${ROOT}/third_party/Isaac-GR00T" \
    ISAAC_LAB_PATH="${ROOT}/third_party/IsaacLab" \
        bash requirements/install.sh embodied --env isaaclab \
            --venv "${ROOT}/.venvs/ppo-isaaclab"
)
"${ROOT}/.venvs/ppo-isaaclab/bin/python" \
    "${ROOT}/scripts/preflight.py" runtime ppo \
    --output "${ROOT}/runs/preflight/ppo-environment.json"
