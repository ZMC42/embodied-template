#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv sync --frozen --project "${ROOT}/third_party/Isaac-GR00T"
"${ROOT}/third_party/Isaac-GR00T/.venv/bin/python" \
    "${ROOT}/scripts/preflight.py" runtime sft \
    --output "${ROOT}/runs/preflight/sft-environment.json"

(
    cd "${ROOT}/third_party/RLinf"
    GR00T_PATH="${ROOT}/third_party/Isaac-GR00T" \
    ISAAC_LAB_PATH="${ROOT}/third_party/IsaacLab" \
        bash requirements/install.sh embodied --env isaaclab
)
"${ROOT}/third_party/RLinf/.venv/bin/python" \
    "${ROOT}/scripts/preflight.py" runtime ppo \
    --output "${ROOT}/runs/preflight/ppo-environment.json"
