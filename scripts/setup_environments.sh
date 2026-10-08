#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="${1:-all}"

case "$ENVIRONMENT" in
    all|sft|ppo) ;;
    *) echo 'Usage: bash scripts/setup_environments.sh [all|sft|ppo]' >&2; exit 2 ;;
esac

if [[ "$ENVIRONMENT" == all || "$ENVIRONMENT" == sft ]]; then
    UV_PROJECT_ENVIRONMENT="${ROOT}/.venvs/sft-n1.7" \
        uv sync --frozen --python 3.10 --extra dev --project "${ROOT}/third_party/Isaac-GR00T"
    "${ROOT}/.venvs/sft-n1.7/bin/python" \
        "${ROOT}/scripts/preflight.py" runtime sft \
        --output "${ROOT}/runs/preflight/sft-environment.json"
fi

if [[ "$ENVIRONMENT" == all || "$ENVIRONMENT" == ppo ]]; then
    python3 "${ROOT}/scripts/setup_stack_cube_integration.py"
    "${ROOT}/.venvs/ppo-isaaclab/bin/python" \
        "${ROOT}/scripts/preflight.py" runtime ppo \
        --output "${ROOT}/runs/preflight/ppo-environment.json"
fi
