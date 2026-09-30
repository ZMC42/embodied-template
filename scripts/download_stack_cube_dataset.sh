#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${EMBODIED_ASSET_ROOT:-/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets}"
DATASET_DIR="${ASSET_ROOT}/datasets/isaaclab-stack-cube"

if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required. Install huggingface_hub first." >&2
    exit 1
fi

mkdir -p "$DATASET_DIR"
hf download RLinf/IsaacLab-Stack-Cube-Data \
    --repo-type dataset \
    --local-dir "$DATASET_DIR"

echo "Dataset available at ${DATASET_DIR}."
echo "Project path: ${PROJECT_ROOT}/datasets/isaaclab-stack-cube"
