#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${EMBODIED_ASSET_ROOT:-/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets}"

link_asset() {
    local name="$1"
    local target="$2"
    local link_path="${PROJECT_ROOT}/${name}"

    mkdir -p "$target"

    if [[ -L "$link_path" ]]; then
        if [[ "$(readlink -f "$link_path")" == "$(readlink -f "$target")" ]]; then
            return
        fi
        echo "Refusing to replace existing link: ${link_path}" >&2
        exit 1
    fi

    if [[ -e "$link_path" ]]; then
        echo "Refusing to replace existing path: ${link_path}" >&2
        exit 1
    fi

    ln -s "$target" "$link_path"
}

git -C "$PROJECT_ROOT" submodule update --init \
    third_party/RLinf \
    third_party/Isaac-GR00T \
    third_party/IsaacLab

if ! git -C "${PROJECT_ROOT}/third_party/RLinf" remote get-url upstream >/dev/null 2>&1; then
    git -C "${PROJECT_ROOT}/third_party/RLinf" remote add upstream \
        https://github.com/RLinf/RLinf.git
fi

link_asset "models" "${ASSET_ROOT}/models"
link_asset "datasets" "${ASSET_ROOT}/datasets"
link_asset "isaac-sim" "${ASSET_ROOT}/simulators/isaac-sim-5.1.0"
link_asset "runs" "${ASSET_ROOT}/runs"

echo "Workspace assets are linked from ${ASSET_ROOT}."
