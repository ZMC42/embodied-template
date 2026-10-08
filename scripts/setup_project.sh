#!/usr/bin/env bash
# Configure the project on a Linux x86_64 NVIDIA GPU server.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="dev,sft-n1.7,ppo-isaaclab,isaaclab-n1.5,n1.7-libero"
SYSTEM_DEPS=1
DOWNLOAD_ASSETS=0
SMOKE_TEST=0
ASSET_ROOT="${EMBODIED_ASSET_ROOT:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_project.sh [options]

Default: install all five environments. Model/data downloads are optional.
Requires an existing NVIDIA driver and CUDA Toolkit (nvcc).

  --envs LIST          Comma-separated environment names, or all (default)
                       dev,sft-n1.7,ppo-isaaclab,isaaclab-n1.5,n1.7-libero
  --asset-root PATH    Asset storage (default: existing links, or .assets/)
  --skip-system-deps   Use already installed system libraries; do not run apt
  --download-assets    Also download pinned models, demonstrations and LIBERO assets
  --smoke-test         Also run one IsaacLab headless reset with cameras
  -h, --help           Show this help

Examples:
  bash scripts/setup_project.sh --asset-root /data/embodied-assets
  bash scripts/setup_project.sh --envs dev,sft-n1.7,ppo-isaaclab --download-assets

Cosmos downloads require Hugging Face access approved on the model page and
HF_TOKEN or a cached Hugging Face login. Tokens are never saved by this script.
EOF
}

while (( $# )); do
    case "$1" in
        --envs) ENVS="${2:?--envs requires a list}"; shift 2 ;;
        --asset-root) ASSET_ROOT="${2:?--asset-root requires a path}"; shift 2 ;;
        --skip-system-deps) SYSTEM_DEPS=0; shift ;;
        --download-assets) DOWNLOAD_ASSETS=1; shift ;;
        --smoke-test) SMOKE_TEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$ENVS" == all ]]; then
    ENVS="dev,sft-n1.7,ppo-isaaclab,isaaclab-n1.5,n1.7-libero"
fi
if (( SMOKE_TEST )) && [[ ",$ENVS," != *,ppo-isaaclab,* ]]; then
    echo '--smoke-test requires ppo-isaaclab in --envs.' >&2
    exit 2
fi
IFS=, read -r -a ENVIRONMENTS <<< "$ENVS"
for environment in "${ENVIRONMENTS[@]}"; do
    case "$environment" in
        dev|sft-n1.7|ppo-isaaclab|isaaclab-n1.5|n1.7-libero) ;;
        *) printf 'Unknown environment: %s\n' "$environment" >&2; exit 2 ;;
    esac
done

cd "$ROOT"
mkdir -p tmp/setup
LOG="$ROOT/tmp/setup/setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'printf "\nSetup failed at line %s. Log: %s\nRerun the same command after resolving the error.\n" "$LINENO" "$LOG" >&2' ERR

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'The pinned GPU wheels require Linux x86_64.' >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if (( SYSTEM_DEPS )); then
    APT=()
    if (( EUID != 0 )); then APT=(sudo); fi
    "${APT[@]}" apt-get update
    "${APT[@]}" apt-get install -y --no-install-recommends \
        build-essential pkg-config curl ca-certificates git git-lfs ffmpeg \
        libgl1 libegl1 libglib2.0-0 libgomp1 libaio-dev libvulkan1 \
        vulkan-tools procps util-linux unzip
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh -o tmp/setup/install-uv.sh
    UV_NO_MODIFY_PATH=1 sh tmp/setup/install-uv.sh
fi
uv --version

if [[ -z "$ASSET_ROOT" ]]; then
    if [[ -L "$ROOT/models" ]]; then
        ASSET_ROOT="$(dirname "$(readlink -f "$ROOT/models")")"
    else
        ASSET_ROOT="$ROOT/.assets"
    fi
fi
mkdir -p "$ASSET_ROOT"
EMBODIED_ASSET_ROOT="$(cd "$ASSET_ROOT" && pwd)"
export EMBODIED_ASSET_ROOT
# Project data is downloaded from pinned HF revisions, not GR00T demo LFS data.
export GIT_LFS_SKIP_SMUDGE=1
bash scripts/setup_assets.sh
source scripts/project_env.sh
nvcc --version

# Managed Python supplies headers needed by Triton/DeepSpeed on fresh servers.
export UV_MANAGED_PYTHON=1
uv python install 3.10 3.11.14
uv venv --allow-existing --seed .venvs/dev --python 3.10
uv pip install --python .venvs/dev/bin/python --no-deps -e .
uv pip install --python .venvs/dev/bin/python 'huggingface-hub==0.35.3'
export PATH="$ROOT/.venvs/dev/bin:$PATH"

BASELINE_ARGS=()
if (( ! DOWNLOAD_ASSETS )); then BASELINE_ARGS=(--skip-assets); fi
for environment in "${ENVIRONMENTS[@]}"; do
    printf '\nInstalling %s\n' "$environment"
    case "$environment" in
        dev) ;;
        sft-n1.7) bash scripts/setup_environments.sh sft ;;
        ppo-isaaclab) bash scripts/setup_environments.sh ppo ;;
        isaaclab-n1.5) .venvs/dev/bin/python scripts/setup_isaaclab_baseline.py "${BASELINE_ARGS[@]}" ;;
        n1.7-libero) .venvs/dev/bin/python scripts/setup_n1_7_baseline.py "${BASELINE_ARGS[@]}" ;;
    esac
    uv pip install --python ".venvs/$environment/bin/python" --no-deps -e .
    if [[ "$environment" != dev ]]; then
        ".venvs/$environment/bin/python" scripts/check_environment.py \
            --output "runs/preflight/$environment-check.json"
        uv pip freeze --exclude-editable --python ".venvs/$environment/bin/python" \
            > "runs/preflight/$environment-requirements.lock.txt"
    fi
done

if (( DOWNLOAD_ASSETS )); then
    .venvs/dev/bin/python scripts/download_assets.py all
    mkdir -p models/nvidia
    if [[ ! -e models/nvidia/Cosmos-Reason2-2B && ! -L models/nvidia/Cosmos-Reason2-2B ]]; then
        ln -s ../cosmos-reason2-2b models/nvidia/Cosmos-Reason2-2B
    fi
    .venvs/dev/bin/python scripts/preflight.py offline
fi

if (( SMOKE_TEST )); then
    OMNI_KIT_ACCEPT_EULA=YES .venvs/ppo-isaaclab/bin/python scripts/isaaclab_smoke.py
fi

printf '\nSetup complete. Environments: %s\nAssets: %s\nLog: %s\n' "$ENVS" "$EMBODIED_ASSET_ROOT" "$LOG"
echo 'For subsequent commands: source scripts/project_env.sh'
echo 'Activate an environment: source .venvs/<name>/bin/activate'
if (( ! DOWNLOAD_ASSETS )); then
    echo 'Models/data were not downloaded. Add --download-assets when preparing training assets.'
fi
