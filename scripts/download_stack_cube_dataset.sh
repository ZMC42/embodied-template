#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${PROJECT_ROOT}/scripts/download_assets.py" stack_cube_dataset
