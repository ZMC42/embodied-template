#!/usr/bin/env bash
# Prerequisite: the official SFT environment and pinned assets (see docs/STACK_CUBE_DATA_CONTRACT.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
cd "$EMBODIED_TEMPLATE_ROOT"
export PYTHONPATH="$EMBODIED_TEMPLATE_ROOT/src:$GR00T_PATH:$RLINF_PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4
PYTHON="$EMBODIED_TEMPLATE_ROOT/.venvs/sft-n1.7/bin/python"
REPORT="$EMBODIED_TEMPLATE_ROOT/runs/stack-cube/contract"
mkdir -p "$REPORT"
"$PYTHON" scripts/prepare_stack_cube_dataset.py | tee "$REPORT/prepare.log"
"$PYTHON" -m pytest tests/test_stack_cube_contract.py -q \
    --junitxml="$REPORT/pytest.xml" | tee "$REPORT/pytest.log"
for EPISODE in 0 7; do
    "$PYTHON" scripts/replay_stack_cube_dataset.py --episode "$EPISODE"
done
uv pip freeze --python "$PYTHON" > "$REPORT/requirements.lock.txt"
"$PYTHON" - <<'PY'
import importlib.metadata
import platform
import subprocess
from pathlib import Path
import av
import torch
import torchcodec
from embodied_template.stack_cube import ROOT, CONTRACT_PATH, SPLIT_PATH, read_json, sha256, write_json

report = ROOT / 'runs/stack-cube/contract'
libs = sorted(Path(torchcodec.__file__).parent.glob('libtorchcodec_decoder*.so'))
linkage = {p.name: subprocess.run(['ldd',str(p)],capture_output=True,text=True,check=True).stdout for p in libs}
ffmpeg_libraries = {}
for output in linkage.values():
    for line in output.splitlines():
        if any(name in line for name in ['libav', 'libswscale', 'libswresample']) and '=>' in line and 'not found' not in line:
            path = line.split('=>')[1].split()[0]
            ffmpeg_libraries[path] = sha256(path)
loaded_paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line}
loaded_decoders = sorted(p for p in loaded_paths if 'libtorchcodec_decoder' in p)
assert loaded_decoders
assert ffmpeg_libraries
write_json(report / 'environment.json', {
    'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['torch','torchvision','transformers','av','torchcodec','numpy','pandas','scipy','pytest','flash-attn']},
    'torch_cuda': torch.version.cuda, 'pyav_ffmpeg_libraries': av.library_versions,
    'torchcodec_linkage': linkage, 'torchcodec_loaded_decoders': loaded_decoders,
    'ffmpeg_shared_library_sha256': ffmpeg_libraries,
    'official_uv_lock_sha256': sha256(ROOT/'third_party/Isaac-GR00T/uv.lock'),
    'requirements_lock_sha256': sha256(report/'requirements.lock.txt'),
    'source_commits': {k:v['commit'] for k,v in read_json(ROOT/'configs/dependencies.lock.json')['sources'].items()},
    'contract_sha256': sha256(CONTRACT_PATH), 'split_sha256': sha256(SPLIT_PATH),
    'dataset_revision': read_json(SPLIT_PATH)['revision'],
    'asset_manifests': {name: sha256(ROOT / path / '.asset-manifest.json') for name,path in [('dataset','datasets/isaaclab-stack-cube'),('base_model','models/gr00t-n1.7-3b'),('cosmos','models/cosmos-reason2-2b')]},
    'scope': 'CPU data decoding and processor validation; no policy weights or SFT updates loaded',
})
PY
