# Embodied Template

`embodied-template` is a reusable source project for learning and extending an
embodied training pipeline with NVIDIA Isaac-GR00T, IsaacLab, and RLinf. The
first reference experiment trains GR00T N1.7 on the IsaacLab Franka stack-cube
task through SFT followed by PPO.

Formal SFT and PPO are provisionally planned on one RTX PRO 6000 96GB
(Blackwell) rented from AutoDL with pay-as-you-go billing. Both stages will
freeze the vision/language backbone and train the full action head; PPO also
trains the value head. Target-GPU compatibility and resource validation are
still required; see [the training plan](docs/TRAINING_PIPELINE.md).

The GR00T N1.5 + IsaacLab and N1.7 + LIBERO baselines have passed PPO updates,
checkpoint saves and resume on one RTX 4090. The N1.7 + IsaacLab integration
remains to be validated. See the [IsaacLab report](docs/ISAACLAB_BASELINE.md)
and [N1.7 report](docs/N1_7_BASELINE.md) for setup, reproduction commands and
measured results.

## Repository layout

```text
embodied-template/
├── configs/                  Project-owned experiment configurations
├── docs/                     Architecture and training plan
├── experiments/stack_cube/  First end-to-end reference experiment
├── scripts/                  Workspace and asset setup
├── src/embodied_template/    Project-specific adapters and entry points
├── tests/                    Contract and regression tests
├── .venvs/                   Local Python environments, ignored by Git
└── third_party/
    ├── Isaac-GR00T/          NVIDIA N1.7 source, pinned submodule
    ├── IsaacLab/             RLinf IsaacLab fork, pinned submodule
    └── RLinf/                ZMC42 RLinf fork, pinned submodule
```

Large assets live under
`/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets`. The project exposes stable
paths (`models`, `datasets`, `isaac-sim`, and `runs`) through local symbolic
links.

## Initialize the workspace

```bash
bash scripts/setup_assets.sh
source scripts/project_env.sh
```

To download the public stack-cube demonstrations after installing the
Hugging Face CLI:

```bash
bash scripts/download_stack_cube_dataset.sh
```

Read [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) before installing
GPU environments. It records the validated combinations, known compatibility
gaps, and the order in which the pipeline will be implemented.

All Python environments live under `.venvs/`:

| Directory | Purpose | Setup |
| --- | --- | --- |
| `dev/` | Lightweight project environment | Commands below |
| `isaaclab-n1.5/` | Validated N1.5 + IsaacLab baseline | `python scripts/setup_isaaclab_baseline.py` |
| `n1.7-libero/` | Validated N1.7 + LIBERO baseline | `python scripts/setup_n1_7_baseline.py` |
| `sft-n1.7/` | Official N1.7 SFT dependencies | `bash scripts/setup_environments.sh` |
| `ppo-isaaclab/` | IsaacLab PPO foundation for N1.7 integration | `bash scripts/setup_environments.sh` |

Create the lightweight environment when needed:

```bash
uv venv .venvs/dev --python 3.10
uv pip install --python .venvs/dev/bin/python --no-deps -e .
```

Use the desired environment's `bin/python` directly, or activate it with
`source .venvs/<name>/bin/activate`. With an activated environment, use
`uv run --active` to avoid creating a separate root `.venv`.

The baselines install different GR00T source versions and Transformers pins;
official SFT also requires a different Python/Torch combination. Keep these
environments separate. Create only the environments needed for the current
stage. Rebuild at the target path when relocating an environment: activation
scripts, CLI shebangs and LIBERO configuration contain absolute paths.

The commands for frozen downloads, isolated environments, runtime manifests,
IsaacLab reset, and WebRTC route checks are in
[docs/PREFLIGHT.md](docs/PREFLIGHT.md).
