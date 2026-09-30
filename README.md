# Embodied Template

`embodied-template` is a reusable source project for learning and extending an
embodied training pipeline with NVIDIA Isaac-GR00T, IsaacLab, and RLinf. The
first reference experiment trains GR00T N1.7 on the IsaacLab Franka stack-cube
task through SFT followed by PPO.

The repository is currently a scaffold. Its supported baselines must be
validated before the new N1.7 + IsaacLab path is treated as runnable.

## Repository layout

```text
embodied-template/
├── configs/                  Project-owned experiment configurations
├── docs/                     Architecture and training plan
├── experiments/stack_cube/  First end-to-end reference experiment
├── scripts/                  Workspace and asset setup
├── src/embodied_template/    Project-specific adapters and entry points
├── tests/                    Contract and regression tests
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

The commands for frozen downloads, isolated environments, runtime manifests,
IsaacLab reset, and WebRTC route checks are in
[docs/PREFLIGHT.md](docs/PREFLIGHT.md).
