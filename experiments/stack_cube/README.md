# Stack-cube reference experiment

This experiment adapts GR00T N1.7 to the existing IsaacLab Franka stack-cube
task. It is the first concrete implementation of the reusable SFT-to-PPO flow.

## Inputs

- Dataset: `datasets/isaaclab-stack-cube`
- Base model: `models/gr00t-n1.7-3b`
- Backbone cache: `models/cosmos-reason2-2b`
- Isaac Sim: `isaac-sim`

## Outputs

- SFT checkpoint: `models/stack-cube-n1.7-sft`
- Experiment logs and PPO checkpoints: `runs/stack-cube/`

## Current state

The `sft` directory will hold the validated NVIDIA launch configuration. The
`ppo` directory will hold the RLinf Hydra experiment after the N1.7 + IsaacLab
contract has focused tests. Until then, the executable references remain the
two supported upstream baselines described in
[`docs/TRAINING_PIPELINE.md`](../../docs/TRAINING_PIPELINE.md).
