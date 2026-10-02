# Configurations

Project-owned SFT and PPO launch settings will live here after their contracts
have passed the corresponding baseline stage in
[`docs/TRAINING_PIPELINE.md`](../docs/TRAINING_PIPELINE.md).

Do not copy the GR00T N1.5 IsaacLab YAML and only change `model_type`. The N1.7
checkpoint processor, embodiment metadata, state/action representation, and
FSDP settings must be validated together.

The validated baseline environments are frozen in
`isaaclab_n1_5_requirements.lock.txt` and `n1_7_libero_requirements.lock.txt`.
Their model revisions, file hashes and source versions are recorded under
`baselines` in `dependencies.lock.json`. The N1.7 checkpoint fix is archived
in `patches/rlinf-n1_7-checkpoint.patch` so its pinned worktree can be rebuilt
without fetching an unpublished commit.
