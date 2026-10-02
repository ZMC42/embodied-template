# SFT stage

Formal SFT is provisionally planned on one RTX PRO 6000 96GB (Blackwell)
rented from AutoDL with pay-as-you-go billing. Freeze the vision/language
backbone and train the full action head (projector, DiT, and vlln). Validate
target-GPU compatibility and resources before the full run; see
[the training plan](../../../docs/TRAINING_PIPELINE.md).

The data contract is validated in `docs/STACK_CUBE_DATA_CONTRACT.md`.
`episode_split.json` freezes 117 train / 15 validation / 15 test episodes;
`data_contract.json` records the explicit state conversion, camera mapping,
relative IK semantics, and processor settings.

Run `bash scripts/validate_stack_cube_contract.sh` from the repository root
in the official SFT environment. Use the prepared
`runs/stack-cube/data/train` and `validation` datasets for the next micro SFT
stage. The raw dataset stores Euler xyz rotation, while the prepared state
uses principal axis-angle. Do not feed the raw parquets directly to SFT.

Step 4 is complete: `bash scripts/run_stack_cube_sft_micro.sh` runs one
projector-only update through the official N1.7 training pipeline, packages
`models/stack-cube-n1.7-sft`, and verifies it in a fresh, network-isolated
process against held-out episode 6. The full 292-frame open-loop plot and
raw predictions are in `runs/stack-cube/visualization/open-loop/`.
See `docs/STACK_CUBE_SFT_MICRO.md` for setup, reproduction, resources, and
metrics. This micro checkpoint verifies the engineering path; it is not a
trained stack-cube policy. RLinf + IsaacLab integration remains step 5.
