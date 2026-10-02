# SFT stage

The data contract is validated in `docs/STACK_CUBE_DATA_CONTRACT.md`.
`episode_split.json` freezes 117 train / 15 validation / 15 test episodes;
`data_contract.json` records the explicit state conversion, camera mapping,
relative IK semantics, and processor settings.

Run `bash scripts/validate_stack_cube_contract.sh` from the repository root
in the official SFT environment. Use the prepared
`runs/stack-cube/data/train` and `validation` datasets for the next micro SFT
stage. The raw dataset stores Euler xyz rotation, while the prepared state
uses principal axis-angle. Do not feed the raw parquets directly to SFT.

The official SFT wrapper and micro checkpoint are step 4 and remain pending.
