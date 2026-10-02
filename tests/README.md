# Tests

Add tests at the layer that owns each contract:

- dataset and modality validation belongs in this project;
- reusable model/environment behavior belongs in the RLinf fork;
- the final stack-cube run belongs in RLinf's embodied e2e suite.

The real-asset stack-cube contracts run with
`bash scripts/validate_stack_cube_contract.sh` in the official SFT environment.
The real IsaacLab observation/action manager check is
`scripts/verify_stack_cube_runtime.py`, run in the validated IsaacLab baseline
environment. Reproduction and evidence are documented in
[`STACK_CUBE_DATA_CONTRACT.md`](../docs/STACK_CUBE_DATA_CONTRACT.md).
