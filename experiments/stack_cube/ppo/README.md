# PPO stage

Formal PPO is provisionally planned on one RTX PRO 6000 96GB (Blackwell)
rented from AutoDL with pay-as-you-go billing. Actor, rollout, and IsaacLab
will use GPU 0 on one Ray node, with offload and evaluation scheduling set
from target-GPU measurements. Freeze the vision/language backbone and train
the full action head plus value head. The existing projector-only smoke does
not validate this training scope; see
[the training plan](../../../docs/TRAINING_PIPELINE.md).

`isaaclab_n1_5_smoke.yaml` validates the existing GR00T N1.5 + IsaacLab path on
one GPU, with one environment, two action chunks and one PPO update. Setup,
checkpoint resume and measured results are in
[the baseline report](../../../docs/ISAACLAB_BASELINE.md).

`isaaclab_n1_7_smoke.yaml` runs the step-5 SFT policy evaluation.
`isaaclab_n1_7_ppo_smoke.yaml` runs a minimal PPO update with checkpoint saving
and fresh-process resume; see [the smoke report](../../../docs/STACK_CUBE_PPO_SMOKE.md).
Reusable adapter and worker changes belong in the RLinf fork.
