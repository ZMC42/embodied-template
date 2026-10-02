# N1.7 + IsaacLab PPO smoke

本文记录训练流水线第 6 步的最小运行入口：将微型 stack-cube SFT bundle 交给 RLinf，运行 PPO update，保存训练状态，再用新进程恢复并将更新后的权重同步到 rollout worker。

## 配置与复现

`experiments/stack_cube/ppo/isaaclab_n1_7_ppo_smoke.yaml` 使用一个环境、固定 seed 0、5 个环境 step、1 步 action chunk、1 步 denoising、global batch 5、micro batch 1 和一个 update epoch。保留 256×256 front/wrist 双相机、8 维 state 和 7 维 relative IK action；embodiment 为 `libero_sim`，converter 为 `isaaclab_stack_cube`。

微型 SFT checkpoint 的配置是 projector-only，本次 PPO 沿用其冻结范围并新增可训练 value head：`tune_projector=true`，LLM、视觉、DiT 与 vlln 均冻结。FSDP 必须使用 `use_orig_params=true` 才能管理混合的冻结与可训练参数。首次遗漏该设置的初始化失败保存在 `fsdp-config-failure/`，未进入 PPO，也没有 OOM。此 smoke 的资源测量只覆盖该训练范围，正式 PPO 如解冻 DiT/vlln 需重新测量。

使用第 5 步的 `.venvs/ppo-isaaclab`，不修改 SFT bundle。actor/rollout offload、gradient checkpointing、128 MiB CPU bucket 同步与 `MALLOC_TRIM_THRESHOLD_=0` 沿用已验证的 N1.7 基线。RLinf `024713eb4c9a0608a1f77e68f5764955dfc8c005` 在第 5 步集成 commit `0bf6fd74` 上复用 optimizer 保存修复：checkpoint 写入期间保持 Adam state 在 CPU；16 项 FSDP 测试已通过。

当前 commit 仅在本地，归档 `configs/patches/rlinf-n1_7-isaaclab.patch` 包含集成和 checkpoint 修复两个提交。已在独立 worktree 实际验证以下重建得到相同 hash：

```bash
git -C third_party/RLinf worktree add --detach "$PWD/tmp/RLinf-ppo-smoke-replay" 61ba34e640035f2e4ac4ef3c6078f02d43a00c9a
git -C tmp/RLinf-ppo-smoke-replay am --committer-date-is-author-date "$PWD/configs/patches/rlinf-n1_7-isaaclab.patch"
GIT_COMMITTER_DATE='2026-10-02T18:30:54+08:00' git -C tmp/RLinf-ppo-smoke-replay commit --amend --no-edit
```

```bash
python scripts/setup_stack_cube_integration.py
bash scripts/run_stack_cube_ppo_smoke.sh train
bash scripts/run_stack_cube_ppo_smoke.sh resume \
  runner.max_epochs=2 \
  "runner.resume_dir=$PWD/runs/stack-cube/ppo-smoke/train/isaaclab_n1_7_ppo_smoke/checkpoints/global_step_1"
```

为避免同时驻留两个 Isaac Sim 进程，本配置关闭训练内 eval。首次进程运行 SFT 初始化策略的 rollout 并保存 step 1；恢复进程加载 step-1 模型、Adam、scheduler 与 RNG，在下一个 PPO update 前通过原生 CPU bucket 将恢复的模型同步给 rollout，再录制该 PPO checkpoint 的目标任务视频。恢复进程继续 update 并保存 step 2。两份 `video/train/seed_0/0.mp4` 因而分别记录 SFT 初始化策略和 step-1 PPO 策略，可按相同 seed 成对查看；这与第 5 步的 32 步 eval 配置分开运行。

两份视频均来自 train-mode rollout，保留相同的探索噪声。decoder 输出的 gripper sign 会在训练探索中受到扰动，实际 relative IK 环境仍按命令正负决定开合；轨迹中记录这些实际传入的浮点命令。短录像用于核对工程行为，正式 SFT/PPO 对比仍需独立固定-seed eval。

RLinf 自带 `CollectEpisode` 保存实际 observation/action/reward。`verify_isaaclab_baseline.py --baseline stack_cube_n1_7` 检查全部 TensorBoard scalar 有限、gradient norm 非零、同步耗时、DCP model/optimizer/scheduler metadata、Adam step、实际 action/state 形状、有限值和 MP4 解码，并把 processor/statistics/embodiment mapping 与 bundle manifest 的 SHA256 写进 `summary.json`。

恢复验证还比较首次报告中的 processor/bundle 校验值、action shape 与固定 seed 的初始 8 维 EEF state，确认 Adam step 延续；读取 DCP 中的 action decoder bias，与按加载精度转换为 BF16 的 SFT 权重比较，避免把初始化量化误差计为 PPO 更新。资源测量同时写入 `resource/*` TensorBoard scalar。

```bash
.venvs/ppo-isaaclab/bin/tensorboard --logdir runs/stack-cube/ppo-smoke --host 127.0.0.1 --port 6006
```

沿用 [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md) 的 SSH tunnel 方法查看；日志包含 success、loss、KL、value、gradient norm、同步耗时、采样峰值显存和完整 wall time。

## 产物与验收边界

运行产物保存在 NAS，经 `runs/stack-cube/ppo-smoke/` 访问。每次运行包含 `console.log`、200 ms 采样的 `gpu.csv`、每秒主机资源的 `memory.log`、TensorBoard、实际轨迹、视频、DCP checkpoint 和验证报告。checkpoint 不另存 `full_weights.pt`；恢复仍依赖固定 SFT bundle 的 processor 和本地 Cosmos backbone，不将 DCP 宣称为独立模型 bundle。

第 5 步的禁网加载与 WebRTC 证据继续有效。PPO 入口使用 `HF_HUB_OFFLINE=1` 与 `HF_DATASETS_OFFLINE=1` 加载本地模型；Isaac Sim 场景资产访问不纳入模型禁网检查。

5 步零 reward/success 只能验证工程链路，不能评价堆叠能力。正式 SFT、50-episode SFT 闭环评测及正式 PPO 仍须分别通过第 7–9 步的 gate。正式 SFT/PPO 暂定在 AutoDL 单张 RTX PRO 6000 96GB 上运行，冻结视觉与语言骨干、训练完整动作头；目标机器的兼容性及该训练范围的资源预算尚未验证，见 [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)。

## 2026-10-02 实测

第 6 步工程 gate 已通过，两次训练和验证入口均正常退出，没有 OOM。恢复进程使用 step-1 PPO 模型生成第二份 rollout，验证初始 8 维 EEF state 与首次运行在 seed 0 下一致，processor/bundle SHA256 与 action shape 均匹配；此检查不保证相机图像逐像素一致。

| 检查 | 首次训练 | 新进程恢复 |
| --- | --- | --- |
| checkpoint / TensorBoard / Adam step | 1 / 0 / 1 | 2 / 1 / 2 |
| actor total loss | 0.0535404 | 0.0466018 |
| critic value loss | 0.2677023 | 0.2330087 |
| approximate KL | 0.0 | 0.0 |
| gradient norm | 44.0 | 41.25 |
| action / state 记录形状 | `(5, 7)` / `(6, 8)` | `(5, 7)` / `(6, 8)` |
| action decoder bias 实际变化元素 | 7 | 7 |
| decoder 相对 SFT 的最大变化 | 1.00136×10⁻⁵ | 2.00272×10⁻⁵ |
| env reward / success | 0 / 0 | 0 / 0 |
| 峰值显存 | 22,674 MiB | 22,676 MiB |
| 完整 wall time | 408.0 s | 470.5 s |
| actor update wall time | 5.974 s | 5.830 s |
| CPU bucket 同步 wall time | 6.629 s | 5.285 s |
| MP4 | 6 帧，256×256，20 FPS | 6 帧，256×256，20 FPS |
| DCP 大小 | 8,247,558,227 bytes | 8,247,558,227 bytes |

全部 49 项 TensorBoard scalar（含 3 项资源指标）有限。DCP 保存 model、Adam、scheduler 和 RNG；实际核对的 decoder tensor 为 `action_head.action_decoder.layer2.b`。两份短视频位于 `train/video/train/seed_0/0.mp4` 和 `resume/video/train/seed_0/0.mp4`，原始双相机和实际执行 action 在同一 run 的 `trajectories/*.pkl` 中。

已解码并查看两份视频的首尾帧，对照图为 `sft-ppo-seed-0.png`。记录的 EEF xyz 各轴范围分别约 4.44 / 3.51 / 2.22 cm 和 2.44 / 4.02 / 1.23 cm；物理运动已发生，方块尚未堆叠。`MANIFEST.json` 汇总源码、配置/入口/环境 SHA256、两份验收报告、训练范围和限制。

CUDA 报告容量为 24,080 MiB，峰值约占 94%，超过正式作业 90% 的目标。主机约 62.6 GiB RAM、2 GiB swap，恢复保存阶段 swap 已达到 2 GiB；主机资源采样完整保留在 `memory.log`。wall time 包含 NAS 模型和 DCP 读写；首次运行期间曾启动全资产 checksum sweep，随后停止以免竞争 checkpoint I/O，未将其计为通过检查。本轮源码、lock 输入和归档 patch 校验见 `source-lock-check.log`；资产和依赖兼容边界沿用第 4–5 步。

首次成功训练结束、指标与 checkpoint 写出后，Ray 清理阶段出现 Python 3.11 `resource_tracker` 的 semaphore cleanup `KeyError`；训练进程 exit 0，产物验证通过，GPU 回到 42 MiB。该退出清理问题未修复；新进程恢复运行没有出现相同输出。保留原始日志，不宣称全程没有警告。
