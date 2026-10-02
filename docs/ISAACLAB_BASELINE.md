# IsaacLab N1.5 基线验证

本文记录训练流水线第 1 步的实测结果和复现命令：在一张 RTX 4090 上使用 RLinf 的 GR00T N1.5 + IsaacLab stack-cube 链路，完成短 rollout、PPO update、checkpoint 保存与新进程恢复。

## 配置与版本

项目配置是 [`isaaclab_n1_5_smoke.yaml`](../experiments/stack_cube/ppo/isaaclab_n1_5_smoke.yaml)，由固定 RLinf commit 中的 `isaaclab_franka_stack_cube_ppo_gr00t` 缩小得到。保留 256 × 256 双相机、7 维 action、16 步 action chunk、4 步 denoising 和官方 PPO loss；训练环境数量为 1，每轮执行 32 个 simulation step，global batch 为 2，micro batch 为 1，update epoch 为 1。actor 与 rollout 都启用 offload，actor 启用 gradient checkpointing。

`configs/dependencies.lock.json` 的 `baselines.isaaclab_n1_5` 固定额外的 N1.5 源码、模型 revision、模型文件 SHA256 和环境锁。N1.5 源码放在 `tmp/Isaac-GR00T-n1.5` worktree，运行环境为 `.venvs/isaaclab-n1.5`。

| 项目 | 验证版本 |
| --- | --- |
| GPU / driver | RTX 4090 24 GB / 580.178.04 |
| Python | 3.11.14 |
| Torch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| FlashAttention | 2.8.3，cu12 / torch2.8 / cxx11abiTRUE wheel |
| Transformers / Ray | 4.51.3 / 2.49.2 |
| Isaac Sim | 5.1.0.0，pip 安装，含 extension cache |
| IsaacLab | 源码 VERSION 为 2.3.0，Python distribution 为 0.48.8 |
| GR00T N1.5 源码 | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` |
| N1.5 stack-cube 权重 | `RLinf/RLinf-Gr00t-SFT-Stack-cube`，revision `911e6b13f68ff97ac959a131dd643709ff89f042` |

RLinf、IsaacLab 和 N1.7 源码沿用项目原有的固定 commit。N1.5 checkpoint 仓库没有提供许可证文件，保留 N1.5 源码中的 NVIDIA `LICENSE`；此记录不替代 checkpoint 发布者的授权声明。

## 复现

先初始化源码与 NAS 资产路径，再建立独立环境并下载固定 revision 的权重：

```bash
bash scripts/setup_assets.sh
python scripts/setup_isaaclab_baseline.py
```

安装器校验三个原有源码 commit、N1.5 worktree commit、环境锁和模型文件哈希。它使用 `configs/isaaclab_n1_5_requirements.lock.txt` 安装实际验证的运行时组合，再以 editable 方式接入固定源码。Isaac Sim 使用 pip package，不需要空的 `isaac-sim` NAS 目录中存在独立安装包。

环境锁使用 `--no-deps`，因为 Isaac Sim 5.1 的 metadata 固定 Torch 2.7，而本次组合使用 Torch 2.8；N1.5 的 metadata 固定 Ray 2.40，而 RLinf 要求 Ray ≥ 2.47。N1.5 editable 安装还保留了官方训练工具的依赖声明，因此 `uv pip check` 会报告其他未沿用的 SFT 依赖。完整检查输出保存在 `runs/isaaclab-n1.5-baseline/dependency-check.txt`。本环境的验证范围是这条 RL 基线；实际通过了 policy import、FlashAttention GPU kernel、headless reset 和训练检查。

运行一次 update，然后在新进程中恢复到 step 1，再执行一次 update：

```bash
bash scripts/run_isaaclab_baseline.sh train
bash scripts/run_isaaclab_baseline.sh resume \
  runner.max_epochs=2 \
  "runner.resume_dir=$PWD/runs/isaaclab-n1.5-baseline/train/isaaclab_n1_5_smoke/checkpoints/global_step_1"
```

`runner.max_epochs=2` 是累计训练到 step 2；恢复进程只增加一次 update。入口使用 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 加载本地模型，关闭 DISPLAY，启用 headless 相机渲染，并每 200 ms 采样整张 GPU 的显存和利用率。IsaacLab 场景资产仍可能通过 NVIDIA 资产服务加载，以上设置不是对仿真进程的网络隔离。

入口在训练进程成功退出后调用 `verify_isaaclab_baseline.py`：检查全部 TensorBoard scalar 的有限性、非零 gradient norm、最后的 logging step、模型权重、optimizer 与 scheduler 的 checkpoint metadata，并解码 MP4。结果写入各运行目录的 `summary.json`。

## 2026-10-01 实测

第 1 步已完成。目标 task `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` 已完成一次独立 headless reset。首次训练完成环境 reset、两次 action chunk rollout、reward 收集、权重同步、一次 actor update 和 step-1 checkpoint 保存。新进程加载 step-1 checkpoint 后，再次完成 rollout、权重同步、actor update 和 step-2 保存。两次训练进程均以 exit code 0 退出，GPU 显存回到运行前水平，没有 shape、device、非有限值或 worker 生命周期错误。

| 检查 | 首次训练 | 新进程恢复 |
| --- | --- | --- |
| checkpoint step / TensorBoard step | 1 / 0 | 2 / 1 |
| actor total loss | 0.00557448 | 0.00646290 |
| critic value loss | 0.01114896 | 0.01292581 |
| approximate KL | 0 | 0 |
| gradient norm | 68.5 | 197 |
| episode length / reward / success | 32 / 0 / 0 | 32 / 0 / 0 |
| GPU 峰值显存，含初始化、update 和保存 | 23,826 MiB | 23,013 MiB |
| wall time，含初始化、加载和 checkpoint 写入 | 348.6 s | 465.1 s |
| actor update wall time | 5.52 s | 5.48 s |
| rollout MP4 | 33 帧，256 × 256，20 FPS | 33 帧，256 × 256，20 FPS |

所有记录的 scalar 均为有限值。短 rollout 的 reward 和 success 为零，只证明工程链路可以执行，不能证明任务学习效果。GPU 峰值约为容量的 97%，尚未达到正式作业将峰值控制在 90% 以内的预算目标；此 smoke 的 batch 和环境规模不能直接作为正式训练规模。

原始产物保存在 `runs/isaaclab-n1.5-baseline/`：

```text
train/、resume/
├── console.log
├── gpu.csv
├── summary.json
├── tensorboard/
├── video/train/seed_0/0.mp4
└── isaaclab_n1_5_smoke/checkpoints/global_step_<1 或 2>/actor/
    ├── dcp_checkpoint/       模型、Adam optimizer 与 scheduler
    └── model_state_dict/     full_weights.pt
```

每次 checkpoint 约 14.2 GiB，其中包含约 9.1 GiB 的分布式训练状态和 5.1 GiB 的完整模型权重。模型与 checkpoint 保存在 NAS；仿真、shader、HF 和 Torch cache 使用本地目录。实测 wall time 包含 NAS 上的模型加载和 checkpoint 写入，不能当作纯 rollout/update 吞吐。
