# N1.7 + LIBERO 基线验证

本文记录训练流水线第 2 步的复现方法和单卡验证结果：使用 RLinf 维护的 GR00T N1.7 + LIBERO Spatial 链路，检查模型与 processor 加载、短 rollout、PPO update、权重同步、checkpoint 保存和新进程恢复。

## 配置与版本

配置见 `experiments/libero/ppo/n1_7_smoke.yaml`。它由固定 RLinf 的 `libero_spatial_ppo_gr00t_n1d7` 缩小得到：1 个环境、固定 task/reset ID 0、5 个环境 step、1 步 action chunk、1 步 denoising、global batch 5、micro batch 1、update epoch 1。保留 256 × 256 双相机和 7 维动作；actor/rollout 启用 offload，actor 启用 gradient checkpointing。训练后立即同步更新权重并录制 eval 视频。

使用独立环境 `.venvs/n1.7-libero`。`configs/dependencies.lock.json` 的 `baselines.n1_7_libero` 固定基线模型 revision、模型文件 SHA256、LIBERO 场景资产 revision 和运行环境锁。

| 项目 | 版本 |
| --- | --- |
| GPU / driver | RTX 4090 24 GB / 580.178.04 |
| Python | 3.11.14 |
| Torch / CUDA | 2.8.0+cu128 / 12.8 |
| FlashAttention | 2.8.3 |
| Transformers / Ray | 4.57.3 / 2.58.0 |
| LIBERO / robosuite / MuJoCo | rlinf-libero 0.1.3 / 1.4.1 / 3.3.7 |
| RLinf 基线 worktree | `0e60183035192ccd79279018f5e3366ee6346695`，父 commit 为项目固定的 `61ba34e6` |
| Isaac-GR00T N1.7 | `23ace64f17aa5015259b8609d371eb61a357c776` |
| N1.7 LIBERO 权重 | `nvidia/GR00T-N1.7-LIBERO`，`2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`，仅 `libero_spatial` |
| Cosmos backbone | `nvidia/Cosmos-Reason2-2B`，`9ce19a195e423419c349abfc86fd07178b230561` |
| LIBERO 场景资产 | `RLinf/LIBERO-assets`，`3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f` |

LIBERO 专用模型附带 `libero_sim` 的 processor、statistics 和 embodiment mapping；它与后续 stack-cube SFT 使用的 `GR00T-N1.7-3B` base 是不同资产。模型 card 保留 NVIDIA 许可证声明；LIBERO 场景资产仓库没有声明许可证，manifest 如实记录。

## 复现

```bash
bash scripts/setup_assets.sh
python scripts/setup_n1_7_baseline.py
PATH="$PWD/.venvs/n1.7-libero/bin:$PATH" \
  .venvs/n1.7-libero/bin/python scripts/download_assets.py cosmos_reason2_2b

bash scripts/run_n1_7_baseline.sh train
bash scripts/run_n1_7_baseline.sh resume \
  runner.max_epochs=2 \
  "runner.resume_dir=$PWD/runs/n1.7-libero-baseline/train/n1_7_smoke/checkpoints/global_step_1"
```

Cosmos 下载沿用已接受许可证的 HF 授权。安装器校验 RLinf/N1.7 源码 commit、环境锁和归档 patch，通过独立 `tmp/RLinf-n1.7-baseline` worktree 复用 checkpoint 修复，按固定 revision 下载模型与场景资产；LIBERO 的 `assets` 链接到 NAS，配置写入独立环境的 `libero-config/`，由入口通过 `LIBERO_CONFIG_PATH` 传给 Ray worker。

环境锁以 `--no-deps` 安装，N1.7 editable 安装显式使用 `--ignore-requires-python`：官方 SFT metadata 要求 Python 3.10/Torch 2.7.1，本次验证的是 RLinf 的 Python 3.11/Torch 2.8 组合。`uv pip check` 报告 24 项不兼容，完整输出保存在 `runs/n1.7-libero-baseline/dependency-check.txt`。本次兼容性结论只覆盖该 RL smoke；不能据此替代官方 SFT 环境验证。FlashAttention GPU 前向与反向已单独实测。

入口使用 EGL、关闭 DISPLAY，以 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 加载本地模型。每 200 ms 记录整卡显存，每秒通过 `vmstat` 记录主机资源。训练进程成功退出后，复用 `verify_isaaclab_baseline.py --baseline n1_7_libero` 验证 TensorBoard 指标有限、gradient norm 非零、训练 step、DCP 模型/optimizer/scheduler metadata、Adam 实际 step 计数和视频解码，写入 `summary.json`。

## 2026-10-02 实测

第 2 步已完成。首次训练完成模型与本地 processor 加载、reset、rollout、一次 PPO update、更新后的权重同步和 eval，以及 step-1 checkpoint 保存。全新进程读取该 DCP checkpoint，再完成一次 update、同步、eval 与 step-2 保存；两次训练均以 exit code 0 退出，GPU 显存恢复到运行前水平。

| 检查 | 首次训练 | 新进程恢复 |
| --- | --- | --- |
| GPU 数 | 1 | 1 |
| checkpoint / TensorBoard / Adam step | 1 / 0 / 1 | 2 / 1 / 2 |
| actor total loss | 0.00049234 | 0.00012148 |
| critic value loss | 0.00246175 | 0.00060736 |
| approximate KL | 0.0 | 0.0 |
| gradient norm | 23.125 | 37.0 |
| episode length / env reward / success | 5 / 0 / 0 | 5 / 0 / 0 |
| GPU 峰值显存 | 23,476 MiB | 23,476 MiB |
| wall time | 390.2 s | 531.8 s |
| actor update wall time | 7.54 s | 7.57 s |
| train / eval MP4 | 各 7 帧，256 × 256，30 FPS | 各 7 帧，256 × 256，30 FPS |

全部 TensorBoard scalar 均为有限值，checkpoint 中的模型、Adam 和 scheduler metadata 存在，Adam 实际 step 从 1 延续到 2；恢复进程只记录 TensorBoard step 1。两次运行均没有 shape、device、CUDA OOM 或 worker 生命周期错误。每份 DCP 约 12.5 GiB。此次没有达到任务成功；5 步 episode 的零 env reward/success 只表示短工程验证的结果。

峰值 23,476 MiB 已接近 CUDA 报告的容量 24,080 MiB，未达到正式作业 90% 的预算目标。主机具有约 62.6 GiB RAM 和 2 GiB swap；保存阶段可用内存仅约 5.4–5.5 GiB。这组单卡 smoke 参数不能直接用作正式训练配置，N1.7 + IsaacLab 并发资源仍需另行验证。

## 资源限制与产物

单卡配置使用 CPU bucket 权重同步（128 MiB bucket），保存 DCP 训练状态，关闭额外 `full_weights.pt` 导出。DCP 保留模型、Adam、scheduler 与 RNG，恢复仍需固定的原始 processor 和 Cosmos 路径；它不是后续 SFT 阶段要求的独立 bundle。

首次采用默认 patch 同步器时，更新后的权重通过 GPU IPC 接收，出现 CUDA OOM。改用 CPU bucket 后，保存 checkpoint 时又触发 CPU 内存阈值；因此入口加入 `MALLOC_TRIM_THRESHOLD_=0`，让 glibc 及时归还释放的临时分配，保持 Ray 默认内存阈值。仅调整 allocator 仍不足以完成保存：FSDP 将 Adam 搬回 GPU 后复制模型 state dict，出现 CUDA OOM。基线因此复用 `0e601830` 修复，让已 offload 的 optimizer 在保存期间保留于 CPU。三次失败分别保存在 `patch-sync-oom-20261002/`、`checkpoint-cpu-oom-20261002/` 和 `checkpoint-cuda-oom-20261002/`，均位于基线产物根目录，不计入通过记录。

最终产物位于 NAS，通过项目 `runs/n1.7-libero-baseline/` 访问：

```text
train/、resume/
├── console.log、gpu.csv、memory.log
├── summary.json
├── tensorboard/
├── video/train/seed_0/0.mp4
├── video/eval/seed_0/0.mp4
└── n1_7_smoke/checkpoints/global_step_<1 或 2>/actor/dcp_checkpoint/
```

`asset-check.json` 和 `libero-asset-check.json` 保存本轮模型、Cosmos 和场景资产校验结果；`kernel-smoke.log` 保存 FlashAttention 验证结果。wall time 包含 NAS 权重加载与 checkpoint 写入，不等于纯 rollout/update 吞吐。5 步 rollout 的零 env reward/success 只用于工程链路验证，不能评价模型的任务能力。

`configs/patches/rlinf-n1_7-checkpoint.patch` 归档原始修复及其回归测试；安装器按原提交身份与日期重建 commit 并校验 hash，无需远端分支包含该 commit。本轮旧版本的 4 个保存显存测试失败，修复版本的 16 个 FSDP 测试全部通过；日志分别是 `checkpoint-before-20261002.log` 和 `checkpoint-after-20261002.log`。
