# Stack-cube N1.7 微型 SFT

本步骤仅验证官方 SFT 的数据读取、前向/反向、权重更新、保存和离线
open-loop inference。1 次 projector update 不代表模型已学会堆叠。

**第 4 步已完成（2026-10-02）**：微型 bundle 和全新禁网进程的 held-out
open-loop 验收均通过。RLinf + IsaacLab 集成属于后续第 5 步。

## 4090 运行记录（2026-10-02）

官方训练入口完成 1 次 update 并保存 step-1 checkpoint：loss=1.5655，
gradient norm=1.2690935，均为有限值。模型共 3,144,016,000 个参数，其中
327,356,544 个 projector 参数可训练（10.41%）。

前向/反向及 update 约 1.34 秒，Trainer wall time（含 step-1 保存）
97.42 秒；官方 run wall time（含模型初始化与最终保存）287.97 秒。
后续 bundle 复制、SHA256 和离线验收不计入该 wall time。NAS 写入主导本次
运行时间，这些数值不能作为正式 H800 训练吞吐估计。

PyTorch 峰值 allocated/reserved 为 14,785,916,928 / 16,536,043,520 bytes；
500 ms 采样的 GPU 使用峰值为 16,295 MiB。本次没有反向 OOM。
完整 bundle 为 9,543,147,495 bytes（约 8.9 GiB），其中模型权重
9,529,201,304 bytes。原始证据为 `training.log`、`training-resources.json`、
`gpu.csv` 和 checkpoint 的 `trainer_state.json`。

离线验收验证 decoder 的 `action_head.action_decoder.layer2.W` 有 5,250 个
元素变化，最大绝对变化约 1.00002e-4。所有保存文件 SHA256、processor flags、
train-only statistics、state/action dimension 和 horizon 检查均通过。
新进程在禁网状态下加载并推理成功，退出码为 0。

validation episode 6 共 292 帧，19 次 inference、每次 horizon=16、
4 denoising steps。总体 MSE=0.231604、MAE=0.183870；逐维指标如下。

| 相对 IK action 维度 | MSE | MAE |
| --- | ---: | ---: |
| x | 0.005548 | 0.062742 |
| y | 0.004960 | 0.061671 |
| z | 0.006296 | 0.067513 |
| rotvec x（roll） | 0 | 0 |
| rotvec y（pitch） | 0 | 0 |
| rotvec z（yaw） | 0.005253 | 0.063623 |
| gripper | 1.599171 | 1.031542 |

roll/pitch 的 train action 范围为常量 0，decode 后也为 0；零误差不能证明
模型掌握旋转控制。gripper 预测与专家开合仍明显不符，不能把本次产物当作
完成任务的策略。该图表示专家观测下的预测，没有执行 closed-loop rollout。

离线 model load + inference wall time 为 125.63 秒（不含先行 SHA256 和
权重变化检查）；首次 chunk inference 0.290 秒，其余约 0.053–0.055 秒。
推理峰值 allocated/reserved 为 6,387,751,424 / 6,484,393,984 bytes。
精确报告在 `runs/stack-cube/sft-micro/offline-verification.json`；图与原始
数值在 `runs/stack-cube/visualization/open-loop/micro-episode-000006.{png,npz}`。

## 入口与配置

```bash
bash scripts/run_stack_cube_sft_micro.sh
```

使用已有 `.venvs/sft-n1.7` 环境和第 3 步的
`runs/stack-cube/data/train`、`validation`。配置见
[`stack_cube_sft_micro.json`](../configs/stack_cube_sft_micro.json)：batch=1、
gradient accumulation=1、max steps=1、AdamW、learning rate=1e-4、无 warmup。
冻结 VLM、diffusion model 和 VL layer normalization，仅训练官方 action
projector（state/action encoder、action decoder 和 position embedding）。

固定的 `launch_finetune.py` 会强制设置 `use_relative_action=true`，CLI 也没有
本地 Cosmos 路径及 `use_percentiles=false` 参数。因此项目 wrapper 直接调用
该入口使用的 `gr00t.experiment.experiment.run`，完整复用官方 pipeline、
dataset、collator、Trainer 和 checkpoint callback；不修改上游，不做 monkey patch。

初始化目录位于 `tmp/stack-cube-sft-micro/initial`：权重引用固定的
`nvidia/GR00T-N1.7-3B`，processor 使用第 3 步的显式配置和 train-only statistics。
保存和重新加载后均断言 `use_percentiles=false`、`use_relative_action=false`、
`state_dropout_prob=0`。官方 DatasetFactory 的 metadata 写入位于本地 staging
目录，不改动第 3 步的派生数据或原始数据。

上游 `get_backbone_cls()` 要求路径包含 `nvidia/Cosmos-Reason2`。
`models/nvidia/Cosmos-Reason2-2B` 因此是固定 Cosmos snapshot 的相对软链接。
模型和 processor 保存该本地路径，bundle manifest 同时记录原始 snapshot
revision、文件校验信息和运行时 backbone 路径。

## Python 编译头文件

本机 SFT 使用系统 Python 3.10.12，但没有 `python3.10-dev`。Accelerate 初始化
Trainer 时导入 DeepSpeed，后者触发 Triton driver 编译；缺少 `Python.h` 会失败。
有相应系统头文件的机器无需补充。无 sudo 的本机采用本地解包：

```bash
mkdir -p tmp/sft-python-headers
cd tmp/sft-python-headers
apt download libpython3.10-dev=3.10.12-1~22.04.18
sha256sum libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb
dpkg-deb -x libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb .
cd ../..
```

该 Ubuntu Jammy 包 SHA256 为
`c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9`。
runner 通过 CPATH 引用本地头文件。Python、Torch 和 frozen uv.lock 未改变。
`runs/stack-cube/sft-micro/gpu-kernel-check.log` 记录实际 DeepSpeed/Triton
初始化和 FlashAttention BF16 forward/backward 检查。

## 交接与离线验证

交付位于 `models/stack-cube-n1.7-sft/`：

```text
checkpoint/       完整模型权重、config、processor 文件、trainer_state
processor/        官方保存的 processor_config、statistics、embodiment_id
experiment_cfg/   官方配置、项目配置、contract、split 和环境记录
LICENSE           原始 GR00T 模型许可证
MANIFEST.json     源码 commits、HF revisions、backbone 路径及文件 SHA256
```

必须与 manifest 记录的固定 Cosmos snapshot 一起交接。processor 配置和
statistics 中包含 `libero_sim`；保留原始 base 的 embodiment mapping。
这是 `save_only_model=true` 的微型交接 checkpoint，不包含 optimizer，不能用来
恢复 SFT optimizer 状态。

runner 训练结束后用 `unshare --user --map-root-user --net` 启动全新 Python
进程，设置 `HF_HUB_OFFLINE=1`。验收脚本断言网络 namespace 与宿主不同，且
只有 loopback interface；模型加载和推理均在此禁网进程完成。

验收检查 1 次 update 的有限 loss、非零有限 gradient norm、实际 decoder
权重变化、train-only statistics、state dim=8、decoded action dim=7 和
action horizon=16。使用 validation 的第一条 episode，沿完整专家轨迹每 16 帧
调用一次官方 `Gr00tPolicy`，保存 7 维预测曲线、原始数值和逐维 MSE/MAE。
test split 不参与此检查。

已有 bundle 的离线验收可独立执行：

```bash
source scripts/project_env.sh
export PYTHONPATH="$PWD/src:$GR00T_PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export SFT_HOST_NETWORK_NAMESPACE="$(readlink /proc/self/ns/net)"
unshare --user --map-root-user --net \
  .venvs/sft-n1.7/bin/python scripts/verify_stack_cube_sft_micro.py
```

完整 runner 用于首次生成；再次生成前应归档原有
`runs/stack-cube/sft-micro/training` 和 `models/stack-cube-n1.7-sft`，避免官方
Trainer 自动恢复已有 checkpoint 或覆盖交接产物。

结果日志在 `runs/stack-cube/sft-micro/`，逐维图和 NPZ 在
`runs/stack-cube/visualization/open-loop/`。open-loop 指标不代表 closed-loop
成功率；下一步仍是 RLinf + IsaacLab 集成和真实 rollout。
