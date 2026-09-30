# GR00T N1.7 + IsaacLab 完整训练方案

本文确定 `embodied-template` 的实施范围与验证顺序：使用公开的 IsaacLab 仿真示范完成 NVIDIA GR00T N1.7 SFT，再将 SFT checkpoint 交给 RLinf，使用 PPO 在同一个 stack-cube 任务上继续训练。

## 已确定的范围

- 使用 RLinf 已有的 IsaacLab 任务 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0`，第一阶段不新增仿真任务。
- 使用公开数据集 [RLinf/IsaacLab-Stack-Cube-Data](https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data)，不采集实机数据。
- 使用 NVIDIA Isaac-GR00T 官方入口完成 N1.7 SFT，再由 RLinf 加载 SFT checkpoint 并运行 PPO。
- RTX 4090 只负责开发、调试与最小验证；完整 SFT 和 PPO 在后续申请的单机 H800 环境中运行。
- `embodied-template` 是可复用的工程模板，`stack_cube` 是第一个完整 reference experiment。

## 当前支持边界

RLinf 已经支持 GR00T N1.7，但当前维护的 N1.7 RL 示例是 LIBERO Spatial。RLinf 的 IsaacLab 示例目前使用 GR00T N1.5 或 OpenPI pi0.5，安装脚本也会拒绝 `gr00t_n1d7` 与 `isaaclab` 的组合。因此，N1.7 + IsaacLab 是本项目需要实现和验证的组合，而不是复制现有 YAML 即可运行的示例。

| 组合 | 在本项目中的用途 | 当前上游状态 |
| --- | --- | --- |
| GR00T N1.5 + IsaacLab | 验证 IsaacLab、Ray worker 与 PPO 环境链路 | RLinf 已提供文档、配置与 e2e 路径 |
| GR00T N1.7 + LIBERO Spatial | 验证 N1.7 模型、processor、rollout 与 actor 更新 | RLinf 当前维护的 N1.7 示例 |
| GR00T N1.7 + IsaacLab | 最终训练目标 | 需要完成集成与验证 |

这里的关键区别是：框架分别支持一个模型和一个环境，并不表示两者的任意组合都已经经过验证。

## 源码依赖

RLinf 和 Isaac-GR00T 都作为 Git submodule 固定到具体 commit，使 SFT 与 PPO 使用可追溯的源码版本。

| 依赖 | 仓库 | 固定版本 |
| --- | --- | --- |
| RLinf | `https://github.com/ZMC42/RLinf.git` | `61ba34e640035f2e4ac4ef3c6078f02d43a00c9a` |
| Isaac-GR00T | `https://github.com/NVIDIA/Isaac-GR00T.git` | `n1.7-release` tag，即 `23ace64f17aa5015259b8609d371eb61a357c776` |

RLinf 使用个人 fork，因为 N1.7 + IsaacLab 的通用集成需要修改框架代码并补充测试。Isaac-GR00T 暂时保持 NVIDIA 官方仓库，仅当确定官方 SFT 无法通过项目级配置完成时才建立 fork。`GR00T_PATH` 始终指向这份固定源码，RLinf 安装脚本不得在虚拟环境中重复克隆 Isaac-GR00T。

## 工程边界

`embodied-template` 负责项目级配置、训练入口、数据 contract、实验记录和 stack-cube 验证。能够复用于其他任务的 model/env adapter、安装逻辑、worker 行为、回归测试与 e2e 配置放在 RLinf fork 中。这个边界可以避免通过 monkey patch 隐藏框架改动，也能让后续任务复用相同接口。

```text
embodied-template/
├── configs/                  项目级 SFT 与 PPO 配置
├── docs/                     方案与运行说明
├── experiments/stack_cube/  第一个完整 reference experiment
├── scripts/                  初始化、下载和启动脚本
├── src/embodied_template/    项目级 adapter 与入口
├── tests/                    数据和项目 contract 测试
└── third_party/
    ├── Isaac-GR00T/          NVIDIA N1.7 固定源码
    └── RLinf/                ZMC42 RLinf fork 固定源码
```

## NAS 资产布局

代码保留在本地工作区，大文件统一存放在 `/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets`：

```text
/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets/
├── datasets/
│   └── isaaclab-stack-cube/
├── models/
│   ├── gr00t-n1.7-3b/
│   ├── cosmos-reason2-2b/
│   └── stack-cube-n1.7-sft/
├── simulators/
│   └── isaac-sim-5.1.0/
└── runs/
```

`scripts/setup_assets.sh` 在项目根目录创建 `datasets`、`models`、`isaac-sim` 和 `runs` 软链接。Isaac Sim cache、shader cache 和其他临时文件保留在本地磁盘，避免共享 NAS 的小文件延迟影响运行。

## 数据 contract

stack-cube 数据集公开且无需审批，包含 147 条 Franka 仿真示范、53,265 帧，总时长约 44 分钟，采用 LeRobot v2 格式。每一帧包含 256 x 256 的 front/wrist 双相机图像、8 维绝对 EEF state、7 维相对 EEF action，以及 stack-cube 的自然语言指令。

数据集 `meta/modality.json` 的字段与 N1.7 的 `libero_sim` modality 一致，包括 `image`、`wrist_image`、`x`、`y`、`z`、`roll`、`pitch`、`yaw` 和 `gripper`。因此，第一条验证路径是复用 `LIBERO_PANDA` modality 完成 SFT 和 checkpoint 交接。不过，字段相同不能直接证明行为兼容；进入 PPO 前仍须分别验证 action representation、normalization、gripper 约定、processor metadata 和 decode 后的 action shape。

## 实施顺序与完成条件

### 1. 验证 IsaacLab 链路

在 RTX 4090 上缩小 GR00T N1.5 + IsaacLab 配置。完成条件是成功执行一次环境 reset、一次 rollout、reward 收集和一次 actor update，期间没有 shape、device 或 worker 生命周期错误。

### 2. 验证 N1.7 链路

以 smoke-test 规模运行 RLinf 维护的 GR00T N1.7 + LIBERO 配置。完成条件是成功加载模型与 processor，并完成一次 rollout 和一次 actor update。

### 3. 完成 stack-cube N1.7 SFT

使用 `nvidia/GR00T-N1.7-3B` 作为 base checkpoint，通过 NVIDIA 的 `gr00t/experiment/launch_finetune.py` 训练。先在 4090 上验证数据读取、processor 和单步前向，再在 H800 上运行完整 SFT。

SFT 的完成条件是得到一个自包含 checkpoint：模型权重、processor、embodiment mapping、statistics 和 metadata 均可在离线状态下加载，并能在保留的 stack-cube 轨迹上完成 open-loop inference。

### 4. 在 RLinf 中集成 N1.7 + IsaacLab

需要在 RLinf fork 中完成以下通用改动：

- 允许并安装 `gr00t_n1d7` + `isaaclab` 组合；
- 增加 N1.7 IsaacLab 实验配置；
- 在模型边界验证 observation key 与 state dimension；
- 在环境边界验证 relative EEF 与 gripper 转换；
- 从 SFT checkpoint 加载 processor 与 embodiment metadata，避免隐含的 LIBERO task 假设；
- 增加聚焦的 unit test 和 embodied e2e smoke 配置；
- 补充经过验证的模型—环境组合文档。

### 5. 完成 4090 集成 smoke test

缩小环境数量、batch、episode 长度和训练步数，并按需启用 offload。该阶段只验证 wiring，不评价吞吐或收敛。完成条件是执行一个完整 PPO update、保存 checkpoint，并生成目标 IsaacLab 任务的 rollout 视频。

### 6. 在 H800 上正式训练

H800 数量确定后再设置 placement、global batch 和 micro batch。在单个 Ray node 上依次运行完整 SFT、open-loop evaluation、PPO、周期性 IsaacLab evaluation 和 checkpoint 保存。主要任务指标是 `env/success_once`，同时记录 loss、KL、value、显存与吞吐指标。

## 环境隔离

SFT 使用 Isaac-GR00T 官方验证的 Python 与依赖组合，PPO 使用 RLinf + IsaacLab 环境。在确认两边的 Python、Torch、FlashAttention 与 Isaac Sim 依赖完全一致之前，保持两个虚拟环境独立，使用数据集与 checkpoint 作为阶段接口。这样可以避免为了 RL runtime 修改官方 SFT 环境，也能让错误明确归属于某个阶段。

## 最终完成标准

1. 全新 checkout 可以初始化两个顶层 submodule 和 NAS 软链接。
2. 公开 stack-cube 数据可以通过 schema、样本解码和 modality contract 检查。
3. 官方 N1.7 SFT 能生成自包含的 stack-cube checkpoint。
4. RLinf 能加载该 checkpoint，并在 IsaacLab 中完成 PPO update。
5. RTX 4090 上的最小验证可以稳定复现。
6. H800 训练记录包含评测成功率、checkpoint、日志和完整源码版本。
