# GR00T N1.7 + IsaacLab 完整训练方案

本文确定 `embodied-template` 的实施范围、验证顺序和 go/no-go 条件：使用公开的 IsaacLab 仿真示范完成 NVIDIA GR00T N1.7 SFT，再将 SFT 产物交给 RLinf，使用 PPO 在同一个 stack-cube 任务上继续训练。

## 可行性结论与执行状态

该路线技术上可行，但当前属于**有条件可行**，尚不是可直接启动完整 H800 训练的已验证配置。数据 schema、官方 SFT 入口、RLinf 的 N1.7 模型支持和 checkpoint processor 保存机制均已存在；尚未闭合的风险集中在：

- N1.7 + IsaacLab 已完成微型 SFT bundle 交接、禁网加载、闭环和 WebRTC；PPO update/保存/恢复仍属第 6 步；
- SFT 与 RLinf rollout 间的 embodiment、principal axis-angle、state/action 和 normalization contract 已实测，训练状态恢复仍待验证；
- N1.7 + IsaacLab 是本项目 fork 验证的组合，已加入 e2e 配置，CI runner 仍需配置固定资产；
- 单张 RTX 4090 已完成 N1.5 + IsaacLab 和 N1.7 + LIBERO 的 actor update；N1.7 + IsaacLab eval/WebRTC 峰值 12,532 / 13,158 MiB，actor update 显存尚未测量；
- stack-cube 数据、processor、model/env adapter 和项目级配置已闭合；正式 SFT/PPO 的学习效果与资源 gate 仍需完成。

在“微型 SFT checkpoint → RLinf 离线加载 → IsaacLab rollout → 一次 PPO update → 保存并恢复”完整通过前，不启动完整 SFT 或正式 PPO。

## 已确定的范围

- 使用 RLinf 已有的 IsaacLab 任务 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0`，第一阶段不新增仿真任务。
- 使用公开数据集 [RLinf/IsaacLab-Stack-Cube-Data](https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data)，不采集实机数据。
- 使用 NVIDIA Isaac-GR00T 官方入口完成 N1.7 SFT，再由 RLinf 加载 SFT 产物并运行 PPO。
- RTX 4090 是无桌面的远程 headless server，负责开发、数据与 processor 验证、单步前向和最小 rollout；开发阶段必须能通过 Isaac Sim WebRTC/Livestream 实时查看单环境行为。actor update 是资源允许时的目标，不作为单卡必须满足的前提。
- 完整 SFT 和 PPO 在后续申请的单机 H800 环境中运行；正式申请前必须通过资源预估和最小端到端验证。
- 数据、SFT 和 PPO 每个阶段都必须产生人可观察的回放或图表；不能只用 loss 和成功率判断机器人行为。
- `embodied-template` 是可复用的工程模板，`stack_cube` 是第一个完整 reference experiment。

## 当前支持边界

RLinf 上游维护的 N1.7 RL 示例主要是 LIBERO Spatial，IsaacLab 示例主要使用 GR00T N1.5 或 OpenPI pi0.5。本项目 fork 已补齐 `gr00t_n1d7` + `isaaclab` 的安装、模型转换、实验和 e2e 配置，并实测微型 SFT checkpoint 的禁网加载、Ray eval 和 WebRTC 闭环。PPO update 仍需按第 6 步单独验收。

| 组合 | 在本项目中的用途 | 当前上游状态 |
| --- | --- | --- |
| GR00T N1.5 + IsaacLab | 验证 IsaacLab、Ray worker 与 PPO 环境链路 | RLinf 已提供文档、配置与 e2e 路径 |
| GR00T N1.7 + LIBERO Spatial | 验证 N1.7 模型、processor、rollout 与 actor 更新 | RLinf 当前维护的 N1.7 示例 |
| GR00T N1.7 + IsaacLab | 最终训练目标 | 项目 fork 已完成第 5 步，PPO smoke 待验证 |

这里的关键区别是：框架分别支持一个模型和一个环境，并不表示两者的任意组合都已经经过验证。

## 源码与资产版本锁定

所有影响数据解释、仿真动力学、模型加载和训练行为的源码都必须固定到 commit。下表同时是实施要求；状态为“待纳入”的依赖在完成前不满足可复现性标准。

| 依赖 | 仓库 | 固定版本 | 状态 |
| --- | --- | --- | --- |
| RLinf | `https://github.com/ZMC42/RLinf.git` | `0bf6fd743bb3d652fd45784d35b859d6f1250345` | 已固定本地集成 commit，含可复现归档 patch |
| Isaac-GR00T | `https://github.com/NVIDIA/Isaac-GR00T.git` | `n1.7-release`，即 `23ace64f17aa5015259b8609d371eb61a357c776` | 已作为 submodule 固定 |
| IsaacLab | `https://github.com/RLinf/IsaacLab.git` | `4246b6b4f4a3e74ee20e002ed7536b1c788d39f4` | 已作为 submodule 固定 |

RLinf 使用个人 fork，因为 N1.7 + IsaacLab 的通用集成需要修改框架代码并补充测试。Isaac-GR00T 暂时保持 NVIDIA 官方仓库，仅当官方 SFT 无法通过项目级 wrapper、配置或本地 HF cache 完成时才建立 fork。

`GR00T_PATH` 和 `ISAAC_LAB_PATH` 必须分别指向上述固定源码。RLinf 安装脚本不得在虚拟环境中重复克隆或跟随远端默认分支；安装开始前要校验三个源码目录的 commit。

Hugging Face 资产同样使用不可变 revision 下载：

| 资产 | 固定 revision | 访问要求 |
| --- | --- | --- |
| `RLinf/IsaacLab-Stack-Cube-Data` | `2e27f04b2e31f293fd020164917e4cce09e108d6` | 公开 |
| `nvidia/GR00T-N1.7-3B` | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` | 公开，仍须保留许可证文件 |
| `nvidia/Cosmos-Reason2-2B` | `9ce19a195e423419c349abfc86fd07178b230561` | Hugging Face 自动 gated；须提前接受 NVIDIA 许可证并准备 token |

下载脚本必须显式传入 revision，并在运行 manifest 中记录 revision、文件校验结果和许可证版本。未经 revision 固定的 HF `main` 不能作为可复现实验输入。

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
    ├── IsaacLab/             RLinf IsaacLab fork 固定源码，计划新增
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

`scripts/setup_assets.sh` 在项目根目录创建 `datasets`、`models`、`isaac-sim` 和 `runs` 软链接。Isaac Sim cache、shader cache、HF cache 和训练临时文件保留在本地 NVMe，避免共享 NAS 的小文件延迟影响运行。正式训练前对数据视频（固定 revision 实测为 MPEG-4 Part 2）随机读取做吞吐测试；若 NAS 吞吐不足，将固定 revision 的数据集 staging 到训练节点本地盘，并记录源 revision。

## 数据与模型 contract

### 数据 contract

stack-cube 数据集公开且无需审批，包含 147 条 Franka 仿真示范、53,265 帧，总时长约 44 分钟，采用 LeRobot v2 格式。每一帧包含 256 × 256 的 front/wrist 双相机图像、8 维绝对 EEF state（原始旋转为 Euler xyz，SFT 派生数据显式转为 principal axis-angle）、7 维相对 EEF action，以及 stack-cube 的自然语言指令。

数据集 `meta/modality.json` 的字段与 N1.7 的 `libero_sim` modality 一致，包括 `image`、`wrist_image`、`x`、`y`、`z`、`roll`、`pitch`、`yaw` 和 `gripper`。字段相同不能直接证明行为兼容，必须验证：

- SFT 派生数据与 runtime state 精确为 `[eef_xyz(3), principal_eef_axis_angle(3), left_finger_pos, -right_finger_pos]`；原始示范的 `roll/pitch/yaw` 经数值诊断为 Euler xyz，必须显式转换并记录，不能直接作为 axis-angle 使用；
- action 精确为 IsaacLab relative IK 接受的 7 维命令，包含旋转与 gripper 的方向、范围和单位；
- front/wrist 相机顺序、分辨率、颜色通道、相机位姿和图像 dtype 与数据采集时一致；
- gripper state 的双指维度、action 的单维开合约定，以及 `[-1, 1]` 符号与 IsaacLab action manager 一致；
- processor normalization 后再 decode 的 action 与原始 action 数值 round-trip 一致；
- 实际 MPEG-4 Part 2 视频能在 SFT 环境中稳定随机解码，并记录 TorchCodec 使用的系统 FFmpeg shared libraries；原始 metadata 的 AV1 声明已确认不准确。

已验证的数据处理、固定 split、train-only statistics 与复现入口见 [`STACK_CUBE_DATA_CONTRACT.md`](STACK_CUBE_DATA_CONTRACT.md)。公开数据没有采集时的相机标定；本项目固定 runtime 相机位姿并记录此验证边界，后续通过 closed-loop gate 检查域匹配。

数据在训练前按 episode ID 固定划分，禁止按 frame 随机拆分：默认 117 条 train、15 条 validation、15 条 test。具体 episode ID、随机种子和划分算法保存到实验配置。test 集在最终模型选择前不参与调参；若沿用数据集全局 statistics，必须在报告中标注，优先生成仅基于 train split 的 statistics。

### Embodiment 与 processor contract

第一条验证路径复用官方 `LIBERO_PANDA` modality。N1.7 中 `LIBERO_PANDA` 的实际 tag value 是 `libero_sim`，因此 contract 固定为：

```yaml
actor:
  model:
    embodiment_tag: libero_sim
    obs_converter_type: isaaclab_stack_cube
```

- 官方 SFT CLI 使用 `--embodiment-tag LIBERO_PANDA` 或等价的 `libero_sim`；
- SFT 产物的 `processor_config.json`、`statistics.json` 和 `embodiment_id.json` 必须包含 `libero_sim`；
- RLinf actor 与 rollout 均使用 `libero_sim`，不得沿用 N1.5 IsaacLab 示例中的 `isaaclab_franka`；
- `obs_converter_type` 独立使用 `isaaclab_stack_cube`，负责把 IsaacLab observation 转为 checkpoint 的 `libero_sim` schema，并将 wrapper 的 axis-angle 统一到与 SFT 派生数据一致的 principal branch；
- 模型初始化后必须断言 observation keys、state dim = 8、decoded action dim = 7 和 action horizon 一致。

如上述复用路径无法通过数值 contract，再切换到 custom modality；不得通过静默重命名、hard-coded statistics 或 monkey patch 绕过 processor metadata。

### SFT 交接产物

官方训练入口会保存模型 checkpoint 和 processor 文件，但 processor 仍引用 canonical Cosmos backbone/processor。交付单位因此定义为**可离线加载的 bundle**，而不是假设单个 checkpoint 目录天然包含所有依赖：

```text
stack-cube-n1.7-sft/
├── checkpoint/              模型权重和 config
├── processor/ 或根目录文件  processor_config、statistics、embodiment_id
├── experiment_cfg/          完整 SFT 配置和 metadata
└── MANIFEST.json            源码 commit、HF revisions、数据 split、backbone 路径与校验信息
```

bundle 必须与固定 revision 的本地 `Cosmos-Reason2-2B` snapshot 一起，在 `HF_HUB_OFFLINE=1` 且禁止网络的进程中成功加载。RLinf 的 `backbone_model_path` 显式指向该 snapshot；不得依靠某台机器上未记录的 HF cache。

## 可视化、回放与观察路径

具身智能里的“可视化”有不同含义。为避免把数据回放误认为模型推理，所有产物按以下四层区分：

| 层级 | 看到的内容 | 能回答的问题 | 不能证明什么 |
| --- | --- | --- | --- |
| 数据集回放 | 专家示范的 front/wrist 图像与记录动作 | 数据是否正确、相机和任务是否匹配 | 模型是否学会任务 |
| SFT open-loop | 固定示范状态下，预测动作与专家动作的曲线 | processor 和动作预测是否基本正确 | 机器人执行预测后能否恢复误差 |
| IsaacLab closed-loop | 模型控制机器人后产生的新 rollout 视频 | 真正的推理行为、失败模式和成功过程 | PPO 是否稳定改进，需要多 seed 指标 |
| 训练面板 | success、return、loss、KL、value、显存和吞吐 | 训练是否稳定、是否在学习 | 单条曲线不能替代视频和固定-seed 评测 |

### 1. 回放公开示范数据

下载数据后，可以先直接查看两路相机视频。这一步不需要加载模型：

```bash
ffplay datasets/isaaclab-stack-cube/videos/chunk-000/observation.images.front/episode_000000.mp4
ffplay datasets/isaaclab-stack-cube/videos/chunk-000/observation.images.wrist/episode_000000.mp4
```

项目需要提供 `scripts/replay_stack_cube_dataset.py`，把同一 episode 的 front 与 wrist 画面并排，并叠加 task description、frame、EEF state 和 7 维 action。输出保存到：

```text
runs/stack-cube/visualization/dataset/
```

至少固定回放一个成功 episode、一个最长 episode，以及包含 gripper 开合切换的 episode。回放脚本同时用于人工检查相机顺序、时间同步、画面方向和 action 是否晚一帧或早一帧。

### 2. 查看 SFT 的 open-loop 动作预测

Isaac-GR00T 的 `open_loop_eval.py` 会计算 MSE/MAE，并输出专家动作与模型预测动作的曲线图。例如在 SFT 虚拟环境中运行派生 validation 数据的第一条 held-out 轨迹（loader index 0，即 episode 6）：

```bash
python third_party/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
  --dataset-path runs/stack-cube/data/validation \
  --embodiment-tag LIBERO_PANDA \
  --model-path models/stack-cube-n1.7-sft/checkpoint \
  --traj-ids 0 \
  --steps 292 \
  --action-horizon 16 \
  --save-plot-path runs/stack-cube/visualization/open-loop/traj-6.jpeg
```

这里显示的是“如果仍然处在专家轨迹的观测上，模型会预测什么”，不是机器人执行模型动作后的录像。必须分别观察 xyz、axis-angle 和 gripper，不能只看所有维度平均后的 MSE。

### 3. 实时查看并录制真正的闭环推理

真正的机器人推理来自 IsaacLab closed-loop evaluation。4090 本机没有桌面，因此开发阶段使用 Isaac Sim WebRTC/Livestream 实时传输 viewport；正式训练仍保持 headless，并通过固定-seed MP4 留下可复现记录。

#### 远程 4090 实时画面

固定的 IsaacLab 版本支持两种 WebRTC 模式：

- `LIVESTREAM=2`：私网或本地网络，优先用于 VPN、Tailscale、WireGuard 或同一内网；
- `LIVESTREAM=1`：公网连接，必须同时设置客户端可达的 `PUBLIC_IP`，并按 Isaac Sim 5.1 文档配置防火墙和安全组。

开发时优先使用私网/VPN：

```bash
LIVESTREAM=2 ENABLE_CAMERAS=1 \
  python scripts/run_interactive_eval.py \
  --checkpoint models/stack-cube-n1.7-sft/checkpoint \
  --seed 0
```

如果只能通过公网连接：

```bash
LIVESTREAM=1 \
PUBLIC_IP=<client-reachable-server-ip> \
ENABLE_CAMERAS=1 \
  python scripts/run_interactive_eval.py \
  --checkpoint models/stack-cube-n1.7-sft/checkpoint \
  --seed 0
```

`scripts/run_interactive_eval.py` 是必须实现的项目入口。它只启动一个 Isaac Sim 实例和一个环境，加载指定 checkpoint，并允许从 NVIDIA Isaac Sim WebRTC Streaming Client 实时查看 viewport。至少支持：暂停/继续、单步、reset、固定 seed、切换 table/wrist camera，以及在终端同步输出当前 action、reward、success 和推理延迟。

当前 RLinf 的 `AppLauncher(headless=True, enable_cameras=True)` 会继续保持服务端 headless，但能读取 `LIVESTREAM` 和 `PUBLIC_IP`。接入时仍需验证这些环境变量能传入实际创建 Isaac Sim 的 Ray/subprocess；interactive 模式只允许一个 env worker，避免多个 Isaac Sim 实例争用同一个 livestream 服务。每个 Isaac Sim 实例同时只接受一个 streaming client。

WebRTC 是 UDP/实时媒体链路，普通 SSH TCP port forwarding 不能等价替代完整的 WebRTC 网络连通。优先通过受控私网或 VPN 使用 `LIVESTREAM=2`；如使用公网模式，只开放 Isaac Sim 5.1 官方文档要求的端口，并限制来源地址。

实时查看表示“画面随仿真执行持续更新”，不保证 GR00T 推理达到数据集的 20 Hz。interactive runner 要分别显示 policy inference latency、simulation step time 和实际 FPS；模型推理较慢时允许仿真等待 action，不能通过丢帧隐藏延迟。

#### 固定-seed MP4 回放

实时画面便于调试，但不能替代实验记录。目标 PPO 配置还必须启用 MP4 录制，eval 配置至少包含：

```yaml
runner:
  only_eval: true
  logger:
    logger_backends: [tensorboard]

env:
  eval:
    total_num_envs: 1
    use_fixed_reset_state_ids: true
    video_cfg:
      save_video: true
      info_on_video: true
      fps: 20
      extra_info_on_video:
        - episode.success_once
        - episode.episode_len
      video_base_dir: ${runner.logger.log_path}/video/eval
```

RLinf 会把录像写到：

```text
<log_path>/video/eval/seed_<seed>/<video_index>.mp4
```

默认 recorder 使用 `main_images`，在本任务中即桌面/front 视角，并可在画面上叠加 reward、termination、success 和 episode length。若需要同时观察腕部相机，应扩展 recorder 生成 front/wrist 并排视频，并为双相机帧同步增加测试。

每次验证至少保留：一个成功 episode、一个失败 episode、一个 action 饱和或 gripper 异常 episode。文件名或同目录 manifest 必须记录阶段、checkpoint、seed、是否成功和源码版本。

实时 WebRTC 和 MP4 共用 headless 渲染能力，但用途不同：WebRTC 用于开发时立即观察，MP4 用于复现实验和比较 checkpoint。正式并行训练不启用 WebRTC；只有单环境 interactive/eval 作业允许启动 livestream。

### 4. 观察 PPO 训练过程

RLinf 配置使用 `logger_backends: [tensorboard]`，事件文件位于 `<log_path>/tensorboard/`。在保存日志的机器上启动：

```bash
tensorboard --logdir runs/stack-cube --host 127.0.0.1 --port 6006
```

如果日志就在本机，直接打开 `http://127.0.0.1:6006`。如果命令运行在远程 H800 机器上，再从本地建立 SSH tunnel，不直接把 TensorBoard 端口暴露到公网：

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<h800-host>
```

新手优先关注：

- `env/success_once`：固定评测集上的成功率是否随 checkpoint 提升；
- return/reward：必须结合视频确认奖励是否对应真正堆叠成功；
- KL：突然增大通常表示 PPO 更新离 SFT policy 太远；
- policy/value loss：用于发现发散，数值越小不一定代表机器人越好；
- gradient norm：尖峰或非有限值表示更新不稳定；
- GPU memory、rollout/update throughput：用于判断配置是否可持续运行。

训练环境默认不持续录像，因为大量并行环境的视频编码会显著降低吞吐并占用内存。只在固定间隔的 eval 中启用视频，使用相同 seeds 录制 step-0、SFT、PPO 中间 checkpoint 和最终 checkpoint，形成可直接比较的回放集合。

### 5. 可视化完成条件

进入正式训练前，以下产物必须可以由新 checkout 复现：

1. 一条双相机专家示范回放；
2. 一张 held-out 轨迹的逐动作维度 open-loop 对比图；
3. 从远程 4090 成功连接一次 WebRTC，并实时看到单环境 SFT closed-loop rollout；
4. 一个 SFT policy 的 IsaacLab 闭环 MP4；
5. 一个 PPO smoke checkpoint 的闭环 MP4；
6. 一个能打开并显示 success、KL、value 和资源指标的 TensorBoard logdir；
7. 同一固定 seed 下的 SFT 与 PPO 并排或成对回放。

## 实施顺序与 go/no-go 条件

### 0. 冻结依赖与运行前检查

先完成 IsaacLab 固定源码接入、HF revision 下载、Cosmos 许可证授权和两套虚拟环境的 lock/manifest。验证 Isaac Sim 5.1、IsaacLab、Torch、CUDA、FlashAttention、Ray 和 driver 的实际版本。

项目入口见 [`docs/PREFLIGHT.md`](PREFLIGHT.md)：`dependencies.lock.json` 是版本
唯一来源，`download_assets.py` 只按固定 revision 下载，`preflight.py` 负责离线、
运行时和网络检查，`isaaclab_smoke.py` 负责目标 task 枚举与 headless reset。

完成条件：离线检查脚本能解析全部源码 commit 和资产 revision；IsaacLab 能列出目标 task ID，并成功完成一次 headless reset；网络预检确认 4090 到开发者客户端具备 WebRTC 私网/VPN 或受控公网路径。

### 1. 验证 IsaacLab 基线链路

**已完成（2026-10-01）**：单张 RTX 4090 已运行 GR00T N1.5 + IsaacLab，完成 reset、rollout、reward 收集、PPO update、权重同步与 step-1 checkpoint 保存；新进程恢复后继续 update 并保存 step 2。两次运行正常退出，所有记录指标均为有限值。采样峰值显存分别为 23,826 / 23,013 MiB，完整 wall time 为 348.6 / 465.1 s。配置、环境锁、复现命令和产物说明见 [`ISAACLAB_BASELINE.md`](ISAACLAB_BASELINE.md)。本次为短 rollout 工程验证，reward 与 success 为零，不代表任务学习效果。

缩小 GR00T N1.5 + IsaacLab 配置，验证环境 reset、rollout、reward 收集和 actor update。若单张 4090 无法容纳 actor update，先记录峰值显存和失败位置，再在最小多卡/H800 配置完成该基线，不把 OOM 误判为接口失败。

完成条件：至少一次完整 update，无 shape、device、NaN 或 worker 生命周期错误，并能保存和恢复 checkpoint。

### 2. 验证 N1.7 基线链路

**已完成（2026-10-02）**：单张 RTX 4090 已运行 GR00T N1.7 + LIBERO Spatial，完成本地模型与 processor 加载、rollout、PPO update、更新后的权重同步、视频录制和 step-1 checkpoint 保存；新进程恢复后再次 update、同步并保存 step 2。两次进程正常退出，所有记录指标有限，Adam 实际 step 从 1 延续到 2。采样峰值显存均为 23,476 MiB，wall time 分别为 390.2 / 531.8 s。基线使用独立 RLinf worktree `0e601830` 修复 checkpoint 保存时的 optimizer GPU staging，并采用 CPU bucket 同步；归档 patch、环境锁、复现命令和资源限制见 [`N1_7_BASELINE.md`](N1_7_BASELINE.md)。本次短 rollout 的 env reward/success 为零，且显存未满足正式作业 90% 的预算目标。

以 smoke-test 规模运行 RLinf 维护的 GR00T N1.7 + LIBERO 配置。

完成条件：成功加载模型与 processor，完成 rollout、actor update、权重同步、保存和恢复。记录基线所需 GPU 数、峰值显存和 wall time。

### 3. 验证 stack-cube 数据 contract

**自动化数据与数值检查已完成（2026-10-02）**：官方 SFT 环境中 10 项真实资产测试通过，覆盖 147 episodes / 53,265 帧、样本随机 RGB 解码、state/action encode/decode、horizon padding、边界 frame 与 gripper 切换；真实 IsaacLab reset/action-manager 检查通过。已固定 117 / 15 / 15 的 episode split，生成 train-only statistics 和 episode 0 / 最长 episode 7 的双相机专家回放。实测修正了原始 Euler xyz 旋转、视频 codec 声明和数值 dtype 的差异；SFT 使用显式派生的 principal axis-angle 数据，train action round-trip 最大误差约 6×10⁻⁹。公开数据缺少采集标定及 IK scale provenance，无法证明这些采集参数完全一致；本轮固定 runtime 相机及动作配置并记录该边界，后续 closed-loop gate 仍须验证域匹配。复现命令、环境与 FFmpeg 记录、数值诊断及产物见 [`STACK_CUBE_DATA_CONTRACT.md`](STACK_CUBE_DATA_CONTRACT.md)。

在不启动完整训练的情况下运行 schema、样本解码、processor encode/decode 和 observation/action 数值测试，并生成固定的 episode split。

完成条件：上述数据与模型 contract 全部变成自动化测试；至少抽取 episode 开头、中间和结尾样本，验证 horizon padding、边界 frame 和 gripper 切换；生成一条 front/wrist 双相机专家示范回放。

### 4. 生成微型 N1.7 SFT checkpoint

**已完成（2026-10-02）**：单张 RTX 4090 完成官方 N1.7 的 1 次 projector-only SFT update，loss=1.5655、gradient norm=1.2691，采样峰值显存 16,295 MiB。已生成 `models/stack-cube-n1.7-sft/` 离线 bundle；全新进程在 `HF_HUB_OFFLINE=1` 和独立禁网 namespace 中完成加载、train-only statistics 检查、实际权重变化检查，以及 held-out episode 6 的全部 292 帧 open-loop 推理，保存 7 维预测—专家对比图和 NPZ。该微型产物仅验收工程链路，gripper 误差仍明显，不代表学会堆叠。入口、官方 CLI 限制、本地 Cosmos 路径、环境修复、资源和逐维指标见 [`STACK_CUBE_SFT_MICRO.md`](STACK_CUBE_SFT_MICRO.md)。第 5 步已进入 RLinf + IsaacLab 的集成验证，当前证据见 [`STACK_CUBE_N1_7_INTEGRATION.md`](STACK_CUBE_N1_7_INTEGRATION.md)。

使用 `nvidia/GR00T-N1.7-3B` 和官方训练 pipeline，仅运行足以产生 checkpoint 的 1–10 个 update。固定的 `gr00t/experiment/launch_finetune.py` 无法表达完整 stack-cube processor contract，项目 wrapper 因此直接调用同一官方 `experiment.run()`，显式保留 normalization 和本地 backbone 配置。4090 先验证数据读取、processor、单步前向/反向和保存；如反向 OOM，可在 H800 上完成该微型产物，但仍不得直接进入完整 SFT。

完成条件：生成符合 bundle contract 的产物，并在全新进程、离线模式下加载，完成 held-out 样本的 open-loop inference，同时保存逐动作维度的预测—专家对比图。

### 5. 集成 N1.7 + IsaacLab 并交接微型 checkpoint

**已完成（2026-10-02）**：RLinf fork `0bf6fd74` 已增加组合安装、N1.7 converter、SFT processor/backbone 加载、实验与 e2e 配置。独立 PPO 环境完成禁网 `(1, 16, 7)` 推理及实际 SFT decoder 权重核对；Ray rollout/env workers 完成 32 步闭环、6 个有限 eval 指标和 34 帧 MP4。macOS 客户端通过 Tailscale 确认看到了最终 SFT closed-loop 的实时方块相对位置变化，50 步作业保存 51 帧双相机 MP4；修复了 Kit timeline pause 导致 Fabric 画面静止的问题。Ray eval/WebRTC 采样峰值显存分别 12,532 / 13,158 MiB，均正常退出且没有 OOM。67 项模型测试通过、1 项跳过；reward/success=0，仅代表工程 gate。完整版本、资源、失败归档、复现与证据见 [`STACK_CUBE_N1_7_INTEGRATION.md`](STACK_CUBE_N1_7_INTEGRATION.md)。第 6 步 PPO update、权重同步、训练 checkpoint 保存与恢复尚未启动。

在 RLinf fork 中完成以下通用改动：

- 允许并安装 `gr00t_n1d7` + `isaaclab` 组合；
- 安装脚本复用固定的 `GR00T_PATH` 和 `ISAAC_LAB_PATH`，不得重复 clone；
- 增加 N1.7 IsaacLab 实验配置；
- 从 SFT bundle 加载 processor、statistics、embodiment mapping 和本地 Cosmos backbone；
- 在模型边界验证 observation key、principal axis-angle branch（与第 3 步派生数据一致）和 state dimension；
- 在环境边界验证 relative EEF、action scale 和 gripper 转换；
- 增加聚焦的 unit test 和 embodied e2e smoke 配置；
- 补充经过验证的模型—环境—IsaacLab commit 组合文档。

完成条件：RLinf 使用微型 SFT bundle 在目标 IsaacLab task 上完成 reset、至少一个 action chunk、reward 收集和无网络加载；远程客户端能实时看到该 closed-loop rollout，并产出可播放的 MP4。

### 6. 完成集成 smoke test

缩小环境数量、batch、episode 长度、denoising steps 和训练步数，并按需启用 actor/rollout offload。

4090 的强制完成条件是模型加载、环境 reset、rollout 和视频；一次完整 PPO update 是资源允许时的目标。如果 4090 OOM，必须保存显存测量和最小复现，并在已记录的最小多卡/H800 配置完成以下工程 gate：

- 一次完整 PPO update；
- actor/rollout 权重同步；
- checkpoint 保存和恢复后输出 shape、processor 和 step 计数一致；
- 生成目标任务 rollout 视频；
- loss、KL、value、gradient norm 和 action 均为有限值。

### 7. 完成正式 N1.7 SFT

仅在微型 checkpoint 已成功交给 RLinf 后，在 H800 上运行完整 SFT。完整运行前固定 GPU 数、global batch、micro batch、gradient accumulation、max steps、评测间隔、存储预算和预计 wall time。

完成条件：

- validation open-loop MSE/MAE 优于训练开始时保存的 step-0 初始化基线；如原始 base 能直接构造 `libero_sim` processor，则同时报告原始 base，否则不得为得到基线而伪造 embodiment metadata；
- 最终候选 checkpoint 能作为离线 bundle 加载；
- 保存训练曲线、峰值显存、吞吐、wall time 和全部版本信息；
- test split 只用于最终候选模型的一次报告，不参与 checkpoint 选择。

### 8. 执行 SFT-only 闭环评测

在 PPO 前先对 SFT policy 做 IsaacLab 闭环评测，避免用 PPO 掩盖 observation/action contract 错误。使用固定 reset seeds，至少评测 50 个 episode，同时保存成功/失败视频、action 分布，以及与 step-0 使用相同 seed 的对比回放。

进入 PPO 的最低 gate：无 contract/runtime 错误，action 未长期饱和，且 `success_once` 非零。若为零，先排查 SFT、相机域差异和动作转换，不直接增加 PPO 预算。

### 9. 在 H800 上正式 PPO

根据 smoke test 的实测显存设置 placement、global batch 和 micro batch。在单个 Ray node 上运行 PPO、周期性 IsaacLab evaluation 和 checkpoint 保存。GPU 数和 placement 必须在正式运行前冻结，不留到训练启动后临时决定。

固定 task commit 中，Rewarded 环境的唯一正奖励是 `cubes_stacked`，因此 `env/success_once` 可作为主任务指标；同时记录 `success_at_end`、return、loss、KL、value、gradient norm、显存和吞吐。

每次正式评测使用相同的一组至少 100 个固定 seeds，并报告成功率、episode 数和 95% Wilson 置信区间。工程完成与学习效果分开判定：

- **工程完成**：训练可持续运行、定期评测、保存、恢复且无非有限值；
- **学习成功**：PPO 最终固定-seed 成功率相对 SFT baseline 至少提升 5 个百分点。若未达到，仍保留完整负结果，不得仅以“完成 update”宣称训练有效。

## 环境隔离与资源约束

所有虚拟环境统一放在项目根目录的 `.venvs/` 下。已验证基线使用
`isaaclab-n1.5/` 和 `n1.7-libero/`；后续官方 SFT 和 IsaacLab PPO 基础环境
分别使用 `sft-n1.7/` 和 `ppo-isaaclab/`，按阶段需要创建。

SFT 使用 Isaac-GR00T 官方验证的 Python 与依赖组合，PPO 使用 RLinf + 固定 IsaacLab 环境。在确认两边的 Python、Torch、FlashAttention 与 Isaac Sim 依赖完全一致之前，保持两个虚拟环境独立，使用数据集与 SFT bundle 作为阶段接口。

两套环境分别生成 lock/manifest，至少记录 Python、Torch、CUDA runtime、FlashAttention、Transformers、AV/TorchCodec、Ray、Isaac Sim 和 IsaacLab commit。任何通过“后装包覆盖版本”形成的环境都必须通过 import、GPU kernel、视频解码和最小运行测试，不能只保存 `pip freeze`。

正式 H800 作业前完成资源预算：

- actor、rollout、Isaac Sim 分别及并发时的峰值显存；
- CPU RAM、共享内存、本地 NVMe 和 NAS 吞吐；
- 每个 PPO epoch 的 rollout/update wall time；
- checkpoint 大小、保存频率和最大保留数量。

目标是稳定运行时峰值显存不超过单卡容量的 90%，并预留评测视频与 checkpoint 写入空间。

## 最终完成标准

1. 全新 checkout 可以初始化三个固定源码依赖，并校验其 commit。
2. 数据、base model 和 Cosmos backbone 均按固定 HF revision 下载并通过校验；gated 许可证流程有文档。
3. 公开 stack-cube 数据通过 schema、实际 codec 样本解码、episode split 和数值 modality contract 测试。
4. 官方 N1.7 SFT 能生成符合离线 bundle contract 的 stack-cube 产物。
5. RLinf 使用 `libero_sim` processor 与 `isaaclab_stack_cube` converter 加载该 bundle。
6. 目标 IsaacLab task 能完成 rollout、PPO update、权重同步、checkpoint 保存和恢复。
7. 4090 完成强制开发 gate，其中包括单环境 WebRTC 实时 closed-loop 查看；若 actor update OOM，已有可复现记录和经过验证的替代 GPU 配置。
8. SFT-only 闭环评测在固定 seeds 上得到非零成功率后才进入正式 PPO。
9. H800 训练记录包含 SFT baseline、周期性 PPO 评测、成功率置信区间、checkpoint、TensorBoard 日志、资源指标和完整版本 manifest。
10. 数据示范、SFT 和 PPO 均有可回放产物；同一固定 seed 下可以直接比较 SFT 与 PPO 行为。
11. 报告分别给出工程是否完成、PPO 是否达到相对 SFT 至少 5 个百分点的学习改进，不混淆链路可运行与策略有效。
