# GR00T N1.7 + IsaacLab 完整训练方案

本文确定 `embodied-template` 的实施范围、验证顺序和 go/no-go 条件：使用公开的 IsaacLab 仿真示范完成 NVIDIA GR00T N1.7 SFT，再将 SFT 产物交给 RLinf，使用 PPO 在同一个 stack-cube 任务上继续训练。

## 可行性结论与执行状态

该路线技术上可行，但当前属于**有条件可行**，尚不是可直接启动完整 H800 训练的已验证配置。数据 schema、官方 SFT 入口、RLinf 的 N1.7 模型支持和 checkpoint processor 保存机制均已存在；尚未闭合的风险集中在：

- IsaacLab 源码和仿真依赖尚未纳入顶层版本锁定；
- SFT 与 PPO 间的 embodiment、状态表示和 normalization contract 尚未通过运行时测试；
- N1.7 + IsaacLab 仍是未被上游 e2e 覆盖的新组合；
- 单张 RTX 4090 能否完成 actor update 尚无实测依据；
- 当前仓库仍是 scaffold，项目级配置、contract test 和目标 e2e 尚未实现。

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

RLinf 已经支持 GR00T N1.7，但当前维护的 N1.7 RL 示例是 LIBERO Spatial。RLinf 的 IsaacLab 示例目前使用 GR00T N1.5 或 OpenPI pi0.5，安装脚本也会拒绝 `gr00t_n1d7` 与 `isaaclab` 的组合。因此，N1.7 + IsaacLab 是本项目需要实现和验证的组合，而不是复制现有 YAML 即可运行的示例。

| 组合 | 在本项目中的用途 | 当前上游状态 |
| --- | --- | --- |
| GR00T N1.5 + IsaacLab | 验证 IsaacLab、Ray worker 与 PPO 环境链路 | RLinf 已提供文档、配置与 e2e 路径 |
| GR00T N1.7 + LIBERO Spatial | 验证 N1.7 模型、processor、rollout 与 actor 更新 | RLinf 当前维护的 N1.7 示例 |
| GR00T N1.7 + IsaacLab | 最终训练目标 | 需要完成集成与验证 |

这里的关键区别是：框架分别支持一个模型和一个环境，并不表示两者的任意组合都已经经过验证。

## 源码与资产版本锁定

所有影响数据解释、仿真动力学、模型加载和训练行为的源码都必须固定到 commit。下表同时是实施要求；状态为“待纳入”的依赖在完成前不满足可复现性标准。

| 依赖 | 仓库 | 固定版本 | 状态 |
| --- | --- | --- | --- |
| RLinf | `https://github.com/ZMC42/RLinf.git` | `61ba34e640035f2e4ac4ef3c6078f02d43a00c9a` | 已作为 submodule 固定 |
| Isaac-GR00T | `https://github.com/NVIDIA/Isaac-GR00T.git` | `n1.7-release`，即 `23ace64f17aa5015259b8609d371eb61a357c776` | 已作为 submodule 固定 |
| IsaacLab | `https://github.com/RLinf/IsaacLab.git` | `4246b6b4f4a3e74ee20e002ed7536b1c788d39f4` | 待作为 submodule 或等价的 commit-locked source 纳入 |

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

`scripts/setup_assets.sh` 在项目根目录创建 `datasets`、`models`、`isaac-sim` 和 `runs` 软链接。Isaac Sim cache、shader cache、HF cache 和训练临时文件保留在本地 NVMe，避免共享 NAS 的小文件延迟影响运行。正式训练前对 AV1 视频随机读取做吞吐测试；若 NAS 吞吐不足，将固定 revision 的数据集 staging 到训练节点本地盘，并记录源 revision。

## 数据与模型 contract

### 数据 contract

stack-cube 数据集公开且无需审批，包含 147 条 Franka 仿真示范、53,265 帧，总时长约 44 分钟，采用 LeRobot v2 格式。每一帧包含 256 × 256 的 front/wrist 双相机图像、8 维绝对 EEF state、7 维相对 EEF action，以及 stack-cube 的自然语言指令。

数据集 `meta/modality.json` 的字段与 N1.7 的 `libero_sim` modality 一致，包括 `image`、`wrist_image`、`x`、`y`、`z`、`roll`、`pitch`、`yaw` 和 `gripper`。字段相同不能直接证明行为兼容，必须验证：

- runtime state 精确为 `[eef_xyz(3), eef_axis_angle(3), gripper_joint_pos(2)]`；虽然 modality key 名为 `roll/pitch/yaw`，不得把这 3 维误解释为 Euler angle；
- action 精确为 IsaacLab relative IK 接受的 7 维命令，包含旋转与 gripper 的方向、范围和单位；
- front/wrist 相机顺序、分辨率、颜色通道、相机位姿和图像 dtype 与数据采集时一致；
- gripper state 的双指维度、action 的单维开合约定，以及 `[-1, 1]` 符号与 IsaacLab action manager 一致；
- processor normalization 后再 decode 的 action 与原始 action 数值 round-trip 一致；
- AV1 视频能在 SFT 环境中稳定随机解码，不依赖未记录的系统 FFmpeg 配置。

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
- `obs_converter_type` 独立使用 `isaaclab_stack_cube`，负责把 IsaacLab observation 转为 checkpoint 的 `libero_sim` schema；
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

Isaac-GR00T 的 `open_loop_eval.py` 会计算 MSE/MAE，并输出专家动作与模型预测动作的曲线图。例如在 SFT 虚拟环境中运行单条 held-out 轨迹：

```bash
python third_party/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
  --dataset-path datasets/isaaclab-stack-cube \
  --embodiment-tag LIBERO_PANDA \
  --model-path models/stack-cube-n1.7-sft/checkpoint \
  --traj-ids 132 \
  --action-horizon 16 \
  --save-plot-path runs/stack-cube/visualization/open-loop/traj-132.jpeg
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

完成条件：离线检查脚本能解析全部源码 commit 和资产 revision；IsaacLab 能列出目标 task ID，并成功完成一次 headless reset；网络预检确认 4090 到开发者客户端具备 WebRTC 私网/VPN 或受控公网路径。

### 1. 验证 IsaacLab 基线链路

缩小 GR00T N1.5 + IsaacLab 配置，验证环境 reset、rollout、reward 收集和 actor update。若单张 4090 无法容纳 actor update，先记录峰值显存和失败位置，再在最小多卡/H800 配置完成该基线，不把 OOM 误判为接口失败。

完成条件：至少一次完整 update，无 shape、device、NaN 或 worker 生命周期错误，并能保存和恢复 checkpoint。

### 2. 验证 N1.7 基线链路

以 smoke-test 规模运行 RLinf 维护的 GR00T N1.7 + LIBERO 配置。

完成条件：成功加载模型与 processor，完成 rollout、actor update、权重同步、保存和恢复。记录基线所需 GPU 数、峰值显存和 wall time。

### 3. 验证 stack-cube 数据 contract

在不启动完整训练的情况下运行 schema、样本解码、processor encode/decode 和 observation/action 数值测试，并生成固定的 episode split。

完成条件：上述数据与模型 contract 全部变成自动化测试；至少抽取 episode 开头、中间和结尾样本，验证 horizon padding、边界 frame 和 gripper 切换；生成一条 front/wrist 双相机专家示范回放。

### 4. 生成微型 N1.7 SFT checkpoint

使用 `nvidia/GR00T-N1.7-3B` 和官方 `gr00t/experiment/launch_finetune.py`，仅运行足以产生 checkpoint 的 1–10 个 update。4090 先验证数据读取、processor、单步前向/反向和保存；如反向 OOM，可在 H800 上完成该微型产物，但仍不得直接进入完整 SFT。

完成条件：生成符合 bundle contract 的产物，并在全新进程、离线模式下加载，完成 held-out 样本的 open-loop inference，同时保存逐动作维度的预测—专家对比图。

### 5. 集成 N1.7 + IsaacLab 并交接微型 checkpoint

在 RLinf fork 中完成以下通用改动：

- 允许并安装 `gr00t_n1d7` + `isaaclab` 组合；
- 安装脚本复用固定的 `GR00T_PATH` 和 `ISAAC_LAB_PATH`，不得重复 clone；
- 增加 N1.7 IsaacLab 实验配置；
- 从 SFT bundle 加载 processor、statistics、embodiment mapping 和本地 Cosmos backbone；
- 在模型边界验证 observation key、axis-angle 语义和 state dimension；
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
3. 公开 stack-cube 数据通过 schema、AV1 样本解码、episode split 和数值 modality contract 测试。
4. 官方 N1.7 SFT 能生成符合离线 bundle contract 的 stack-cube 产物。
5. RLinf 使用 `libero_sim` processor 与 `isaaclab_stack_cube` converter 加载该 bundle。
6. 目标 IsaacLab task 能完成 rollout、PPO update、权重同步、checkpoint 保存和恢复。
7. 4090 完成强制开发 gate，其中包括单环境 WebRTC 实时 closed-loop 查看；若 actor update OOM，已有可复现记录和经过验证的替代 GPU 配置。
8. SFT-only 闭环评测在固定 seeds 上得到非零成功率后才进入正式 PPO。
9. H800 训练记录包含 SFT baseline、周期性 PPO 评测、成功率置信区间、checkpoint、TensorBoard 日志、资源指标和完整版本 manifest。
10. 数据示范、SFT 和 PPO 均有可回放产物；同一固定 seed 下可以直接比较 SFT 与 PPO 行为。
11. 报告分别给出工程是否完成、PPO 是否达到相对 SFT 至少 5 个百分点的学习改进，不混淆链路可运行与策略有效。
