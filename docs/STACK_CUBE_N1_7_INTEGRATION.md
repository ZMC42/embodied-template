# N1.7 + IsaacLab 微型 checkpoint 集成

本文记录训练流水线第 5 步的代码、独立运行环境、禁网模型加载与 stack-cube 闭环验证。微型 checkpoint 只更新过一次 action projector；本轮用于工程验收，不用于判断堆叠能力。

## 复现入口

```bash
bash scripts/setup_assets.sh
python scripts/setup_stack_cube_integration.py
bash scripts/run_stack_cube_integration.sh offline
bash scripts/run_stack_cube_integration.sh closed-loop
bash scripts/run_stack_cube_rlinf_eval.sh
```

安装器校验固定源码，在 `.venvs/ppo-isaaclab` 中按锁文件安装 Python 3.11.14、Torch 2.8.0+cu128、Transformers 4.57.3、FlashAttention 2.8.3、Ray 2.58.0、Isaac Sim 5.1.0 和固定 IsaacLab。GPU wheels、Isaac Sim wheels 与普通 wheels 分别使用对应 index，复用本地 wheel cache。源码以 editable 方式接入，直接使用已有 `GR00T_PATH` / `ISAAC_LAB_PATH` 对应的 checkout；N1.7 官方 SFT metadata 固定 Python 3.10，本运行环境使用 `pip --no-deps --ignore-requires-python`，不会覆盖官方 `.venvs/sft-n1.7`。

RLinf 集成固定到 `0bf6fd743bb3d652fd45784d35b859d6f1250345`，保留 Isaac-GR00T `23ace64f` 和 IsaacLab `4246b6b4`。RLinf commit 目前只在本地；`configs/patches/rlinf-n1_7-isaaclab.patch` 已实际从原始 `61ba34e6` replay 并重建出相同 commit hash，便于离线交接。

实际环境锁在 `runs/stack-cube/integration/environment.lock.txt`，依赖 metadata 检查记录在 `dependency-check.txt`。组合安装后有 72 项 metadata 不兼容，包含 Isaac Sim 的 Torch 2.7 声明与官方 GR00T SFT 依赖；兼容性结论仅覆盖本轮实测的加载与闭环路径，不能视为正式 PPO 环境已验收。

`offline` 在独立 user/network namespace 中运行，检查实际 interfaces 仅含 loopback，用 RLinf 加载 SFT checkpoint、processor 与本地 Cosmos，并对第 3 步保存的真实 reset observation 推理一个完整 chunk。实际 SFT decoder tensor 与 bundle 完全一致；train-only statistics 与 embodiment mapping 均与保存文件一致。它不启动 Isaac Sim，避免将 NVIDIA 场景资产网络访问混入模型离线验证。

`closed-loop` 使用 RLinf N1.7 模型和 IsaacLab observation wrapper，启动一个目标 task，执行两个 16 步 chunk，并写出 front/wrist 并排 MP4、逐步 action/reward 与显存记录。`run_stack_cube_rlinf_eval.sh` 使用 RLinf 的 Ray rollout/env workers 和正式 evaluation runner；actor update 属于第 6 步。

## 状态与动作边界

N1.7 独立 converter 保留 N1.5/N1.6 的既有行为。它严格接收 256×256 RGB uint8 双相机与 8 维 state，把 runtime axis-angle 转为与派生示范一致的 principal branch，并保留 `[left_finger_pos, -right_finger_pos]`。stack-cube 输入在 processor normalization 前保留 float32，避免先量化 BF16 使接近 pi 的旋转跨过 branch 边界。

N1.7 的 decode keys 为 `x/y/z/roll/pitch/yaw/gripper`，不同于 N1.5 的 `action.*`。输出保持原 relative IK 命令，六维 arm scale 0.5 由 IsaacLab action manager 施加；gripper 取 sign，正值打开、负值关闭、零值按环境约定打开。实际传入 action 为 float32，形状 `(1, 16, 7)`。

初始化检查 `libero_sim`、相机顺序、state/action keys、state dim=8、action dim=7、min/max normalization 和关闭 state dropout。processor 有效 horizon 为 16；模型 `action_horizon` 和 processor padding 容量为 40。第一次运行误将两者要求相等，被初始化检查拦截；修正为有效 horizon 不超过 padding 容量，实际禁网推理通过。

Cosmos 显式使用 `models/nvidia/Cosmos-Reason2-2B`。该相对软链接指向固定 snapshot，加载时保留 canonical 路径字串，直接传入官方模型和 processor，不为此路径增加 loader monkey patch。既有 LIBERO 非 canonical 本地路径的上游兼容路径继续保留。

## 远程实时查看

Mac 与服务器保持 Tailscale 在线，安装 NVIDIA [WebRTC Streaming Client 1.1.5](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)。服务器执行：

```bash
LIVESTREAM=1 PUBLIC_IP=100.110.52.16 \
  bash scripts/run_stack_cube_integration.sh interactive --interactive --seed 0
```

该模式显式把 WebRTC endpoint 广播为 Tailscale 地址。客户端 Server 填 `100.110.52.16`，使用 TCP 49100 与 UDP 47998。这里沿用 IsaacLab 的模式 1 来显式指定地址，传输仍走 Tailscale。只启动一个 Isaac Sim 实例。

等待服务器输出 `STREAM_READY` 后再连接；NAS 权重加载期间主线程暂不刷新 viewport，提前连接可能看到黑屏。命令在服务器终端输入：`play`、`pause`、`step`、`reset`、`table`、`wrist`、`quit`。暂停只停止调用物理 step，继续调用 `SimulationContext.render()`；保持 Kit timeline playing，以免 pause/resume 后 Fabric 画面不再跟随 GPU 物理状态。单步执行一个 simulator step，reset 使用固定 seed。终端输出每步 action、reward、success、policy inference latency、simulation step time 和实际 FPS；MP4 按仿真 20 Hz 保存，不等同于 WebRTC 实际 FPS。

## 2026-10-02 实测

| 验证 | 结果 |
| --- | --- |
| N1.7 converter 聚焦测试 | 7 passed |
| RLinf 模型单元测试 | 67 passed、1 skipped |
| 真正禁网加载与推理 | passed，输出 `(1, 16, 7)` |
| headless 闭环 | 32 steps，reward/success=0，正常退出 |
| MP4 | 33 帧，512×256，20 FPS，已解码并查看首尾帧 |
| 模型推理采样峰值显存 | 6,869 MiB |
| 模型 + IsaacLab headless 采样峰值显存 | 11,549 MiB（首次 standalone）；12,532 MiB（Ray eval） |
| 最终 WebRTC + SFT 闭环 | 50 步、51 帧双相机 MP4，峰值 13,158 MiB |
| WebRTC | passed；macOS 客户端确认实时方块相对位置变化 |
| Ray evaluation | passed；32 步、6 个有限指标、34 帧 MP4，正常退出 |

首次 headless 运行的报告在 Kit shutdown 后写出，因此未执行到写报告语句；该次 `rollout.json` 明确标注由成功运行的逐步 console log 重建，wall time 留空。入口已改为在 Kit shutdown 前写报告。首次缺失正确 EULA 环境变量与 horizon 检查失败的日志独立归档，不计入成功记录。

产物均在 `runs/stack-cube/integration/`，微型 bundle 仍为 `models/stack-cube-n1.7-sft/`。SFT bundle 内源版本记录保持训练时的原值；集成源码和环境版本单独记录。第 6 步的 PPO update、权重同步、训练 checkpoint 保存和恢复尚未启动。

## 实时画面静止的诊断

第一次 interactive 循环每步调用 Kit pause/play，客户端能连接、切换相机，但场景静止；第二次虽已连续推进物理，调用过 Kit pause 后相机仍停在初始位置。后者的 EEF tensor 已有明显位移，录像却没有同步反映，不能计入 WebRTC closed-loop gate。

不加载模型的最小仿真复现将固定相对 y 命令执行 20 步，分别只改变暂停方式。两种方式 EEF 均移动约 0.108 m；Kit timeline pause 的 wrist 图像 mean pixel delta 约 2.01（渲染噪声），保持 timeline playing 并仅 render 的 delta 约 10.64，已查看图像确认方块相对位置改变。最终入口保留 Fabric，只以是否调用 `env.step` 实现用户暂停。每步记录 simulation time 与 EEF state，验证器检查物理时间递增及真实 EEF 位移。最终 50 步运行已通过这些检查，macOS 客户端确认看到了方块在腕部画面中的相对位置变化。

`interactive-static/` 和 `interactive-fabric-failure/` 保留失败运行，不能用其中的有限 action 或连接成功声明可视化完成。

最终 WebRTC 作业的 EEF xyz 范围为约 0.050 / 0.057 / 0.058 m，左指范围约 0.039 m；客户端看不到腕部视角画面外的夹爪，但确认了闭环视角变化。该作业 wall time 424.3 s 包括权重加载和等待人工连接。Ray eval 的采样 wall time 为 149.2 s，纯 eval loop 约 6 s，全部 6 个 TensorBoard scalar 有限。两次均无 OOM，退出后 GPU 恢复约 42 MiB。

已有微型 SFT bundle 的训练 provenance 保留在其原 `MANIFEST.json`。本轮使用同一 bundle，不生成 SFT optimizer 状态；RLinf 的 PPO value head 在需要时新建，原 action decoder 的 SFT 权重经过实际比较确认被保留。reward/success 均为零，动作仍有饱和与频繁 gripper 切换，禁止把本轮工程成功解释为学会堆叠。

RLinf 已加入带资产条件的 CI e2e job，需在 runner 上预先配置 `STACK_CUBE_SFT_BUNDLE`、`COSMOS_BACKBONE_PATH` 和固定源码路径。CI 本身及通用在线安装器全流程未在本轮运行；本轮实际安装使用项目级 pinned 环境入口，Ray 使用与 e2e 配置等价的项目配置。Docker 构建未纳入本轮范围。
