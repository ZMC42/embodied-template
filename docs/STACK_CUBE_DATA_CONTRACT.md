# Stack-cube 数据 contract 验证

本轮只验证数据、processor 和 IsaacLab 数值接口，没有加载 policy 权重或执行 SFT/PPO update。原始数据保持固定 revision；SFT 必须使用显式转换后的数据，不能直接把原始 parquet 的旋转字段解释成 axis-angle。

## 输入与实际发现

- 数据：`RLinf/IsaacLab-Stack-Cube-Data@2e27f04b2e31f293fd020164917e4cce09e108d6`；下载 manifest 中全部文件的 SHA256 已复验。
- 源码：沿用 `configs/dependencies.lock.json` 的三个 submodule commit，本轮没有修改 submodule。
- Processor 设置来自固定 `GR00T-N1.7-3B` base；加载固定的本地 Cosmos processor。base 的 modality/statistics 没有 `libero_sim`，本轮通过官方 modality 注册表和真实 train statistics 显式新增该 embodiment；这不是 base 模型的 stack-cube 推理或 SFT checkpoint。

| 字段 | 实测结果与处理 |
| --- | --- |
| 原始 state | 8 维；旋转三维按 extrinsic Euler xyz、弧度解释。依据为全部轨迹的旋转连续性、示范画面末端方向和实际 IsaacLab quaternion 对照；公开数据未提供采集/转换源码，因此这是记录在 contract 中的数值推断 |
| SFT state | `[eef_xyz(3), principal_axis_angle(3), left_finger_pos, -right_finger_pos]`，float32；原始 xyz 和双指符号保留 |
| Runtime state | 固定 RLinf wrapper 已输出 axis-angle，但可能 angle > π。后续 `isaaclab_stack_cube` converter 必须取 principal branch，与派生数据一致 |
| Action | 7 维 relative IK manager 原始命令；不再次减去 state，不预乘 scale。manager 对前 6 维乘 0.5，得到 root-frame 平移（米）和左乘旋转向量（弧度） |
| Gripper | action `-1` 关闭、`+1` 打开、`0` 打开；双指目标为 `[0,0]` 或 `[0.04,0.04]` 米，state 的第二指额外取负 |
| 视频 | 294 个视频均为 **MPEG-4 Part 2 (`mpeg4`)**，256×256、20 Hz；原始 `meta/info.json` 声称 AV1，与实际文件不符。派生 metadata 修正 codec，不转码 |
| 数值 dtype | 原始 parquet 数组实际为 float64，metadata 声明 float32；派生 state/action 显式转换为 float32 |
| 相机 | `image ← front/table_cam`、`wrist_image ← wrist/wrist_cam`，RGB uint8。Runtime 默认 200×200，必须覆盖为 256×256 |

episode 0 的原始旋转若误按 axis-angle，最大相邻旋转变化为 **2.4112 rad**，首帧末端局部 z 轴朝上；按 Euler xyz 为 **0.00945 rad**，局部 z 轴朝下。全部 147 条示范按 Euler xyz 解释的最大相邻旋转变化为 **0.03995 rad**。转换后与原 Euler 旋转矩阵的最大误差为 **1.30×10⁻⁷ rad**。

## Split 与 normalization

`experiments/stack_cube/sft/episode_split.json` 固定完整 episode ID、seed=42、算法 `numpy.random.Generator(PCG64).permutation(sorted IDs)`，划分 117 train / 15 validation / 15 test。没有按 frame 拆分。

派生数据位于 `runs/stack-cube/data/{train,validation,test}`，保持原 episode ID。各 split 只包含自身的 parquet 和视频链接；视频使用 NAS 支持的相对链接。`meta/stats.json` 仅由 train 的 **42,435 帧**计算，validation/test 复用同一文件，禁止重新计算各自 statistics。每个目录的 `MANIFEST.json` 记录来源 revision、episode IDs、statistics 来源、contract/split hash 和输出文件 hash。

固定 `use_percentiles=false`，使用 min/max normalization；`use_relative_action=false`，不把已经是相对 IK 的动作再次转为相对动作。保留官方 `clip_outliers=true`。全部 train action encode/decode 最大误差 **5.96×10⁻⁹**。若沿用 q01/q99 clipping，会改变 train 中 **2,918 帧**的真实命令，最大绝对误差 **0.07734**。这些数字来自 train；没有使用 held-out action 指标选择 normalization。

Min/max clipping 对超出 train 范围的 held-out 或 runtime 值仍可能有损；不声称这些值可以无损 round-trip。测试集只做格式检查，没有模型推理、调参或模型选择。

**后续 SFT 接入注意**：固定上游 `Gr00tN1d7Processor.from_pretrained()` 的 override 白名单不包含 `use_percentiles`。仅把 `--model.use-percentiles false` 传给从 checkpoint 初始化的路径可能被忽略。第 4 步必须使用本轮保存的 processor 配置，或通过显式本地 processor 初始化设置该值，并重新断言实际值；不得只看 CLI。本轮使用官方类构造和序列化/重载，未 monkey patch 上游。

## 自动化覆盖与结果

**2026-10-02 实测通过**：官方 SFT 环境中的 10 项测试全部通过，pytest wall time 12.75 秒。Python 3.10.12、Torch 2.7.1+cu128、TorchCodec 0.4.0、PyAV 16.1.0、Transformers 4.57.3、NumPy 1.26.4；`uv pip check` 检查 169 个包通过。环境按官方 frozen uv.lock + dev extra 安装；大型 wheel 的并行下载已逐个复验该 lock 的 SHA256，最终 frozen sync 正常完成。完整安装与验证日志在 `runs/stack-cube/contract/`。本轮验收的是 CPU data/processor 路径；模型 GPU forward/backward 和 FlashAttention kernel 属于后续微型 SFT gate。

`tests/test_stack_cube_contract.py` 覆盖真实资产：

- 全部 147 episodes、53,265 帧的 schema、维度、frame/timestamp、有限值、旋转等价、双指符号、split 完整性和 train-only statistics；
- episode 0、最长 episode 7、边界 episode 146 的乱序/重复 RGB frame 解码，PTS 与 parquet timestamp 对齐，并由独立 PyAV decoder 交叉核对；
- episode 0/7 的开头、中间、最后完整 horizon、首个越界 horizon、最后一帧，以及每次 gripper 切换前后，共 34 个起始帧；
- 官方 `LeRobotEpisodeLoader`、`extract_step_data`、processor 和 collator；state=8、action=7、action horizon=16，模型 padding 为 40×132，padding 数值与 mask；
- 全 train actions round-trip；processor 保存/离线本地重载后的 `libero_sim` metadata、embodiment ID=2 和配置；
- Euler 被误当 axis-angle 的数值回归与 runtime principal branch 转换。

官方训练 `ShardedSingleStepDataset.get_effective_episode_length()` 为 `L−16+1`，即丢弃末尾 15 个起始帧，**即使 allow_padding=true 也是如此**。显式 `extract_step_data(..., allow_padding=true)` 在边界重复最后一个 action，不产生 validity mask，重复值仍被 processor 当成 16 个有效 action。当前 contract 保持官方训练的截断行为，边界测试只验证显式采样行为，不宣称末尾 padded samples 会进入训练。

`scripts/verify_stack_cube_runtime.py` 在固定 N1.5 IsaacLab 环境启动一次真实 headless task，不加载模型：验证实际 `_wrap_obs`、两路 RGB observation、相机配置、全部 6 轴正负命令、IK 目标位姿、0.5 scale、gripper 的 -1/0/+1 目标和 20 个 simulation steps 后的实际开合。已通过；关闭后的 state 为约 `[0.000110,-0.000115]` 米，打开后为 `[0.0399999,-0.0399999]` 米。

结果、环境版本和 FFmpeg 链接信息保存在 `runs/stack-cube/contract/`，以该目录的实测文件为准。

## 复现

先按 `PREFLIGHT.md` 下载固定资产，并完成前两步的 IsaacLab 基线环境。官方 SFT 环境与项目 Python 依赖分离：

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venvs/sft-n1.7" \
  uv sync --frozen --extra dev --project third_party/Isaac-GR00T
bash scripts/validate_stack_cube_contract.sh

source scripts/project_env.sh
OMNI_KIT_ACCEPT_EULA=YES \
  .venvs/isaaclab-n1.5/bin/python -u scripts/verify_stack_cube_runtime.py \
  > runs/stack-cube/contract/runtime.log 2>&1
```

验证入口会复验原始 hash、重建派生数据、运行测试、生成两条回放，并保存完整环境 lock、官方 uv.lock hash、PyAV 库版本、TorchCodec 的 FFmpeg 动态库链接和库 SHA256。本机 TorchCodec 实际加载 `libtorchcodec_decoder4.so`，使用 Ubuntu 22.04 的 FFmpeg 4.4.2（`libavcodec58`、`libavformat58`、`libavutil56`、`libswscale5`、`libswresample3`，包版本 `7:4.4.2-0ubuntu0.22.04.1`）。需要与记录一致的系统 FFmpeg shared libraries；TorchCodec 使用系统库，PyAV wheel 自带库，两者不能混称。

输入在 NAS 上；本轮验证的是样本随机解码能力，不是正式训练节点的冷缓存吞吐验收。H800 作业前仍须执行本地 staging/NAS 吞吐测试。

## 可观察产物与边界

- `runs/stack-cube/visualization/dataset/episode_000000.mp4`：312 帧，15.6 秒；含 6 次 gripper 切换。
- `runs/stack-cube/visualization/dataset/episode_000007.mp4`：全数据最长，588 帧，29.4 秒；含 6 次 gripper 切换。
- 两条均为同步 front/wrist 并排，叠加 task、frame、timestamp、**原始 Euler state** 和 relative IK action。同目录 JSON 记录来源 revision 和视频 hash。
- 已检查两条视频的开头、中间、末尾画面；末尾可以看到蓝→红→绿堆叠。公开数据没有 success flag，因此不把目视结果当成自动化成功率。
- `runs/stack-cube/contract/runtime-{image,wrist_image}.png` 为真实环境 reset 画面，用于对照视角。

公开数据没有采集时的相机外参、focal length、采集任务 commit、原始 quaternion、IK scale、动作转换代码或成功标志。固定 runtime 的 IK scale=0.5 已验证，但无法据此证明采集时使用了同一 scale。自动化测试能证明本轮固定 runtime 配置与所定义 contract 一致、解码顺序和数值转换一致；不能证明缺失的采集参数与 runtime 完全相同。相机位姿使用固定任务配置，并已做画面对照。这一边界必须随 SFT/PPO bundle manifest 保留；下一步仍仅允许微型 SFT，后续同任务 closed-loop gate 必须验证域匹配。
