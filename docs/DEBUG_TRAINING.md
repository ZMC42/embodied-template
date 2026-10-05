# 使用 VS Code 断点学习 SFT 和 PPO

用 VS Code Remote SSH 连接 4090 服务器，打开项目根目录，在远程端安装
Microsoft Python 和 Python Debugger 扩展。运行和调试面板使用
[launch.json](../.vscode/launch.json)，无需手动切换 Python 环境。

这两个训练入口会真实计算梯度、更新参数并保存 checkpoint。SFT 使用 batch=1、
一次 projector update；PPO 使用一个 IsaacLab 环境、5 步 rollout 和一次
projector/value-head update。沿用已验证的 4090 配置，不追求任务成功率。

每次启动的输出都在 `tmp/debug/sft/<时间戳>/` 或 `tmp/debug/ppo/<时间戳>/`。
SFT 的 staging、checkpoint 和 bundle 都在本次目录中，不恢复旧调试 checkpoint。
这是 SFT 入口的 `--output-dir` 参数指定的输出父目录，脚本自动创建时间戳子目录；
该参数只控制产物位置，VS Code 断点由 debugpy 管理。
PPO 始终从已有 `models/stack-cube-n1.7-sft/checkpoint` 初始化，不自动使用刚生成的
SFT 调试产物。PPO checkpoint、TensorBoard、轨迹和视频也写入本次目录。

## SFT 单步训练

新手可结合 [SFT pipeline 学习指南](SFT_PIPELINE_GUIDE.md)，按顺序观察示范取样、
张量变换、flow matching loss、梯度和权重更新；指南提供具体断点和 Debug Console 表达式。

选择 **SFT: N1.7 单步训练**，按 F5。程序先停在入口；设置断点后按 F5 继续。
入口会校验 NAS 上的模型文件，首次到达训练断点前可能需要等待。

建议依次观察：

1. `scripts/train_stack_cube_sft_micro.py` 的 `run(config)`：查看实际训练配置。
2. `third_party/Isaac-GR00T/gr00t/experiment/trainer.py` 的
   `Gr00tTrainer.get_train_dataloader()`：查看 dataset 和 collator。
3. 同一文件的 `Gr00tTrainer.compute_loss()`：查看 batch、模型输出和 loss。
4. `.venvs/sft-n1.7/lib/python3.10/site-packages/transformers/trainer.py` 的
   `Trainer.training_step()`：跟踪反向传播和后续 optimizer 更新。

`justMyCode=false` 允许进入第三方源码；数据加载 worker 数为 0，数据读取也可
在同一进程内调试。F10 跨过函数、F11 进入函数、Shift+F11 返回。
训练结束后在本次目录的 `bundle/` 中查看保存的产物。

## PPO 主进程

新手可结合 [RL pipeline 学习指南](RL_PIPELINE_GUIDE.md)，先在主进程观察调度顺序，
再连接 actor worker，跟踪 rollout、GAE、PPO loss、梯度累积、参数更新与保存恢复。
指南提供具体断点和 Debug Console 表达式，并解释仿真子进程的额外调试边界。

选择 **PPO: N1.7 单步训练（主进程）**，按 F5。建议在
`third_party/RLinf/rlinf/runners/embodied_runner.py` 的 `EmbodiedRunner.run()`
内观察 `update_rollout_weights()`、`generate_rollouts`、
`compute_advantages_and_returns()` 和 `run_training()`。

这些调用把任务提交给 Ray workers。主进程中的 F11 不会跨进程进入 actor、
rollout 或 env 的实现。配置关闭子进程自动调试，避免把 Ray 后台服务都接入调试器；
调试 worker 使用下面的独立 attach 会话。`RLINF_TIMEOUT=120` 将分布式通信
超时延长到 120 分钟，便于暂停查看变量。

## PPO worker 内部断点

Ray 2.58 已集成 debugpy。第一次连接某个 worker 时，在希望停下的方法中临时加入
`breakpoint()`。例如，在
`third_party/RLinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py` 的
`EmbodiedFSDPActor.run_training()` 开头加入这一行，可以在实际参数更新前暂停。
保存修改后启动 PPO 主进程。

1. 等待终端输出 `Ray debugger is listening on <IP>:<端口>` 和
   `Waiting for debugger to attach`。
2. 保持主进程会话运行，选择 **PPO: Attach Ray worker** 并启动第二个调试会话。
3. 输入日志中的 IP 和动态端口。连接成功后，即可在该 worker 中用编辑器断点、
   F10/F11、变量面板和 Debug Console 调试。
4. 学习结束后删除临时的 `breakpoint()`。

actor 内建议观察 `compute_advantages_and_returns()`、`run_training()`、
`train_micro_batch()`、`loss.backward()` 和 `optimizer_step()`。
调试 rollout 或 env 时，将首次连接用的 `breakpoint()` 放到对应 worker 方法中，
分别连接各自日志中的端口。普通编辑器断点只有在连接对应 worker 后才会命中。

SFT 和 PPO 分开启动；前一个训练进程退出、释放显存后，再启动下一个。
这里直接启动 Python 训练入口，不执行 shell runner 的资源采样和训练后验收脚本。
VS Code 停止主进程不保证 Ray workers 全部退出；确认本次 worker 已退出后再重启。
