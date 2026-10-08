# 在另一台服务器配置项目

`scripts/setup_project.sh` 在 Linux x86_64 NVIDIA GPU 服务器上配置项目的五套独立环境，初始化固定源码、资产目录和 GPU/视频库检查。服务器应已安装 NVIDIA 驱动及 CUDA Toolkit，并能通过 `nvidia-smi` 和 `nvcc --version` 查看版本。

克隆后执行一次配置即可；三个项目 submodule 由脚本按父仓库固定的 commit 初始化：

```bash
git clone https://github.com/ZMC42/embodied-template.git
cd embodied-template
bash scripts/setup_project.sh
```

默认安装 Ubuntu 22.04/24.04 所需的系统包（FFmpeg、编译工具、OpenGL/EGL、Vulkan、libaio），使用 root 或 sudo 执行 apt，随后安装 uv 和带编译头文件的 Python。脚本使用现有驱动和 CUDA Toolkit；系统 Toolkit 与 Torch 的 CUDA 版本需要兼容。Isaac Sim 5.1 的 Linux 驱动要求见[训练计划](TRAINING_PIPELINE.md)，目标 Blackwell GPU 仍需验证运行兼容性。

| 环境 | 用途 | 安装依据 |
| --- | --- | --- |
| `.venvs/dev` | 项目工具及 Hugging Face 下载 CLI | 项目 editable 安装 |
| `.venvs/sft-n1.7` | N1.7 SFT、数据 contract 测试 | 官方 frozen `uv.lock`，Python 3.10，含 dev extra |
| `.venvs/ppo-isaaclab` | N1.7 IsaacLab PPO | `ppo_isaaclab_n1_7_requirements.lock.txt`，Python 3.11.14 |
| `.venvs/isaaclab-n1.5` | N1.5 IsaacLab 基线 | 对应 requirements lock 及固定 GR00T worktree |
| `.venvs/n1.7-libero` | N1.7 LIBERO 基线 | 对应 requirements lock 及固定 RLinf worktree |

默认只安装环境，不下载模型或演示数据，也不执行训练。五套环境和缓存会占用数十 GB 磁盘；后续下载的模型、数据、训练产物另计。只准备 N1.7 SFT → PPO 主线时，可减少安装范围：

```bash
bash scripts/setup_project.sh --envs dev,sft-n1.7,ppo-isaaclab
```

`dev` 是安装及下载工具环境，无论 `--envs` 选择什么都会创建。其他 Linux 发行版，或系统依赖已由管理员准备好的机器，使用 `--skip-system-deps` 跳过 apt；仍需提供脚本列出的系统库与编译工具。

## 资产存储与下载

新 checkout 的资产默认放在项目 `.assets/`，通过 `models`、`datasets`、`isaac-sim` 和 `runs` 链接访问。已有 checkout 会沿用现有 `models` 链接对应的资产根目录，也可以明确指定服务器的数据盘：

```bash
bash scripts/setup_project.sh --asset-root /data/embodied-template-assets
source scripts/project_env.sh
```

后续 `source scripts/project_env.sh` 会从现有链接读取资产位置，无需重新设置原服务器的 NAS 路径。已有链接指向另一位置时，安装器会报告冲突；先迁移已有资产并调整链接，再重新运行配置。不要复制另一台服务器的 `.venvs/`，其中的路径需要在目标机器重新生成。

需要同时下载训练资产时，添加 `--download-assets`。Cosmos 模型需要先在 [Hugging Face 模型页](https://huggingface.co/nvidia/Cosmos-Reason2-2B)取得访问权限，并提供 `HF_TOKEN` 或已有 Hugging Face 登录：

```bash
bash scripts/setup_project.sh --asset-root /data/embodied-template-assets --download-assets
```

模型、演示数据和可选基线资产按 `configs/dependencies.lock.json` 的 revision 下载。Token 通过环境变量或 Hugging Face 的本机登录缓存传入，不写入脚本或仓库。若本机尚未登录，可先完成环境安装，再执行：

```bash
.venvs/dev/bin/hf auth login
bash scripts/setup_project.sh --download-assets
```

GR00T 仓库中的 demo LFS 文件不属于项目训练资产，初始化时跳过自动 smudge；项目使用单独下载的 stack-cube 数据。PPO 中的 Isaac Sim 5.1 通过 Python wheel 安装，无需另行下载 standalone Isaac Sim 压缩包。`isaac-sim` 链接为现有 standalone 安装保留，新服务器无需向该目录填充文件。

训练生成的 `models/stack-cube-n1.7-sft` bundle 不在公开下载资产中；PPO 闭环还需要在目标服务器生成该 bundle，或连同 manifest 一起从原服务器迁移。环境配置成功并不代表训练或评测已完成。

## 安装结果与验证

安装后，每套 GPU 环境会验证 Torch GPU 访问、FlashAttention BF16 前向和反向的有限值、PyAV 与 TorchCodec 动态库加载；SFT/PPO 还会记录依赖和 runtime 版本。报告位于 `runs/preflight/`，完整安装日志位于 `tmp/setup/setup-*.log`。固定 PPO 组合有已知的依赖 metadata 冲突，详见[集成报告](STACK_CUBE_N1_7_INTEGRATION.md)；安装器保留报告，不通过升级包改变已经验证的组合。

需要进一步验证 IsaacLab 的目标 task 注册和双相机 headless reset，可以执行：

```bash
bash scripts/setup_project.sh --envs dev,ppo-isaaclab --smoke-test
```

该检查沿用项目入口的 `OMNI_KIT_ACCEPT_EULA=YES`，运行 Isaac Sim，并可能联网加载场景资产。普通配置只执行小型 GPU kernel 检查。安装失败时脚本立即停止并报告日志位置；解决下载、权限或兼容问题后，重跑原命令会复用已有环境、安装缓存与下载内容。

运行项目命令前加载路径，再选择所需环境：

```bash
source scripts/project_env.sh
source .venvs/sft-n1.7/bin/activate
```
