# 运行前检查

依赖的 commit、HF revision、环境 lock 输入和许可证版本统一记录在
`configs/dependencies.lock.json`。

新服务器可先执行 `bash scripts/setup_project.sh` 一次配置全部环境。指定
`--envs dev,sft-n1.7,ppo-isaaclab` 可只安装 N1.7 训练主线；指定
`--download-assets` 可同时下载固定资产。系统依赖、存储位置和安装日志见
[`SETUP.md`](SETUP.md)。

```bash
bash scripts/setup_assets.sh
uv tool install huggingface_hub
hf auth login
python scripts/download_assets.py all
python scripts/preflight.py offline
```

`nvidia/Cosmos-Reason2-2B` 是 gated 模型。下载前须在 Hugging Face 模型页接受
NVIDIA Open Model License；token 只通过 `hf auth login` 或 `HF_TOKEN` 提供，不写入仓库。

创建相互隔离的 SFT/PPO 环境并记录实际 Python、Torch、CUDA、FlashAttention、
Transformers、AV/TorchCodec、Ray、Isaac Sim、IsaacLab 和 driver 版本：

```bash
bash scripts/setup_environments.sh
```

两套环境分别创建在 `.venvs/sft-n1.7/` 和 `.venvs/ppo-isaaclab/`，
不会在 submodule 中创建 `.venv`。已验证基线使用各自独立的
`.venvs/isaaclab-n1.5/` 和 `.venvs/n1.7-libero/`，安装命令见对应基线报告。

PPO 使用 `setup_stack_cube_integration.py` 安装已验证的固定 N1.7 + IsaacLab
运行环境，包含 RLinf、GR00T 和 IsaacLab editable 源码。Isaac Sim 5.1 使用
Python wheel 安装，版本检查以当前环境中的 `isaacsim` package 为准。

环境 manifest 和可复用的 `*-requirements.lock.txt` 写入 `runs/preflight/`。
锁文件排除已由源码 commit 固定的 editable project；在同版本 Python 环境中可用
`uv pip install -r runs/preflight/ppo-requirements.lock.txt` 复用其余解析结果。
在 PPO 环境中验证目标 task 注册和一次 headless reset：

```bash
source .venvs/ppo-isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/isaaclab_smoke.py
```

WebRTC 使用私网/VPN 时检查到客户端的路由：

```bash
python scripts/preflight.py network private <client-vpn-ip>
# 在客户端执行：
python scripts/preflight.py network-client <server-vpn-ip>
```

公网模式须先设置客户端可达的 `PUBLIC_IP`，并仅开放 Isaac Sim 5.1 文档要求的
端口及来源地址：

```bash
PUBLIC_IP=<server-public-ip> \
  python scripts/preflight.py network public <client-public-ip>
# 在客户端执行：
python scripts/preflight.py network-client <server-public-ip>
```
