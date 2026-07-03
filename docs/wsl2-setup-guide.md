# WSL2 训练环境搭建指南

> **目标**: 在 WSL2 Ubuntu 中运行现有的纯 PyTorch MAPPO 训练脚本
> （`train_dual_actor.py` / `train_attention_actor.py`）
>
> **预计耗时**: 30-45 分钟（含下载）

---

##   致命警告：禁止使用 RLlib

代码仓库中有 `train_formation_mappo.py`、`formation_mappo_env.py` 等早期 RLlib 脚本，
它们是为简单 MLP 架构编写的。你当前的核心资产是：

- 手搓的 `AttentionFormationActor`（33-dim → 3-token Self-Attention）
- Tokenized `AttentionCritic`（3-entity MHA over global state）
- 双 Actor 解耦架构（`train_dual_actor.py`，v6 验证通过）
- 纯定制 BC 预训练管线（`train_attention_bc_2v1.py`）
- 过滤后的协同 BC 数据（18K 样本，P1 不再逃跑）

RLlib 无法承载这些定制架构。**本指南不涉及 Ray/RLlib，只配置运行你自有脚本所需的环境。**

---

## 阶段一：系统层 (System Setup)

### 1.1 启用并安装 WSL2

以**管理员身份**打开 PowerShell，执行：

```powershell
wsl --install
```

这个命令会自动：
- 启用 Windows Subsystem for Linux 功能
- 启用虚拟机平台
- 下载并安装 Ubuntu（默认 WSL2 架构）

**重启电脑**。重启后 Ubuntu 终端会自动弹出，提示你设置 UNIX 用户名和密码（例如用户名 `sean`）。

### 1.2 限制 WSL2 资源占用（极度重要）

WSL2 默认会"吃光"所有可用内存，导致 Windows 宿主机卡死。

在 Windows 资源管理器地址栏输入 `%UserProfile%`，进入你的用户目录（如 `C:\Users\Sean\`）。

新建文件 `.wslconfig`（注意前面有个点），写入：

```
# 16GB 物理内存的电脑 (安全法则: 不超过总内存的 70%)
[wsl2]
memory=10GB
processors=6

# 如果是 32GB 内存的电脑, 改为:
# memory=22GB
# processors=12
```

保存后，在 PowerShell 中执行 `wsl --shutdown`，然后重新打开 Ubuntu 终端使配置生效。

### 1.3 GPU 驱动透传验证（如有 NVIDIA GPU）

**不要在 Ubuntu 内部用 `apt` 安装 NVIDIA 驱动。**

只要 Windows 宿主机的 NVIDIA 驱动是最新的（GeForce Experience 正常更新即可），
WSL2 会自动透传 CUDA。在 Ubuntu 终端验证：

```bash
nvidia-smi
```

如果能看到你的显卡信息和 CUDA 版本，说明透传成功。如果提示命令不存在，说明你的 Windows
NVIDIA 驱动需要更新，或者你的电脑没有 NVIDIA 显卡（CPU 训练也完全可以）。

---

## 阶段二：文件层 (File System Migration)

###   CRLF 换行符陷阱（先执行！）

因为代码仓库是从 Windows 来的，`.sh` 文件的换行符是 Windows 格式的 CRLF。
在 Linux 里直接跑会报 `\r: command not found`。

```bash
sudo apt update && sudo apt install dos2unix -y
```

### 2.1 抛弃 /mnt/c/

WSL2 访问 Windows 文件系统（`/mnt/c/`）的 I/O 性能极差——JSBSim 每次 reset 都要
读取 F-16 配置文件，在 `/mnt/c/` 下训练会显著变慢。

```bash
cd ~   # 回到 Linux 原生家目录 (/home/你的用户名/)
```

### 2.2 克隆/迁移项目

**方式一：重新 git clone（推荐）**

```bash
git clone https://github.com/你的用户名/jsbsim-marl-formation.git
cd jsbsim-marl-formation
```

**方式二：从 Windows 侧复制大文件**

BC 数据（`.npz`）和模型权重（`.pth`）太大，不适合 git。通过 Windows 资源管理器直接拖入：

在 Windows 文件夹地址栏输入：
```
\\wsl$\Ubuntu\home\你的用户名\jsbsim-marl-formation\
```

然后将以下文件从 Windows 项目目录拖入：
- `data/expert/attention_bc_2v1_filtered.npz`（18K 过滤 BC 数据）
- `data/expert/attention_bc_2v1_filtered_pretrained.pth`（BC 预训练权重）
- `data/expert/coop_pid_data.npz`（PID 协同数据，如有需要）
- `benchmarks/dual_actor_coop_best.pth`（v6 最优模型）
- `marl_runs/` 目录下的历史模型（如需保留）

### 2.3 修复换行符

```bash
dos2unix scripts/setup_wsl2.sh
dos2unix scripts/*.py      # 可选，预防性修复
```

---

## 阶段三：环境层 (Environment Setup)

### 3.1 安装 Miniconda (Linux 版)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：
- 按 Enter 浏览 license
- 输入 `yes` 同意条款
- 确认安装路径（默认 `~/miniconda3` 即可）
- 问是否 `conda init`，输入 `yes`

**关闭并重新打开 Ubuntu 终端**，让 conda 生效。

### 3.2 创建专属 Conda 环境

```bash
conda create -n marl_env python=3.10 -y
conda activate marl_env
```

### 3.3 安装核心依赖

**PyTorch (Linux + CUDA 版):**

前往 https://pytorch.org/get-started/locally/ 复制适用于 Linux 的 conda 安装命令，
或直接执行：

```bash
# CUDA 12.1 版本 (推荐)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 如果没有 NVIDIA GPU，使用 CPU 版:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**其他依赖:**

```bash
pip install jsbsim==1.3.1
pip install stable-baselines3 gymnasium tensorboard matplotlib pyyaml
```

### 3.4 验证安装

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')
import jsbsim; print(f'JSBSim: OK')
import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')
from torch.utils.tensorboard import SummaryWriter; print('TensorBoard: OK')
print('All packages OK')
"

python scripts/verify_installation.py
```

---

## 阶段四：开发层 (IDE Integration)

### 4.1 安装 VS Code WSL 插件

在 Windows 的 VS Code 中，搜索并安装插件：**WSL**（微软官方出品，
或搜索 `Remote Development` 扩展包，其中包含 WSL 支持）。

### 4.2 连接到 WSL2 工作区

1. 点击 VS Code 左下角的蓝色 `><` 图标
2. 选择 **"Connect to WSL"**（或 "连接到 WSL"）
3. VS Code 会自动在新窗口中打开 WSL 环境
4. 在 VS Code 左侧文件浏览器中点击 **"Open Folder"**，导航到 `/home/你的用户名/jsbsim-marl-formation`

此时：
- 终端直接是 Ubuntu 的 bash，conda 环境已激活
- 代码补全和 IntelliSense 在 Linux 环境下运行
- `Ctrl+` ` 打开终端直接运行训练脚本
- 所有文件读写都在高性能的 Linux ext4 文件系统上

---

## 启动训练

```bash
# 确保在项目根目录，conda 环境已激活
conda activate marl_env
cd ~/jsbsim-marl-formation

# 双 Actor + 协作模式 (当前最佳配置):
python scripts/train_dual_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000

# 单 Actor 两阶段:
python scripts/train_attention_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000
```

---

## 常见问题

**Q: `nvidia-smi` 命令不存在？**
A: 你的电脑没有 NVIDIA 显卡，或 Windows 驱动需要更新。CPU 训练完全可以正常运行。

**Q: TensorBoard 能访问吗？**
A: 在 WSL2 中启动 `tensorboard --logdir marl_runs/xxx/tensorboard --bind_all`，
然后在 Windows 浏览器访问 `http://localhost:6006`（WSL2 自动端口转发）。

**Q: `import jsbsim` 失败？**
A: `pip install jsbsim==1.3.1`。Linux 下编译速度快，通常几秒完成。

**Q: 训练脚本找不到 `src/` 模块？**
A: 确保在项目根目录执行脚本。`sys.path.insert(0, ...)` 已自动处理路径。

**Q: `.wslconfig` 修改后没生效？**
A: 在 PowerShell 中执行 `wsl --shutdown`，然后重新打开 Ubuntu 终端。

**Q: WSL2 和 Windows 之间的文件怎么互相访问？**
A:
- Windows → WSL2: 资源管理器地址栏输入 `\\wsl$\Ubuntu\`
- WSL2 → Windows: `/mnt/c/Users/你的用户名/`
