# WSL2 训练环境搭建指南

> 目标: 在 WSL2 Ubuntu 中运行现有的纯 PyTorch MAPPO 训练脚本
> （train_attention_actor.py / train_dual_actor.py 等）

##   致命警告：不要使用 RLlib

代码仓库中存在 `train_formation_mappo.py`、`formation_mappo_env.py` 等 RLlib 脚本，
但它们是早期为简单 MLP 架构编写的。你刚刚跑通的 v6 dual 模型的核心壁垒在于：

- 手搓的 `AttentionFormationActor`（33 维观测 → 3-token Self-Attention）
- Tokenized `AttentionCritic`（3-entity MHA over global state）
- 双 Actor 解耦架构（`train_dual_actor.py`）
- 纯定制的 BC 预训练管线（`train_attention_bc_2v1.py`）

RLlib 是高度封装的框架，将这些定制架构接入其 `TorchModelV2` API 的工作量无异于重写整个项目。
**请直接跳过所有 Ray/RLlib 相关内容，继续运行你自己的训练脚本。**

---

## 你需要自己完成的操作（需要 Windows 管理员权限 + 重启）

### Step 1: 启用 WSL2（一次性）

以管理员身份打开 PowerShell，执行:

```powershell
# 启用 WSL 功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 重启计算机
Restart-Computer
```

重启后，以管理员身份打开 PowerShell:

```powershell
# 下载并安装 WSL2 内核更新包
wsl --update

# 设置 WSL2 为默认版本
wsl --set-default-version 2

# 安装 Ubuntu
wsl --install -d Ubuntu-22.04
```

首次启动 Ubuntu 时，会提示创建 Linux 用户名和密码。

### Step 2: GPU 驱动（如果有 NVIDIA GPU）

**不需要去 NVIDIA 官网下载"WSL2 专用驱动"。** 当前的 Windows 11/10 体系下，
只要 Windows 宿主机的常规 NVIDIA 驱动是最新的（GeForce Experience 正常更新即可），
CUDA 会自动透传给 WSL2。跳过一切"WSL2 专用驱动"的旧教程。

---

## WSL2 环境配置

###   CRLF 换行符陷阱

因为代码仓库是从 Windows 拿过来的，`.sh` 文件的换行符大概率是 Windows 格式的 CRLF。
在 WSL2 (Linux) 里直接跑 CRLF 结尾的脚本会报 `\r: command not found` 错误。

**进入 WSL2 后，先修复换行符再执行任何脚本:**

```bash
sudo apt install dos2unix -y
dos2unix scripts/setup_wsl2.sh
```

### Step 3: 文件访问

WSL2 可以直接访问 Windows 文件系统。你的代码仓库在:
```
/mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation/
```

**不需要"传输文件"** —— WSL2 自动挂载所有 Windows 驱动器到 `/mnt/`。

但是，WSL2 访问 Windows 文件系统（/mnt/c/）的 I/O 性能较差。JSBSim 需要频繁读取 F-16
配置文件（每次 reset 都要初始化），建议将代码复制到 WSL2 原生 ext4 文件系统:

```bash
cp -r /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation ~/jsbsim-marl-formation
cd ~/jsbsim-marl-formation
dos2unix scripts/setup_wsl2.sh   # 如果还没执行过
bash scripts/setup_wsl2.sh
```

### Step 4: 验证安装 + 启动训练

```bash
cd ~/jsbsim-marl-formation
source jsbsim_rl/bin/activate
python scripts/verify_installation.py
```

直接运行你现有的训练脚本（不需要 Ray/RLlib）:

```bash
# 双 Actor + 协作模式 (当前最佳配置):
python scripts/train_dual_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000

# 或单 Actor 两阶段:
python scripts/train_attention_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000
```

---

## 注意事项

### WSL2 内存配置

在 Windows 用户目录 (`C:\Users\Sean\`) 创建 `.wslconfig` 文件。
**分配给 WSL2 的内存不要超过物理总内存的 70%**，否则 Windows 宿主机会卡死。

例如你的电脑有 16GB 物理内存:
```
[wsl2]
memory=10GB
processors=6
```

如果你的电脑有 32GB:
```
[wsl2]
memory=22GB
processors=12
```

修改 `.wslconfig` 后需要在 PowerShell 中执行 `wsl --shutdown` 然后重新打开 WSL2 才能生效。

### JSBSim 数据文件

JSBSim 需要 `data/jsbsim/` 目录中的 F-16 配置文件和初始化脚本。
确保这些文件在 WSL2 中可访问。如果是从 Windows 侧复制过来的，路径应该已经正确。

### Python 版本

当前环境使用 Python 3.10。`setup_wsl2.sh` 会自动安装。
WSL2 Ubuntu 22.04 默认不带 Python 3.10，脚本会通过 `apt` 安装。

### 性能预期

- CPU 模式: JSBSim F-16 六自由度物理仿真，性能与 Windows 原生 Python 基本持平
- GPU 模式: 如果 PyTorch 检测到 CUDA，训练中的矩阵运算会自动加速（但 JSBSim 仿真仍在 CPU 上）
- WSL2 原生文件系统（ext4）上的 I/O 显著快于 `/mnt/c/` 挂载点

### 常见问题

**Q: `import jsbsim` 失败？**
```bash
pip install jsbsim==1.3.1
```

**Q: 训练脚本找不到 `src/` 模块？**
确保在项目根目录执行脚本，`sys.path.insert(0, ...)` 会自动处理路径。

**Q: WSL2 中 TensorBoard 能访问吗？**
能。在 WSL2 中启动 `tensorboard --logdir marl_runs/xxx/tensorboard --bind_all`，
然后在 Windows 浏览器访问 `http://localhost:6006`（WSL2 自动端口转发）。
