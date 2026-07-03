# WSL2 MAPPO 训练环境搭建指南

> 目标: 在 WSL2 Ubuntu 中运行 RLlib MAPPO 多智能体强化学习训练

---

## 你需要自己完成的操作（需要 Windows 管理员权限 + 重启）

### Step 1: 启用 WSL2 (一次性)

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

### Step 2: 安装 GPU 驱动（如果有 NVIDIA GPU）

在 Windows 端下载安装 NVIDIA WSL2 驱动:
https://developer.nvidia.com/cuda/wsl

不需要在 WSL 内部安装 CUDA toolkit —— 驱动会自动穿透。

---

## 我可以帮你准备的操作

### Step 3: WSL2 环境初始化脚本

以下脚本 `scripts/setup_wsl2.sh` 已准备好，进入 WSL2 后直接运行:

```bash
# 在 WSL2 Ubuntu 中执行:
cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation
bash scripts/setup_wsl2.sh
```

### Step 4: 文件访问

WSL2 可以直接访问 Windows 文件系统。你的代码仓库在:
```
/mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation/
```

**不需要"传输文件"** —— WSL2 自动挂载所有 Windows 驱动器到 `/mnt/`。

但是，WSL2 访问 Windows 文件系统（/mnt/c/）的 I/O 性能较差。对于 JSBSim 这种需要频繁读取飞机配置文件的场景，建议将代码复制到 WSL2 原生文件系统:

```bash
# 在 WSL2 中执行:
cp -r /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation ~/jsbsim-marl-formation
cd ~/jsbsim-marl-formation
bash scripts/setup_wsl2.sh
```

### Step 5: 验证安装

```bash
# 在 WSL2 中执行:
cd ~/jsbsim-marl-formation
python scripts/verify_installation.py

# 测试 RLlib 能否正常启动:
python -c "import ray; ray.init(); print('Ray OK'); ray.shutdown()"
```

### Step 6: 启动 MAPPO 训练

```bash
# 2v1 编队追猎 (RLlib MAPPO):
python scripts/train_formation_mappo.py

# 或在 Python 中直接运行:
python -c "
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.environment.formation_mappo_env import FormationMAPPOEnv

def env_creator(cfg):
    return FormationMAPPOEnv(cfg)

register_env('formation_2v1', env_creator)

config = (
    PPOConfig()
    .environment('formation_2v1')
    .framework('torch')
    .training(
        lr=3e-4,
        train_batch_size=8192,
        sgd_minibatch_size=1024,
        num_sgd_iter=10,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        grad_clip=0.5,
    )
    .resources(num_gpus=1)
)

algo = config.build()
for i in range(500):
    result = algo.train()
    print(f'Iter {i}: reward={result[\"episode_reward_mean\"]:.1f}')
    if i % 50 == 0:
        algo.save(f'./marl_runs/rllib_2v1/')
"
```

---

## 注意事项

### JSBSim 数据文件

JSBSim 需要 `data/jsbsim/` 目录中的 F-16 配置文件和初始化脚本。确保这些文件在 WSL2 中可访问。

当前代码使用 `jsbsim_data_dir` 参数来指定路径。在 `FormationMAPPOEnv` 初始化时传入:
```python
env = FormationMAPPOEnv({"jsbsim_data_dir": "data/jsbsim"})
```

### Ray 版本

`pyproject.toml` 中声明了 `ray[rllib] ^2.40`。在 WSL2 Ubuntu 中安装:
```bash
pip install "ray[rllib]==2.40.0"
```

### 性能优化

- WSL2 内存限制: 在 Windows 用户目录创建 `.wslconfig`:
  ```
  [wsl2]
  memory=16GB
  processors=8
  ```
- JSBSim 使用 CPU 物理仿真，不需要 GPU
- RLlib 的训练批次可以在 GPU 上运行（如果有 NVIDIA GPU）

### 调试

如果 Ray 启动失败:
```bash
# 检查 Ray 状态
ray status

# 查看 Ray 日志
cat /tmp/ray/session_latest/logs/raylet.out
```
