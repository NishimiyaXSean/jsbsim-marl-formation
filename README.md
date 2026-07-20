# jsbsim-marl-formation

**Multi-agent reinforcement learning for cooperative formation pursuit**, powered by JSBSim 6-DOF F-16 flight dynamics, Self-Attention CTDE, RLlib MAPPO, and discrete tactical action primitives.

>   **Academic Research Project** — *"Token-based CTDE with Self-Attention outperforms centralized PPO on cooperative 2v1 formation pursuit."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![JSBSim](https://img.shields.io/badge/FDM-JSBSim%20F--16-orange.svg)](https://jsbsim-team.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

---

##   项目阶段性总结

### 基线设计

本项目构建了一个**基于高保真空气动力学的多智能体协同编队追击仿真与训练框架**。以 JSBSim 6-DOF F-16 飞行动力学模型为物理底座，将两架 F-16 对一架目标机的协同拦截抽象为 CTDE（中心化训练、去中心化执行）框架下的多智能体强化学习问题。

**核心技术栈**：
- **物理引擎**：JSBSim F-16 6-DOF FDM（60Hz 空气动力学仿真）
- **决策架构**：5Hz 宏观动作 → 60Hz PID 飞行控制（航向/高度/速度三通道）
- **算法**：Parameter-Shared MAPPO (CTDE) via RLlib
- **动作空间**：MultiDiscrete([5转向, 3速度]) = 15 个战术基元
- **神经网络**：Token-Based Self-Attention → FiLM 身份调制 → 离散输出头
- **观测**：39 维（自身 15 + 目标 14 + 队友 10，含广播通信与 One-hot ID）

### 核心创新与探索

1. **Token-Based Self-Attention 架构**：将 39 维观测分解为 Self/Target/Mate 三个语义 Token，通过 4 头自注意力学习动态权重分配。消融实验证明其性能超越纯 MLP 基线 5,887 分，且自发涌现角色分化（Interceptor 较 Striker 多给 Target 31% 注意力，Cohen's d = −0.53）。

2. **速度自适应转向率**：发现 OR-gate 下初始偏航角误差是 eval 方差主因（与 reward 相关系数 r = −0.35）。根据 F-16 物理 G 载荷限制，设计速度分档转向率（Slow 180m/s: ±20°/s, Cruise 250m/s: ±15°/s, Fast 320m/s: ±12°/s），将容错边界从 36° 推至 45°。

3. **FiLM 深层身份调制**：在共享 Actor 的 MLP 层间插入 γ(ID)·feat + β(ID) 调制，以恒等初始化保留 BC 预训练权重，在深层特征空间中打破对称性，让同一物理状态根据 Agent 身份产生不同的动作输出。

4. **三阶段 AND-gate 课程学习**：设计孵化器（Incubator）Stage 1——强制 pursuer 出生在 AND 包线内（spawn 1000-1400m, AND=1600m），从源头切除"长途赶路导致的尾追死锁"，专注学习包抄分离。课程通过 eval sync rate 自适应晋级。

5. **指数衰减 Pincer 塑形**：以 R = 35·min(θ, AND_angle)·exp(−d0/τ)·exp(−d1/τ) 替代硬阈值门，提供连续梯度；引入增强反摸鱼（−10/step，仅判 Slow+逃逸）、软碰撞排斥力场（capped at −50/step）、一次性 OR fallback（+1,000，不终止）等奖励函数创新。

6. **自动熔断看门狗**：内置 TrainingHealthMonitor（三段熔断：数值过载/策略僵死/协同荒漠），自动生成 Markdown 尸检报告并保存崩溃 checkpoint。

### 训练成果与现状

**OR-gate（单机追击）已完全解决**：MAPPO CTDE 在 OR-gate 下 eval 达到 +7,888，超越 SB3 集中式天花板（+5,908）33%。Self-Attention 冷启动自发学会协同追击，训练正向率 95.6%。

**AND-gate（双机协同合围）已取得关键突破但仍未完全攻克**：经过 V2→V11 共 11 个版本迭代，AND-gate 协同入线率（sync）从历史天花板 15% 提升至 V10 孵化器的 **40%**。V10 实现 eval +784（正值），sync 稳定在 20-40%，首次完成完整 500 轮训练且无熔断。然而 40% sync 仍未触发 Stage 2 晋级（需 30% MA，V10 峰值 33% MA 未持续），说明在更长距离和更严角度下维持双机协同仍存在架构性挑战。

**当前瓶颈**：参数共享 CTDE 的隐式协同（梯度冲突迫使角色分化）与独立 Actor 的显式协同（需要时序通信或深层身份解耦）之间存在张力。V6/V7 证明独立 Actor 倾向于"各自为政"（86% 脱靶率），V8-V10 证明共享 Actor 在孵化器加持下能达到 40% sync 但后续阶段晋级困难。V12 计划以 V10 全栈为基础，降低 Stage 1→2 门至 30% 并保持 entropy=0.05，验证课程晋级可行性。

---

##   Key Results

### Continuous Action Space (Box(2))

| Experiment | Architecture | Reward | Notes |
|-----------|-------------|--------|-------|
| **SB3 Centralized (ceiling)** | 66-dim shared policy | +5,908 (92% capture) | Upper bound for 2v1 |
| **Attention BC (no PPO)** | 33-dim CTDE | **+6,846** | Pure BC beats centralized PPO |
| **MAPPO OR-gate (Exp 2)** | Shared Attn CTDE | **+7,888** | 33% above SB3 ceiling |
| **MAPPO AND dynamic (Exp 3v3)** | Shared Attn CTDE | −1,171 (best) | 2000→800m annealing, +4,700 pt improvement |

### Discrete Action Space (MultiDiscrete([5,3])) — July 2026

| Experiment | Architecture | BC | Iters | Best Eval | Pos. Spikes | Key Finding |
|-----------|-------------|-----|-------|-----------|-------------|-------------|
| **Exp 4a** | MLP fallback | None | 120 | −4,542 | 0 | MLP cannot learn coordination |
| **Exp 4a-v2** | **Self-Attention** | None | 120 | **+1,345** | 1× | Self-Attn cold-start > MLP |
| **Exp 4b** | Self-Attention | Discrete BC | 120 | −1,135 | 0 | BC heads never loaded (bug) |
| **★ 4a-v2 ext** | **Self-Attention** | None | 320 | **+2,376** | **3×** | Cold-start peak; high variance |
| **★ 5a BC hotstart** | **Self-Attention** | **Discrete BC** | **500** | **+381** | **3×** | **95.6% train positive rate** |
| **Exp 5b AND-gate** | Self-Attn + Curric. | Discrete BC | 500 | +4,283 | — | AND-gate sync 55% (Stage 1 greenhouse) |
| **Exp 6a V5 full-stack** | Self-Attn + Comm + ID | Discrete BC | 500 | +178 | — | Sync 15% — CTDE implicit ceiling proved |
| **Exp 6b V6 indep. Actor** | **Independent Actors** | Discrete BC | 500 | −3,815 | — | +12,388 train rew but 0% sync — solo flight only |
| **V7 indep+高压** | Indep. Actors | Discrete BC | 225 | −8,891 | — | Collapsed: penalty overload |
| **V8 shared修复** | Shared + Loiter fix | Discrete BC | 58 | −8,223 | — | Collapsed: numerical overload |
| **V9 FiLM** | Shared + FiLM mod. | Discrete BC | 155 | −6,602 | — | FiLM prevents collapse; 0% sync |
| **★ V10 incubator** | **Shared+FiLM+Incubator** | Discrete BC | **500** | **+784** | **40% sync** 🔥 | **AND-gate sync 15%→40% breakthrough** |
| V11 lower-gate | Shared+FiLM+30%gate | Discrete BC | 225 | −1,774 | 30% | entropy=0.03 suboptimal; use 0.05 |

> **Incubator Stage 1 broke the 15% sync ceiling — from 15% to 40%.** V10 spawns pursuers inside the AND envelope (1000-1400m, AND=1600m), eliminating the long-approach tail-chase deadlock. Combined with FiLM deep identity modulation (gamma*feat+beta at MLP layer), soft collision shaping, AND-loiter penalty, and ego-centric Critic ordering. V11 confirmed entropy_coeff=0.05 is the correct exploration level for the incubator. Next: lower Stage 1→2 gate from 60% to 30% (V10 hit 33% MA) with entropy=0.05 to trigger curriculum advancement.

### Self-Attention Attention Analysis (Fig 3)

| Metric | Striker | Interceptor | Effect |
|--------|---------|-------------|--------|
| MHA Self→Target | 0.296 | **0.389** | Cohen's d = −0.53 |
| MHA Self→Mate | 0.450 | 0.439 | Both sustain ~0.44 |
| Pool Mate Weight | **0.341** | 0.298 | Δ = +0.043 |
| Mean Pincer Angle | — | — | **35.8°** (49 eps) |

> 7,858 steps per role across 49 episodes. Both agents sustain high mutual attention (~0.44), demonstrating **continuous implicit coordination** — not binary switching. This is the mathematical proof of spontaneous role differentiation from a parameter-shared network.

---

##   Architecture

```
┌─────────────────────────────────────────────────────────┐
│              RLlib MAPPO Training Pipeline               │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            shared_policy (TorchModelV2)            │   │
│  │  Self-Attention Actor + Centralized Critic         │   │
│  │  MultiDiscrete([5,3]) → turn_head(5) + speed_head(3) │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────┴───────────────────────────┐   │
│  │     FormationRllibEnv (MultiAgentEnv)              │   │
│  │  obs: Dict(obs=Box(33), global_state=Box(21),     │   │
│  │            action_mask=Box(8))                     │   │
│  │  act: MultiDiscrete([5 turn, 3 speed])             │   │
│  │  5 Hz decision rate, 12 physics sub-steps          │   │
│  └──────────────────────┬────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────┴────────────────────────────┐   │
│  │         JSBSim 6-DOF F-16 FDM (60 Hz)             │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

              Self-Attention FormationActor (TorchModelV2)
              ─────────────────────────────────────────

  obs[33] → [Self(13)] [Target(14)] [Mate(6)]
              ↓          ↓            ↓
         Linear     Linear       Linear × mate_scale
              ↓          ↓            ↓
         ┌──────────────────────────────┐
         │  Multi-Head Self-Attention   │
         │  (4 heads, d_model=128)      │
         │  + Token-Type Embedding      │
         │  + Residual + Learned Pool   │
         └──────────────┬───────────────┘
                        ↓
              MLP [256,256] → turn_head(5) + speed_head(3)
              MultiDiscrete([5,3]) = 15 tactical primitives
```

Parameter-Shared MAPPO (via RLlib `multi_agent.policies`):
  shared_policy for both p0/p1 → single Self-Attention Actor + Centralized Critic
  Eliminates IPPO non-stationarity; Self-Attention provides position-invariant role differentiation
  
  **Latest (Exp 5a, 500 iter)**: BC hotstart achieves 95.6% training reward positive rate.
  BC provides backbone (19 keys) + discrete heads (4 keys) = 23 total pretrained weights.
  Resume experiment in progress: LR=3e-4 + entropy_coeff=0.05 + relaxed penalties to push past +381 peak.

Discrete Action Space (MultiDiscrete([5,3]) = 15 tactical primitives):
  Turn:  HardLeft(−15°/s) | SoftLeft(−5°) | Straight(0°) | SoftRight(+5°) | HardRight(+15°)
  Speed: Slow(180m/s) | Cruise(250m/s) | Fast(320m/s)
  Action Masking: anti-stall (<130 m/s), ground-proximity (<200m), overspeed (>95% Vmax)
  Entropy theoretically capped at ln(15) ≈ 2.71 — hard exploration constraint

**Key innovation**: The 33-dim observation is split into three semantic tokens (Self, Target, Mate) and processed through Multi-Head Self-Attention. The Actor learns to dynamically allocate attention — attending more to the Mate token when coordination is needed, more to Target when in pursuit. Attention weights are directly interpretable for paper visualization. **The ablation study proves Self-Attention architecture is the decisive factor**: cold-start Self-Attn outperforms MLP by 5,887 pts with spontaneous role differentiation (Cohen's d = −0.53). **BC hotstart further stabilizes training** (95.6% positive rate vs ~50% cold-start), confirming the architecture is the primary enabler and BC provides stability.

---

##   Installation

Two paths depending on your operating system:

### Path A: WSL2 (Recommended — Linux performance)

See the full guide at **[docs/wsl2-setup-guide.md](docs/wsl2-setup-guide.md)**. Quick summary:

```powershell
# 1. In Windows PowerShell (Admin):
wsl --install
# → Restart computer
```

```bash
# 2. In WSL2 Ubuntu terminal:
cd ~
git clone https://github.com/NishimiyaXSean/jsbsim-marl-formation.git
cd jsbsim-marl-formation

# 3. Fix line endings (CRLF → LF)
sudo apt install dos2unix -y && dos2unix scripts/setup_wsl2.sh

# 4. Install environment
bash scripts/setup_wsl2.sh

# 5. Large files (BC data, model weights) — drag via Windows Explorer:
#    Address bar: \\wsl$\Ubuntu\home\YOUR_USER\jsbsim-marl-formation\
#    Copy from Windows project: data/expert/*.npz, benchmarks/*.pth

# 6. Activate and verify
conda activate marl_env
python scripts/verify_installation.py
```

### Path B: Windows (conda, for evaluation only)

Training on Windows is possible but **CPU-only and significantly slower** than WSL2. For evaluation and Tacview export, Windows is fine:

```bash
conda create -n jsbsim_rl python=3.10 -y
conda activate jsbsim_rl
pip install torch numpy gymnasium stable-baselines3 tensorboard matplotlib pyyaml
pip install jsbsim==1.3.1
python scripts/verify_installation.py
```

### JSBSim Aircraft Data

```bash
git clone https://github.com/JSBSim-Team/jsbsim.git /tmp/jsbsim
cp -r /tmp/jsbsim/aircraft data/jsbsim/
cp -r /tmp/jsbsim/engines  data/jsbsim/
cp -r /tmp/jsbsim/systems  data/jsbsim/
```

---

##   Quick Start

### RLlib MAPPO Training (Recommended)

```bash
# Discrete BC hotstart (Exp 5a) — Best stability, 95.6% positive train reward
python scripts/train_formation_rllib.py \
    --iterations 500 --cooperative \
    --checkpoint-freq 25 --eval-interval 25 --seed 42
    # Auto-loads discrete BC + sets lr=2e-4 (BC-protective)

# Cold-start (no BC) — Higher peak potential, higher variance
python scripts/train_formation_rllib.py \
    --iterations 320 --cooperative --no-bc \
    --checkpoint-freq 25 --seed 42
    # Defaults to lr=3e-4 (standard PPO exploration)

# Resume from best checkpoint with exploration boost
python scripts/train_formation_rllib.py \
    --iterations 200 --cooperative \
    --resume-from PATH_TO_BEST_CHECKPOINT \
    --lr 3e-4 --entropy-coeff 0.05 \
    --checkpoint-freq 25 --eval-interval 25 --seed 42

# Experiment 3: Two-phase OR→AND (full cooperative curriculum)
python scripts/train_formation_rllib.py \
    --iterations 500 --cooperative --warmup 200000 \
    --checkpoint-freq 25 --seed 42
```

### Evaluate the SB3 Baseline

```bash
# Benchmark the centralized ceiling model (100 episodes × 3 difficulties)
python scripts/benchmark_sb3_baseline.py -n 100 -d 0.0,0.3,0.6
```

### Legacy: Custom PyTorch MAPPO Training (Archived)

```bash
# 1v1 BC pretraining + MAPPO fine-tuning
python scripts/train_attention_actor.py --mode 1v1 --steps 200000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth

# 2v1 cooperative two-phase (OR-gate → AND-gate)
python scripts/train_attention_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000

# Dual-Actor decoupled (P0/P1 independent networks)
python scripts/train_dual_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --cooperative --warmup 100000
```

### Diagnose Cooperative Behavior

```bash
# Per-step AND-gate diagnostics + Tacview export
python scripts/diagnose_coop_tacview.py --episodes 3

# Evasive target maneuvers (spiral, lissajous, weave)
python scripts/diagnose_dual_evasion.py --pattern all --episodes 5
```

### Generate BC Training Data

```bash
# PID-based cooperative trajectory generator (clean data, no free-riding)
python scripts/generate_coop_expert.py --episodes 500

# BC pretrain on filtered data
python scripts/train_attention_bc_2v1.py --train --epochs 80
```

---

##   Training Call Chain

```
train_formation_rllib.py          ← V10 training entry (shared MAPPO + curriculum)
  │
  ├─ src/environment/formation_rllib_env.py  ← ★ MultiAgentEnv: obs, reward, AND-gate
  │    ├─ src/dynamics/aircraft.py           ← JSBSim F-16 6-DOF FDM wrapper
  │    ├─ src/dynamics/flight_controller.py  ← PID control (heading/altitude/speed)
  │    ├─ src/dynamics/flight_envelope.py    ← V-n diagram, stall/overspeed limits
  │    ├─ src/dynamics/autopilot.py          ← BFMAutopilot trim + gain scheduling
  │    ├─ src/utils/geometry.py              ← Tactical angles (ATA/AA/HCA/pincer)
  │    └─ src/utils/units.py                 ← kts ↔ m/s, ft ↔ m conversions
  │
  └─ src/models/formation_rllib_model.py     ← RLlib TorchModelV2 (CTDE Actor+Critic)
       └─ src/models/attention_actor.py      ← Self-Attention + FiLM + token projection
```

| 层 | 文件 | 行数 | 功能 |
|------|------|:--:|------|
| **入口** | `scripts/train_formation_rllib.py` | ~730 | PPO 配置, BC 加载, 3-stage 课程调度, 健康看门狗 |
| **环境** | `src/environment/formation_rllib_env.py` | ~1100 | 2v1 MultiAgentEnv, obs(39d), 奖励(progress/pincer/OOC/loiter), AND-gate, 碰撞, 终止, 课程参数热更新 |
| **模型** | `src/models/formation_rllib_model.py` | ~180 | RLlib TorchModelV2: forward(), value_function(), action masking |
| **模型** | `src/models/attention_actor.py` | ~320 | Self-Attention(4 heads), Token projection(Self/Target/Mate), FiLM modulation, Learned pooling, MLP head |
| **动力学** | `src/dynamics/aircraft.py` | ~450 | JSBSim FGFDMExec wrapper: trim, state IO, control surfaces |
| **动力学** | `src/dynamics/flight_controller.py` | ~300 | PID heading/altitude/speed tracking from RL action targets |
| **动力学** | `src/dynamics/flight_envelope.py` | ~150 | Stall warning, overspeed, ground proximity detection |
| **动力学** | `src/dynamics/autopilot.py` | ~500 | BFMAutopilot: trim schedules, gain scheduling, λ-G control law |
| **工具** | `src/utils/geometry.py` | ~80 | ATA, AA, HCA, LOS, pincer angle computation |
| **工具** | `src/utils/units.py` | ~30 | kts↔m/s, ft↔m, imperial↔metric |

##   Project Structure

```
jsbsim-marl-formation/
├── src/
│   ├── dynamics/              # JSBSim F-16 physics + flight control
│   │   ├── aircraft.py        #   Aircraft — JSBSim FGFDMExec wrapper
│   │   ├── autopilot.py       #   BFMAutopilot (λ-G), PIDController, GainScheduler
│   │   ├── bfm_actions.py     #   Basic Fighter Maneuvers action primitives
│   │   ├── flight_controller.py # Stabilized FlightController (heading/alt/speed PID)
│   │   └── flight_envelope.py #   V-n diagram, GPWS, stall/overspeed limits
│   │
│   ├── environment/           # RLlib MultiAgentEnv
│   │   └── formation_rllib_env.py # ★ 2v1 CTDE: obs, rewards, AND-gate, curriculum
│   │
│   ├── models/                # Self-Attention + FiLM neural architectures
│   │   ├── attention_actor.py #   ★ AttentionFormationActor + AttentionCritic
│   │   └── formation_rllib_model.py # ★ RLlib TorchModelV2 wrapper
│   │
│   ├── utils/                 # Math + geometry
│   │   ├── geometry.py        #   Tactical angles: ATA, AA, HCA, LOS, closing speed
│   │   ├── kinematics.py      #   NED↔WGS-84 coordinate transforms
│   │   ├── pn_guidance.py     #   Proportional Navigation with bearing bias
│   │   └── units.py           #   Imperial ↔ SI unit conversions
│   │
│   └── logging/               # Tacview ACMI telemetry export
│       └── tacview_exporter.py
│
├── scripts/                   # ★ Active scripts (13 total)
│   │
│   │   # ── Primary training ───────────────────────────────────────
│   ├── train_formation_rllib.py  # ★ V10 MAPPO cooperative training
│   ├── train_discrete_bc.py   #   Discrete BC pretraining
│   │
│   │   # ── Visualization ──────────────────────────────────────────
│   ├── collect_viz_data.py    #   Trajectory + attention weight collection
│   ├── viz_paper_figures.py   #   Fig 1 (3D trajectory) + Fig 2 (attention timeline)
│   ├── viz_fig3_role_attention.py # Fig 3 (role-grouped attention matrix)
│   ├── viz_trajectory_comparison.py # Multi-checkpoint 3D trajectory comparison
│   ├── generate_paper_charts.py   # 5 paper-quality dataviz charts
│   │
│   │   # ── Analysis ───────────────────────────────────────────────
│   ├── analyze_eval_statistics.py  # Per-episode statistical autopsy
│   ├── analyze_initial_state_sensitivity.py # 100-ep initial-condition sensitivity
│   ├── diagnose_v6_autopsy.py      # V6 reward+termination per-episode analysis
│   ├── benchmark_sb3_baseline.py   # SB3 centralized-policy baseline
│   │
│   │   # ── Utilities ──────────────────────────────────────────────
│   ├── verify_installation.py      # 4-step installation verification
│   └── setup_wsl2.sh               # WSL2 one-command environment setup
│
├── benchmarks/
│   ├── sb3_2v1_97p3/          # SB3 Phase 4.1 centralized ceiling archive
│   │   ├── README.md          #   Benchmark documentation
│   │   ├── model.zip          #   Archived model weights
│   │   └── metrics_diff*.json #   Wilson CI × 3 difficulties
│   └── dual_actor_coop_best.pth   # ★ Best dual-actor checkpoint
│
├── configs/                   # YAML configs (env + model)
├── data/
│   ├── expert/                # BC pretraining data
│   │   ├── attention_bc_2v1_filtered.npz      # Filtered 2v1 data (18K samples)
│   │   ├── attention_bc_2v1_filtered_pretrained.pth  # Filtered BC model
│   │   ├── coop_pid_data.npz                  # PID cooperative data (10K samples)
│   │   └── tiled_2v1_phase36.zip              # SB3 Phase 3.6 tiled weights
│   └── jsbsim/                # JSBSim aircraft/engines/systems data
│
├── docs/                      # Daily work summaries (Jul 1-16)
│   ├── wsl2-setup-guide.md    # ★ WSL2 deployment checklist
│   ├── 2026-07-09-full-summary.md  # Continuous→Discrete migration
│   ├── 2026-07-10-full-summary.md  # BC bug fix + 500-iter BC hotstart
│   ├── 2026-07-13-full-summary.md  # Sensitivity analysis + Env-V2
│   ├── 2026-07-14-full-summary.md  # AND-gate CTDE ceiling + V5 full-stack
│   ├── 2026-07-15-full-summary.md  # V6 autopsy → V7 launch
│   └── 2026-07-16-full-summary.md  # ★ V7→V10: incubator breaks 15%→40% sync
│
├── results/                   # Evaluation outputs
│   └── evasion_diag/          # Evasive maneuver Tacview files (80 episodes)
│
├── marl_runs/                 # Training outputs (git-ignored)
├── tests/                     # Unit tests (dynamics, environment, models)
├── pyproject.toml             # Poetry project config (reference only)
├── CLAUDE.md                  # Claude Code agent instructions
└── README.md                  # This file
```

---

##   Environments

### FormationRllibEnv — Cooperative 2v1 Pursuit (RLlib MultiAgentEnv)

The primary environment for cooperative formation research, now running under RLlib.

- **Agent IDs**: `"p0"`, `"p1"` (independent policy instances)
- **Pursuers**: 2 × F-16 (JSBSim 6-DOF)
- **Target**: 1 × F-16 (scripted straight/evasive)
- **Observation**: `Dict({"obs": Box(33), "global_state": Box(21)})` — per-agent local + global state
- **Action**: `MultiDiscrete([5, 3])` — 15 tactical primitives with action masking
- **Decision Rate**: 5 Hz (0.2 s per macro-action, 12 physics sub-steps at 60 Hz)
- **Cooperative Mode** (`cooperative_mode=True`):
  - **Phase 1 [OR]**: Any pursuer < 200m → success (+5,000) + light pincer guidance
  - **Phase 2 [AND]**: Both < 800m + pincer > 30° + sustained 6 steps → cooperative_success
  - **Pincer Reward**: Bonus for 60°–150° between LOS vectors
  - **Dynamic Roles**: Striker (closer pursuer, tracking ×1.5) + Interceptor (further, pincer ×2.0)
  - **Asymmetric Resets**: 70% probability, random pursuer starts 1,500m behind + facing away
- **Shaping Penalties**:
  - Distance Asymmetry: penalizes free-riding when |d0−d1| > 800m (weight 0.3)
  - Time-Sync Pacing: prevents Striker from rushing ahead alone (weight 0.5)

### FormationEnv — Cooperative 2v1 Pursuit (SB3/Legacy)

The SB3-compatible environment (shared-policy interface).

- **Observation**: 66-dim concatenated (2 × 33-dim per-pursuer)
- **Action**: Box(4) — `[turn_0, speed_0, turn_1, speed_1]`
- **Cooperative Mode**: Same Phase 5 logic as above, shared-policy execution

### SinglePursuitEnv — Continuous 1v1 Pursuit

Single-agent pursuit with 3D continuous action via FlightController.

- **Action**: Box(3) `[d_heading, d_alt, d_speed]`
- **Observation**: 25-dim (body-frame relative state + tactical angles + energy features)
- **Auto-Curriculum**: Continuous difficulty [0, 1] with cliff-collapse rollback

---

##   Key Architecture Details

### Per-Pursuer Observation (33-dim)

| Indices | Content |
|---------|---------|
| 0–2 | Target relative position (body frame) |
| 3–5 | Own velocity (body frame) |
| 6–8 | Own attitude (RPY) |
| 9–11 | Own angular velocity |
| 12 | Own height |
| 13–15 | Target velocity (body frame) |
| 19–21 | Tactical geometry: cos(ATA), cos(AA), cos(HCA) |
| 22 | Angle of Attack |
| 23 | Airspeed |
| 25 | LOS angular rate |
| 26 | Bearing error |
| **27–29** | **Mate relative position** (body frame) |
| **30–32** | **Mate relative velocity** (body frame) |

### Cooperative Reward Components

| Component | Formula | Purpose |
|-----------|---------|---------|
| Progress | 1.5 × Δdist | Closing distance to target |
| ATA alignment | 8.0 × cos(ATA) × dist_factor | Nose-on-target |
| Pincer angle | 15.0 × (angle/150) when 60°–150° | Flanking geometry |
| Striker bonus | 1.5 × tracking reward | Closer pursuer hunts |
| Interceptor bonus | 2.0 × pincer reward | Further pursuer flanks |
| AND-gate success | +5,000 + 2,000 × (angle/180) | Both in position |
| Proximity tiers | +25/+50/+100 at 800/500/300m | Approach milestones |
| Dist Asymmetry Penalty | −0.3 × (|d0−d1| − 800) / 1000 × dt | Prevents free-riding |
| Sync Pacing Penalty | −0.5 × (d1−d0) / 1000 | Prevents Striker rushing ahead |

### Training Hyperparameters (RLlib MAPPO)

| Parameter | Value | Notes |
|-----------|-------|-------|
| γ (gamma) | 0.99 | Discount factor |
| λ (GAE) | 0.95 | GAE trace decay |
| ε (clip) | 0.2 | PPO clip range |
| Train batch | 8,192 | Total rollout per iteration |
| Mini-batch | 1,024 | Per update |
| PPO epochs | 10 | Per rollout |
| LR (BC hotstart) | **2e-4** | Protects BC features (0.67× cold-start) |
| LR (cold-start) | 3e-4 | Standard PPO rate |
| LR (resume/explore) | 3e-4 | Higher exploration with entropy boost |
| Entropy coeff | 0.03–0.05 | 0.05 paired with higher LR for exploration |
| Grad clip | 0.5 | Max gradient norm |
| VF clip | 1,000 | Value function clipping |

---

##   WSL2 Deployment

For full Linux training performance, see the detailed 4-phase guide:

**[  docs/wsl2-setup-guide.md](docs/wsl2-setup-guide.md)**

Quick checklist:

- [ ] `wsl --install` (PowerShell Admin) → reboot
- [ ] Create `.wslconfig` (memory ≤ 70% physical RAM)
- [ ] `nvidia-smi` (verify GPU passthrough)
- [ ] `git clone` into `~/` (Linux native ext4, NOT `/mnt/c/`)
- [ ] `dos2unix scripts/setup_wsl2.sh` (CRLF → LF)
- [ ] `bash scripts/setup_wsl2.sh` (Miniconda + PyTorch + JSBSim)
- [ ] VS Code WSL plugin → "Connect to WSL" → Open Folder
- [ ] `conda activate marl_env && python scripts/verify_installation.py`

>   **RLlib is the primary training framework** (migrated 2026-07-07). **Discrete actions are default** (MultiDiscrete[5,3]). **V10 incubator is the current SOTA**: AND-gate sync 40% via shared Actor + FiLM identity modulation + ego-centric Critic + exponential pincer decay + AND-loiter penalty + soft collision shaping + 3-stage curriculum. The Self-Attention Actor is hosted inside RLlib's `TorchModelV2` API. `pyproject.toml` is outdated — use `scripts/setup_wsl2.sh` or conda.

---

##   Dependencies

| Package | Purpose |
|---------|---------|
| `jsbsim ~=1.3` | 6-DOF F-16 flight dynamics |
| `torch >=2.0` | Neural networks (Self-Attention, MHA) |
| `ray[rllib] >=2.40` | Multi-agent MAPPO training pipeline |
| `gymnasium ~=1.2` | RL environment interface |
| `stable-baselines3` | SB3 baseline (PPO, evaluation only) |
| `numpy` | Numerical computation |
| `tensorboard` | Training monitoring |
| `matplotlib` | Trajectory plots |
| `pyyaml` | Config file parsing |

---

##   Technical Roadmap

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCENARIO LAYER                            │
│  FormationEnv → FormationRllibEnv (MultiAgentEnv)            │
│  2v1 cooperative pursuit, NvM extensible                     │
│  Cooperative modes: OR-gate → AND-gate (curriculum)          │
│  Evasive targets: straight, spiral, lissajous, weave         │
├─────────────────────────────────────────────────────────────┤
│                   ALGORITHM LAYER                            │
│  Parameter-Shared MAPPO (CTDE)                               │
│  Self-Attention Actor: 33-dim → 3 tokens → MHA → action      │
│  Tokenized Attention Critic: 21-dim → 3-entity Attn → value  │
│  Discrete: MultiDiscrete([5,3]) + Categorical heads + masks  │
│  Continuous: Box(2) + DiagGaussian (archived)                │
├─────────────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE LAYER                         │
│  JSBSim 6-DOF F-16 FDM → FlightController (60 Hz PID)       │
│  RLlib MAPPO (Ray 2.40) → TorchModelV2 → shared_policy      │
│  WSL2 + CUDA GPU passthrough                                 │
│  Tacview ACMI export + TensorBoard + Matplotlib              │
└─────────────────────────────────────────────────────────────┘
```

### Evolution Path

```
Phase 1 (Jun):       SB3 shared-policy baseline → 97.3% ceiling sealed
Phase 2 (Jul 1-3):   Self-Attention CTDE → BC beats centralized PPO (+6,846)
Phase 3 (Jul 7):     RLlib migration → IPPO → Parameter-Shared MAPPO
Phase 4 (Jul 8):     OR-gate convergence (+7,888), AND-gate dynamic annealing (-1,171)
Phase 5 (Jul 9):     Continuous → Discrete MultiDiscrete([5,3]) + Action Masking
Phase 6 (Jul 9-10):  Self-Attn cold-start → ablation matrix → paper PPT
Phase 7 (Jul 10):    BC bug fix → 500-iter BC hotstart (95.6% positive)
Phase 8 (Jul 13):    Sensitivity analysis → speed-dependent turns → Env-V2 (+3,421)
Phase 9 (Jul 14):    AND-gate curriculum → reward rebalance → CTDE ceiling at 15% sync
Phase 10 (Jul 15):   V5 full-stack → V6 indep. actors → V7 high-pressure collapse
Phase 11 (Jul 16):   V8 shared repair → V9 FiLM modulation → V10 incubator sync 40% 🔥
Phase 12 (Jul 17):   V11 gate lowering + entropy lesson → V12 pending (30% gate + ent=0.05)
```

### Current Optimization Priorities

1. **AND-gate sync rate breakthrough**: 15% ceiling across 6 experiment versions. V7 (exponential pincer + anti-loiter + soft collision) aims to push past this.
2. **Anti-spectator enforcement**: V6 autopsy revealed P1's 79% Slow + 44% SoftLeft loiter strategy. Enhanced −10/step penalty in V7 directly targets this.
3. **Collision shaping**: Soft repulsive force replacing −3000+termination cliff — allows safe-proximity learning.
4. **Paper writing**: 10 experiment versions, sensitivity analysis, trajectory comparisons, and CTDE ceiling proof ready for manuscript.

---

##   Citation

```bibtex
@software{nishimiya2026jsbsim,
  author       = {Sean Nishimiya},
  title        = {jsbsim-marl-formation: Multi-Agent RL for Cooperative
                  Formation Flight with JSBSim F-16 Dynamics},
  year         = 2026,
  affiliation  = {Zhejiang University},
  url          = {https://github.com/NishimiyaXSean/jsbsim-marl-formation}
}
```

---

##   License

MIT — see [LICENSE](LICENSE) for details.
