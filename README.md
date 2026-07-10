# jsbsim-marl-formation

**Multi-agent reinforcement learning for cooperative formation pursuit**, powered by JSBSim 6-DOF F-16 flight dynamics, Self-Attention CTDE, RLlib MAPPO, and discrete tactical action primitives.

>   **Academic Research Project** — *"Token-based CTDE with Self-Attention outperforms centralized PPO on cooperative 2v1 formation pursuit."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![JSBSim](https://img.shields.io/badge/FDM-JSBSim%20F--16-orange.svg)](https://jsbsim-team.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

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
| **Exp 4b** | Self-Attention | Discrete BC | 120 | −1,135 | 0 | BC stabilizes, no extra peak |
| **★ 4a-v2 ext** | **Self-Attention** | None | **320** | **+2,376** | **3×** | 320-iter convergence proof |

> **Self-Attention is the decisive factor.** Cold-start Self-Attention (4a-v2) outperforms MLP baseline (4a) by **5,887 reward points**. At 320 iterations, Self-Attention achieves **3 eval-positive spikes** (+1,345, +2,376, +4.0) with zero expert knowledge — proving token-based architecture spontaneously learns cooperative pursuit through physical interaction alone. **Structure determines the ceiling.**

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
              MLP [256,256] → action_mean(2)
```

Parameter-Shared MAPPO (via RLlib `multi_agent.policies`):
  shared_policy for both p0/p1 → single Self-Attention Actor + Centralized Critic
  Eliminates IPPO non-stationarity; Self-Attention provides position-invariant role differentiation
  Verified: cold-start Self-Attention achieves 3 eval-positive spikes (+2,376 peak)

Discrete Action Space (MultiDiscrete([5,3]) = 15 tactical primitives):
  Turn:  HardLeft(−15°/s) | SoftLeft(−5°) | Straight(0°) | SoftRight(+5°) | HardRight(+15°)
  Speed: Slow(180m/s) | Cruise(250m/s) | Fast(320m/s)
  Action Masking: anti-stall (<130 m/s), ground-proximity (<200m), overspeed (>95% Vmax)
  Entropy theoretically capped at ln(15) ≈ 2.71 — hard exploration constraint

**Key innovation**: The 33-dim observation is split into three semantic tokens (Self, Target, Mate) and processed through Multi-Head Self-Attention. The Actor learns to dynamically allocate attention — attending more to the Mate token when coordination is needed, more to Target when in pursuit. Attention weights are directly interpretable for paper visualization. **The ablation study proves Self-Attention architecture is the decisive factor**: cold-start Self-Attn outperforms MLP by 5,887 pts and achieves spontaneous role differentiation (Cohen's d = −0.53).

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
# Experiment 1: Non-cooperative baseline (symmetry involution)
python scripts/train_formation_rllib.py \
    --iterations 200 --no-cooperative \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth \
    --eval-interval 20 --seed 42

# Experiment 2: OR-gate cooperative warmup
python scripts/train_formation_rllib.py \
    --iterations 120 --cooperative \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth

# Experiment 3: Two-phase OR→AND (full cooperative curriculum)
python scripts/train_formation_rllib.py \
    --iterations 500 --cooperative --warmup 200000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth
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

##   Project Structure

```
jsbsim-marl-formation/
├── src/
│   ├── dynamics/              # JSBSim F-16 wrapper + flight control
│   │   ├── aircraft.py        #   Aircraft — wraps JSBSim FGFDMExec
│   │   ├── autopilot.py       #   BFMAutopilot (λ-G), PIDController, GainScheduler
│   │   ├── flight_controller.py # Stabilized FlightController (heading/alt/speed)
│   │   └── flight_envelope.py #   V-n diagram, GPWS, stall/overspeed limits
│   │
│   ├── environment/           # Gymnasium environments
│   │   ├── formation_env.py   #   FormationEnv — NvM cooperative pursuit (Phase 5)
│   │   ├── formation_rllib_env.py # ★ RLlib MultiAgentEnv wrapper (2v1 CTDE, Phase 5)
│   │   ├── formation_mappo_env.py # RLlib MultiAgentEnv wrapper (legacy MLP)
│   │   ├── single_pursuit_env.py  # SinglePursuitEnv — 3D continuous pursuit (25-dim)
│   │   ├── continuous_pursuit_env.py # ContinuousPursuitEnv (27-dim obs)
│   │   ├── air_combat_env.py  #   AirCombatEnv — 1v1 adversarial combat
│   │   ├── observations.py    #   19-dim local + 26-dim global observation builders
│   │   ├── rewards.py         #   RewardConfig: progress, ATA, pincer, proximity tiers
│   │   ├── termination.py     #   Collision, CPA, ground/OOB/timeout checks
│   │   ├── curriculum.py      #   3-stage curriculum + auto-advancement
│   │   └── ablation_wrappers.py # FrameStack, CubicAction, LeadPursuitReward
│   │
│   ├── models/                # Neural network architectures
│   │   ├── attention_actor.py #   ★ AttentionFormationActor + Tokenized AttentionCritic
│   │   ├── formation_rllib_model.py # ★ RLlib TorchModelV2 — Self-Attention Actor+Critic
│   │   ├── formation_mappo_model.py # RLlib CTDE model (legacy MLP)
│   │   ├── mappo_model.py     #   RLlib 1v1 model (legacy MLP)
│   │   └── tianshou_networks.py   # Pure PyTorch Actor/Critic for Tianshou MAPPO
│   │
│   ├── training/              # RLlib pipelines (legacy — RLlib not recommended)
│   │   ├── train_mappo.py     #   RLlib MAPPO 1v1 training (3-stage curriculum)
│   │   ├── callbacks.py       #   AirCombatCallbacks — kill/crash/OOB tracking
│   │   └── baselines.py       #   Random agent + pure pursuit baseline
│   │
│   ├── utils/                 # Math + geometry + guidance
│   │   ├── geometry.py        #   Tactical angles: ATA, AA, HCA, LOS, closing speed
│   │   ├── kinematics.py      #   NED↔WGS-84 coordinate transforms
│   │   ├── pn_guidance.py     #   Proportional Navigation with bearing bias
│   │   └── units.py           #   Imperial ↔ SI unit conversions
│   │
│   └── logging/               # Tacview ACMI telemetry export
│       └── tacview_exporter.py
│
├── scripts/                   # ★ Entry-point scripts (30+ total)
│   ├── train_formation_rllib.py  # ★ RLlib Dual-Actor MAPPO (current primary pipeline)
│   ├── train_dual_actor.py    #   Dual-Actor MAPPO (legacy — decoupled P0/P1)
│   ├── train_attention_actor.py  # Attention Actor MAPPO (legacy — EV-gated + KL)
│   ├── train_attention_bc_2v1.py # 2v1 BC pipeline (data collection + pretraining)
│   ├── train_attention_bc.py  #   1v1 BC pipeline
│   ├── generate_coop_expert.py   # PID cooperative trajectory generator
│   ├── diagnose_coop_tacview.py  # Cooperative model diagnostic + Tacview
│   ├── diagnose_dual_evasion.py  # Evasive maneuver diagnostic (4 patterns)
│   ├── benchmark_sb3_baseline.py # SB3 baseline evaluation (Wilson CI + Tacview)
│   ├── train_formation_2v1.py    # SB3 2v1 shared-policy training (Phase 4)
│   ├── train_single_pursuit.py   # SB3 PPO single-pursuit (auto-curriculum)
│   ├── evaluate_and_visualize.py # Full eval: Tacview + 3D plots + Wilson CI
│   ├── quick_tacview.py       #   Quick Tacview export from any model
│   ├── verify_installation.py #   4-step installation verification
│   ├── setup_wsl2.sh          #   ★ WSL2 one-command environment setup
│   └── ...
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
├── docs/                      # Daily work summaries + design docs + WSL2 guide
│   ├── wsl2-setup-guide.md    # ★ WSL2 4-phase deployment checklist
│   ├── 2026-07-01-full-summary.md
│   ├── 2026-07-02-full-summary.md
│   └── 2026-07-03-full-summary.md
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
- **Action**: `Box(2)` — `[turn, speed]` → FlightController
- **Decision Rate**: 2 Hz (0.5 s per macro-action, 30 physics sub-steps at 60 Hz)
- **Cooperative Mode** (`cooperative_mode=True`):
  - **Phase 1 [OR]**: Any pursuer < 200m → success (+5,000) + light pincer guidance
  - **Phase 2 [AND]**: Both < 800m + pincer > 30° + sustained 6 steps → cooperative_success
  - **Pincer Reward**: Bonus for 60°–150° between LOS vectors
  - **Dynamic Roles**: Striker (closer pursuer, tracking ×1.5) + Interceptor (further, pincer ×2.0)
  - **Asymmetric Resets**: 70% probability, random pursuer starts 1,500m behind + facing away
- **Action Clipping**: DiagGaussian bounds enforcement to prevent NaN from unbounded sampling

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

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| γ (gamma) | 0.99 | Discount factor |
| λ (GAE) | 0.95 | GAE trace decay |
| ε (clip) | 0.2 | PPO clip range |
| Rollout steps | 4,096 | Per epoch |
| Mini-batch | 128 | Per update |
| PPO epochs | 10 | Per rollout |
| Actor LR (fine) | 1e-5 | After warmup |
| Critic LR (warmup) | 1e-3 | Attention Critic |
| EV unfreeze gate | 0.3 | Explained variance threshold |
| KL target | 0.015 | Early stopping per minibatch |
| ENT_COEF | 0.005 | With auto-boost on collapse |

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

>   **RLlib is the primary training framework** (migrated 2026-07-07). **Discrete actions are now default** (MultiDiscrete[5,3], migrated 2026-07-09). The Self-Attention Actor is hosted inside RLlib's `TorchModelV2` API. `pyproject.toml` is outdated — use `scripts/setup_wsl2.sh` or conda.

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
Phase 1 (Jun):     SB3 shared-policy baseline → 97.3% ceiling sealed
Phase 2 (Jul 1-3): Self-Attention CTDE → BC beats centralized PPO (+6,846)
Phase 3 (Jul 7):   RLlib migration → IPPO → Parameter-Shared MAPPO
Phase 4 (Jul 8):   OR-gate convergence (+7,888), AND-gate dynamic annealing (-1,171)
Phase 5 (Jul 9):   Continuous → Discrete MultiDiscrete([5,3]) + Action Masking
Phase 6 (Jul 9-10): Self-Attn cold-start → 3 eval positives (+2,376 peak), ablation matrix, paper PPT
```

### Current Optimization Priorities

1. **Discrete BC pretraining enhancement**: turn_acc=72%, speed_acc=81% — improve BC quality to close cold-start gap
2. **AND-gate sync-entry**: 0% sync rate remains the fundamental bottleneck; explore explicit ΔTGO signal
3. **N×M scaling**: MultiDiscrete + parameter sharing naturally supports >2 pursuer scenarios
4. **Self-Play / League Training**: move beyond scripted targets to adversarial co-evolution

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
