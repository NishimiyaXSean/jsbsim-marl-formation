# CLAUDE.md - JSBSim MARL Research Core Guidance

## 🔴 CRITICAL RULES (MUST OBEY)
1. **Virtual Environment**: All scripts, commands, and tests MUST be executed inside the `jsbsim_rl` virtual environment. Always ensure it is activated before running tasks.
2. **Task Initiation Protocol**: BEFORE starting any code modification, feature addition, or file refactoring, you MUST invoke the `superpower` skill workflow (`/superpowers:brainstorm`) to establish the design spec, architectural alignment, and TDD plan.

## 🎓 ULTIMATE GOAL: ACADEMIC PAPER SUBMISSION
- **Core Research Objective**: Build a scalable framework for a high-quality academic paper on **"Multi-agent reinforcement learning for formation cooperative route decision and planning"**.
- **Physics Engine**: Migrate fully from PyBullet to **JSBSim 6-DOF** F-16 flight dynamics model (FDM) to achieve extreme aerodynamic fidelity.
- **Algorithm & Scaling**: Transition from discrete BFM pursuit baselines to continuous multi-agent algorithms (MAPPO/PPO), advancing from 1v1 single pursuit to NvM (e.g., 2v2) collaborative formation training.

## Environment Activation
- Conda: `conda activate jsbsim_rl` or use python at `C:\Users\Sean\anaconda3\envs\jsbsim_rl\python.exe`

## Basic Commands (Run in `jsbsim_rl`)
- Verify installation: `python scripts/verify_installation.py`
- Main single pursuit training: `python scripts/train_single_pursuit.py`

## Research Milestones
- **SB3 97.3% Baseline (SEALED)**: `benchmarks/sb3_2v1_97p3/` — centralized upper bound for 2v1 formation pursuit
- **Benchmark eval**: `python scripts/benchmark_sb3_baseline.py -n 100 -d 0.0,0.3,0.6`
- **Self-Attention CTDE Actor**: `src/models/attention_actor.py` — next-gen architecture for crossing the information asymmetry gap
- **Cold-start Attention training**: `python scripts/train_attention_actor.py --mode curriculum --steps 500000`

## Architecture
```
SB3 Shared Policy (CEILING):  66-dim → [256,256] → Box(4)  → 97.3% capture
CTDE MLP (failed):             33-dim → [256,256] → Box(2)  → lazy pursuer
CTDE Attention (NEW):          33-dim → 3 tokens × Self-Attn → Box(2) → TBD
```