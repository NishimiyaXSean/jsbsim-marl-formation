# CLAUDE.md - JSBSim MARL Research Core Guidance

## 🔴 CRITICAL RULES (MUST OBEY)
1. **Virtual Environment**: All scripts, commands, and tests MUST be executed inside the `jsbsim_rl` virtual environment. Always ensure it is activated before running tasks.
2. **Task Initiation Protocol**: BEFORE starting any code modification, feature addition, or file refactoring, you MUST invoke the `superpower` skill workflow (`/superpowers:brainstorm`) to establish the design spec, architectural alignment, and TDD plan.

## 🎓 ULTIMATE GOAL: ACADEMIC PAPER SUBMISSION
- **Core Research Objective**: Build a scalable framework for a high-quality academic paper on **"Multi-agent reinforcement learning for formation cooperative route decision and planning"**.
- **Physics Engine**: Migrate fully from PyBullet to **JSBSim 6-DOF** F-16 flight dynamics model (FDM) to achieve extreme aerodynamic fidelity.
- **Algorithm & Scaling**: Transition from discrete BFM pursuit baselines to continuous multi-agent algorithms (MAPPO/PPO), advancing from 1v1 single pursuit to NvM (e.g., 2v2) collaborative formation training.

## Basic Commands (Run in `jsbsim_rl`)
- Verify installation: `python scripts/verify_installation.py`
- Main single pursuit training: `python scripts/train_single_pursuit.py`