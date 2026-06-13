# CLAUDE.md - JSBSim MARL Formation & Pursuit

## 🔴 CRITICAL OPERATIONAL RULES (MUST OBEY)
1. **Virtual Environment**: All Python scripts, tests, and commands MUST be executed within the `jsbsim_rl` virtual environment. Always prepend or ensure environment activation (e.g., `conda activate jsbsim_rl`).
2. **Task Initiation Protocol**: BEFORE starting any code modification, refactoring, or file creation, you MUST invoke the `superpower` skill workflow (specifically `/superpowers:brainstorm`) to outline the design specs, code structure, and potential impact. Do not write code blindly.

## Technology Stack & Core Architecture
- **Dynamics**: JSBSim F-16 Flight Dynamics Model (FDM)
- **RL Framework**: Stable Baselines3 / MAPPO (Gymnasium Multi-Agent Env)
- **Core Strategy**: Residual RL + Adaptive Guidance Expert (`src/utils/pn_guidance.py`)
- **Primary Script**: `scripts/train_single_pursuit.py` (3-stage curriculum training)

## Project Structure
- `src/`: `dynamics/` (aircraft/autopilot), `environment/` (single_pursuit_env), `utils/` (pn_guidance)
- `scripts/`: `train_single_pursuit.py` (Main), `verify_pursuit.py` (Validation)
- `configs/`: Hydra YAML configurations
- `docs/superpowers/specs/`: Design specifications

## Useful Commands (Run in `jsbsim_rl`)
- **Main Training**: `python scripts/train_single_pursuit.py`
- **Verify Env**: `python scripts/verify_installation.py` or `python scripts/verify_pursuit.py`
- **Code Style**: Use explicit type hints, follow PEP 8. Maintain G-smoothing and GPWS constraints in world frame (`h_dot_fps`).