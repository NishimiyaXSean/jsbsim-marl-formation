# jsbsim-marl-formation

**Multi-agent reinforcement learning for formation cooperative route decision and planning**, powered by JSBSim 6-DOF F-16 flight dynamics and MAPPO (Multi-Agent PPO).

> 🎓 **Academic Research Project** — targeting paper submission on *"Multi-agent reinforcement learning for formation cooperative route decision and planning"* at Zhejiang University.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![JSBSim](https://img.shields.io/badge/FDM-JSBSim%20F--16-orange.svg)](https://jsbsim-team.github.io/)

---

## Overview

This project provides a **high-fidelity air combat simulation and training framework** that replaces traditional point-mass models with JSBSim's full 6-DOF F-16 flight dynamics. It supports single-agent and multi-agent reinforcement learning for tactical pursuit, basic fighter maneuvers (BFM), and cooperative formation flight.

### Key Features

- **6-DOF Aerodynamics**: JSBSim F-16 model with real G-loading, stall behavior, engine spool-up, and sideslip dynamics
- **Three Environment Classes**: Continuous pursuit (`SinglePursuitEnv`), discrete BFM pursuit (`BFMPursuitEnv`), and multi-agent combat (`AirCombatEnv`)
- **λ-G Flight Control Law**: F-16-class autopilot closing the loop around normal acceleration (Nz) with speed-dependent gain scheduling
- **CTDE Architecture**: Centralized Training Decentralized Execution via RLlib MAPPO with separate actor/critic observations
- **Action Masking**: 5 safety rules (hard deck, stall, overspeed, high alpha) for safe exploration in discrete BFM mode
- **Auto-Curriculum**: Continuous difficulty scheduling with cliff-collapse rollback and sliding-window win rate evaluation
- **Tacview Export**: ACMI v2.1 telemetry for after-action review in Tacview
- **Comprehensive Evaluation**: 3D trajectory plots, altitude profiles, Wilson CI statistics, multi-seed aggregation

---

## Installation

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/NishimiyaXSean/jsbsim-marl-formation.git
cd jsbsim-marl-formation

# Install dependencies (Poetry or pip)
pip install -e .
```

### JSBSim Aircraft Data

Download the official JSBSim aircraft, engines, and systems data into `data/jsbsim/`:

```bash
git clone https://github.com/JSBSim-Team/jsbsim.git /tmp/jsbsim
cp -r /tmp/jsbsim/aircraft data/jsbsim/
cp -r /tmp/jsbsim/engines  data/jsbsim/
cp -r /tmp/jsbsim/systems  data/jsbsim/
```

Alternatively, set the `JSBSIM_DATA_DIR` environment variable to point to an existing JSBSim installation.

### Verify Installation

```bash
python scripts/verify_installation.py
```

This performs a 4-step check: JSBSim bindings → aircraft data → F-16 model load → 1000-step test simulation.

---

## Quick Start

```bash
# Simplest demo: single F-16 tracking a moving target (50K steps)
python scripts/run_single_agent.py

# Single pursuit training with continuous actions (5M steps)
python scripts/train_single_pursuit.py

# Discrete BFM pursuit training with action masking (5M steps)
python scripts/train_bfm_pursuit.py

# 1v1 MAPPO multi-agent combat training
python scripts/run_1v1_training.py
```

---

## Project Structure

```
jsbsim-marl-formation/
├── src/
│   ├── dynamics/              # JSBSim F-16 wrapper + flight control laws
│   │   ├── aircraft.py        #   Aircraft — wraps JSBSim FGFDMExec for a single F-16
│   │   ├── autopilot.py       #   BFMAutopilot (λ-G flight control), PIDController,
│   │   │                      #   GainScheduler, TrimSchedule, AltitudeHoldAP, SpeedHoldAP
│   │   ├── bfm_actions.py     #   13-action BFM set + 9-action pursuit subset definitions
│   │   ├── flight_controller.py # Stabilized FlightController (altitude/speed/heading stabilizers)
│   │   └── flight_envelope.py #   V-n diagram, GPWS, G-onset lag, roll rate limiting,
│   │                          #   stall/overspeed clamps, G-scale curriculum
│   │
│   ├── environment/           # Gymnasium environments
│   │   ├── air_combat_env.py  #   AirCombatEnv (MultiAgentEnv) — 1v1 combat, 3 action modes
│   │   ├── single_pursuit_env.py # SinglePursuitEnv — 3D continuous pursuit with FlightController
│   │   ├── bfm_pursuit_env.py #   BFMPursuitEnv — Discrete(9) BFM pursuit with action masking
│   │   ├── observations.py    #   19-dim local obs + 26-dim global state (CTDE)
│   │   ├── rewards.py         #   RewardConfig: progress, ATA, AA, HCA, closing speed, etc.
│   │   ├── termination.py     #   Collision, CPA, ground crash, out-of-bounds, timeout checks
│   │   ├── scenario.py        #   Spawn generation: 1v1 combat + front-arc pursuit
│   │   ├── curriculum.py      #   3-stage curriculum config + auto-advancement logic
│   │   ├── ablation_wrappers.py # FrameStack, BlendedAction, LeadPursuitReward, ActionRepeat
│   │   └── masked_policy.py   #   MaskableActorCriticPolicy for SB3 action masking
│   │
│   ├── models/                # Neural network architectures
│   │   └── mappo_model.py     #   MAPPOModel (TorchModelV2) — actor [19→256→256→8],
│   │                          #   critic [26→256→256→1] for RLlib MAPPO CTDE
│   │
│   ├── training/              # Training pipelines
│   │   ├── train_mappo.py     #   RLlib MAPPO multi-agent training (500 iters, 3-stage curriculum)
│   │   ├── callbacks.py       #   AirCombatCallbacks — kill/crash/OOB/timeout rate tracking
│   │   └── baselines.py       #   Random agent + pure pursuit guidance baseline
│   │
│   ├── utils/                 # Mathematical utilities
│   │   ├── geometry.py        #   Tactical geometry: ATA, AA, HCA, LOS, closing speed
│   │   ├── kinematics.py      #   NED→WGS-84 coordinate transformation
│   │   ├── pn_guidance.py     #   Proportional Navigation guidance with augmented bearing bias
│   │   └── units.py           #   Imperial ↔ SI unit conversions
│   │
│   ├── logging/               # Telemetry export
│   │   └── tacview_exporter.py #  Tacview ACMI v2.1 writer for after-action review
│   │
│   └── visualization/         # 3D visualization (MeshCat / FlightGear bridge — planned)
│
├── configs/
│   ├── env/
│   │   └── 1v1_combat.yaml    #   1v1 environment: 60 Hz, 0.5s decision interval, 240s max
│   └── model/
│       └── mappo_ctde.yaml    #   MAPPO model: [256,256] actor/critic, tanh activation
│
├── scripts/                   # Entry-point scripts (22 total)
│   ├── train_single_pursuit.py #  Primary training: SB3 PPO on SinglePursuitEnv (5M steps)
│   ├── train_bfm_pursuit.py   #  SB3 PPO on BFMPursuitEnv with action masking (5M steps)
│   ├── run_1v1_training.py    #  RLlib MAPPO 1v1 combat training
│   ├── run_sb3_training.py    #  SB3 PPO 1v1 (avoids Ray/RLlib on Windows)
│   ├── run_single_agent.py    #  Simplest demo: single F-16 tracks moving target
│   ├── evaluate_and_visualize.py # Comprehensive eval: Tacview + 3D plots + Wilson CI
│   ├── run_ablation_study.py  #  4-variant × 3-seed ablation study
│   ├── run_hyperparam_sweep.py #  Orthogonal hyperparameter sweep (fractional factorial)
│   ├── run_v10_5_training.py  #  V10.5 batch with anti-dolphin fixes
│   ├── verify_installation.py #  4-step installation verification
│   ├── verify_pursuit.py      #  3-scenario pursuit physics verification
│   ├── verify_bfm_actions.py  #  Phase 4 BFM action validation
│   ├── diagnose_dynamics.py   #  7-test diagnostic: trim, turn rate, climb/dive, PN, etc.
│   ├── sweep_elevator.py      #  Phase 1: elevator sweep → trim table calibration
│   ├── tune_pitch.py          #  Phase 2: pitch (Nz) PID tuning
│   ├── tune_roll.py           #  Phase 2: roll PID tuning
│   ├── tune_speed.py          #  Phase 2: speed PID tuning
│   ├── quick_tacview.py       #  Quick Tacview export + 3D plots
│   ├── quick_tacview_bfm.py   #  Tacview export for BFM pursuit models
│   └── ...                    #  Additional eval/viz scripts
│
├── tests/
│   ├── test_dynamics/         # Aircraft, autopilot, BFM actions, flight envelope tests
│   ├── test_environment/      # Reset/step validity, ablation wrapper tests
│   └── test_models/           # Model tests (placeholder)
│
├── data/
│   ├── trim_table.json        # F-16 speed-to-elevator-trim lookup (Phase 1 calibration)
│   └── jsbsim/                # JSBSim aircraft/engines/systems data (git-ignored, user-provided)
│
├── docs/                      # Session summaries, training reports, design specs
├── marl_runs/                 # Training outputs (models, logs, CSVs)
├── results/                   # Evaluation outputs (plots, CSVs, Tacview files)
├── notebooks/                 # Jupyter notebooks (reserved)
├── pyproject.toml             # Poetry project config + dependencies
├── CLAUDE.md                  # Claude Code agent instructions
├── CITATION.cff               # Citation metadata
└── LICENSE                    # MIT License
```

---

## Environments

### AirCombatEnv — 1v1 Multi-Agent Combat

Multi-agent environment for adversarial air combat training with RLlib MAPPO.

- **Agents**: `attacker_0`, `evader_0`
- **Action Modes**: `"continuous"` (4-dim Box), `"bfm"` (Discrete 13), `"pursuit"` (Discrete 9)
- **Observation**: Dict with 19-dim local obs + 26-dim global state (CTDE)
- **Termination**: Collision (50 m), CPA (300 m for kill), ground crash (10 m), ceiling (4900 m), timeout (240 s)
- **Curriculum**: 3 stages with increasing evader speed/G coefficients and warning radii
- **Features**: Tacview export, GPWS with hysteresis, FlightEnvelope safety constraints

### SinglePursuitEnv — Continuous Pursuit

Single-agent pursuit environment with 3D continuous action space (recommended for most training).

- **Action Space**: 3-dim continuous `[d_heading, d_alt, d_speed]` via FlightController interface
- **Observation Space**: 25-dim (body-frame relative state + tactical geometry + AoA + airspeed + specific excess power)
- **Decision Rate**: 10 Hz (0.1 s per step)
- **Auto-Curriculum**: Continuous difficulty [0, 1] with 50-episode sliding window win rate
- **Anti-Stall**: Truncation on AoA exceedance + quadratic altitude penalty
- **Training Wrapper Chain**: `SinglePursuitEnv → BlendedAction → LeadPursuitReward → ResidualExpert → ActionRepeat(5×, 2 Hz)`

### BFMPursuitEnv — Discrete BFM Pursuit

Single-agent pursuit with Discrete(9) BFM action space and full autopilot pipeline.

- **Action Space**: Discrete 9 (`PURSUIT_ACTIONS`: pure pursuit, lead pursuit, lag pursuit, high/low yo-yo, etc.)
- **Pipeline**: `PURSUIT_ACTIONS → FlightEnvelope → BFMAutopilot (GainScheduler) → JSBSim`
- **Action Masking**: 5 safety rules preventing self-destructive actions during exploration
- **Observation**: 25-dim Dict with `"obs"` + `"action_mask"` keys
- **Gain Scheduling**: 1/V² adaptive PID gains for Nz channel across speed envelope

---

## Key Architecture

### λ-G Flight Control Law

The `BFMAutopilot` implements a real F-16-class flight control system (Stevens & Lewis) that closes the loop around normal acceleration (Nz) rather than pitch attitude:

```
δ_elevator = PID(Nz_cmd − Nz_actual) + trim(V)
δ_aileron  = PID(φ_cmd − φ_actual)
δ_rudder   = PID(β_cmd − β_actual)   # sideslip suppression
Throttle   = PID(V_cmd − V_actual)
```

### Speed-Dependent Gain Scheduling

- **Nz channel**: 1/V² adaptive PID gains maintain constant loop gain across the speed envelope (150–400 m/s)
- **Trim schedule**: Speed-to-elevator-trim lookup table calibrated from Phase 1 open-loop elevator sweeps

### Proportional Navigation Guidance

The PN guidance law provides a collision-course intercept solution with augmented bearing bias for initial target acquisition:

```
ψ_cmd = ψ_los + N × ψ̇_los × t_go + bearing_bias
```

### CTDE Architecture (MAPPO)

- **Actor**: 19-dim local observation → [256, 256] → 4-dim action `[throttle, elevator, aileron, rudder]`
- **Critic**: 26-dim global state → [256, 256] → scalar value (centralized training only)

---

## Training

### Continuous Pursuit (Recommended)

```bash
# Main training with auto-curriculum (5M steps)
python scripts/train_single_pursuit.py

# Ablation study: 4 variants × 3 seeds
python scripts/run_ablation_study.py

# Hyperparameter sweep
python scripts/run_hyperparam_sweep.py
```

Training features:
- **gSDE** (generalized State-Dependent Exploration) for temporally-correlated exploration
- **Auto-curriculum** with spring mechanism: conservative advancement (consecutive good evals required), cliff-collapse rollback (restores model weights + resets difficulty on performance drop)
- **Lead pursuit reward**: velocity alignment + lead prediction + LOS-rate damping, all V_c-coupled with minimum-wage floor
- **Anti-dolphin fixes** (V10.5): quadratic altitude delta penalty, lowered V_c saturation ceiling, action smoothness penalty

### Discrete BFM Pursuit

```bash
python scripts/train_bfm_pursuit.py
```

Uses `MaskableActorCriticPolicy` with 5 safety rules in the action mask.

### Multi-Agent MAPPO (1v1)

```bash
python scripts/run_1v1_training.py
```

Uses Ray RLlib with dual policies (attacker + evader), 500 iterations, 3-stage curriculum.

---

## Evaluation

```bash
# Comprehensive evaluation (all models)
python scripts/evaluate_and_visualize.py

# Multi-seed evaluation with Wilson CI
python scripts/eval_multi_seed.py

# Quick Tacview export from checkpoint
python scripts/quick_tacview.py

# Generate Tacview + trajectory plots for BFM models
python scripts/quick_tacview_bfm.py
```

Outputs include:
- **Tacview ACMI** (.txt.acmi) — drag-and-drop into [Tacview](https://www.tacview.net/) for 3D after-action review
- **3D trajectory plots** (.png) — matplotlib 3D visualization with start/end markers
- **Altitude profiles** (.png) — time-series altitude with key event annotations
- **Summary statistics** — kill rate, average episode length, Wilson confidence intervals
- **Multi-seed CSV** — per-seed metrics for statistical aggregation

---

## Verification & Diagnostics

```bash
# Check JSBSim + F-16 are working correctly
python scripts/verify_installation.py

# Verify F-16 can physically intercept a target (3 scenarios)
python scripts/verify_pursuit.py

# Validate all 9 BFM actions through the full pipeline
python scripts/verify_bfm_actions.py

# Run 7-test diagnostic suite (trim, turn rate, climb/dive, PN, etc.)
python scripts/diagnose_dynamics.py
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_dynamics/ -v
pytest tests/test_environment/ -v
```

Test coverage includes:
- **Dynamics**: Aircraft creation/reset/step, autopilot output shapes + channel response, BFM action counts/indices/validity, flight envelope V-n diagram + GPWS + roll limiting
- **Environment**: Reset observation validity (shape/dtype/bounds), step data integrity, ablation wrapper correctness (FrameStack, CubicAction, LeadPursuitReward)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `jsbsim` | 6-DOF F-16 flight dynamics engine |
| `gymnasium ~=1.2` | Reinforcement learning environment interface |
| `ray[rllib] ^2.40` | Distributed multi-agent training (MAPPO) |
| `stable-baselines3` | Single-agent PPO training |
| `torch >=2.0` | Neural network backend |
| `numpy`, `scipy` | Numerical computation |
| `matplotlib` | Trajectory and evaluation plots |
| `tensorboard` | Training monitoring |
| `pyyaml` | Configuration file parsing |
| `transforms3d` | 3D rotation and coordinate transforms |

---

## Citation

```bibtex
@software{nishimiya2026jsbsim,
  author       = {Sean Nishimiya},
  title        = {jsbsim-marl-formation: Multi-Agent RL for Formation Flight with JSBSim F-16 Dynamics},
  year         = 2026,
  affiliation  = {Zhejiang University},
  url          = {https://github.com/NishimiyaXSean/jsbsim-marl-formation}
}
```

See [CITATION.cff](CITATION.cff) for complete metadata.

---

## License

MIT — see [LICENSE](LICENSE) for details.
