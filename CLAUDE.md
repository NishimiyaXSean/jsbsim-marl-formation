# CLAUDE.md - JSBSim MARL Research Core Guidance

## 🔴 CRITICAL RULES (MUST OBEY)
1. **Virtual Environment**: ALL scripts, commands, and tests MUST run inside the `marl_env` conda environment. Always activate before executing.
2. **WSL2 Environment**: This project runs on WSL2 (Ubuntu). Do NOT use Windows paths (`C:\`, `\\wsl$\`). All paths are Linux-native.
3. **No RLlib Warnings Ignored**: RLlib deprecation warnings are expected (Ray 2.40). Do NOT try to "fix" them — they're harmless.

## 🖥️ WSL2 Environment

- **OS**: Ubuntu via WSL2 on Windows 11
- **GPU**: NVIDIA CUDA passthrough (verify with `nvidia-smi`)
- **Conda env**: `marl_env` (Python 3.10, PyTorch 2.12, Ray 2.40, JSBSim 1.3)
- **Activation**: `conda activate marl_env`
- **Project root**: `/home/sean/jsbsim-marl-formation`

## Startup Checklist

```bash
# 1. Activate environment
conda activate marl_env

# 2. Verify GPU (training needs CUDA)
nvidia-smi

# 3. Clean any stale Ray instances
ray stop

# 4. Verify installation
python scripts/verify_installation.py
```

## Basic Commands

### Training (RLlib MAPPO — PRIMARY PIPELINE)

```bash
# Discrete action space (CURRENT DEFAULT):
python scripts/train_formation_rllib.py --iterations 120 --cooperative --no-bc
python scripts/train_formation_rllib.py --iterations 300 --cooperative --warmup 200000

# With BC pretraining (continuous model only):
python scripts/train_formation_rllib.py --iterations 120 --cooperative \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth

# Resume from checkpoint:
python scripts/train_formation_rllib.py --iterations 300 --cooperative --warmup 200000 \
    --resume-from /home/sean/jsbsim-marl-formation/marl_runs/PATH/checkpoints/checkpoint_XXXXXX
```

### Evaluation & Visualization

```bash
# SB3 baseline benchmark
python scripts/benchmark_sb3_baseline.py -n 100 -d 0.0,0.3,0.6

# Collect trajectory + attention data
python scripts/collect_viz_data.py --ckpt PATH_TO_CHECKPOINT --episodes 10

# Generate paper figures
python scripts/viz_paper_figures.py --data data/viz/DATA.npz --episode 0
python scripts/viz_fig3_role_attention.py --data data/viz/DATA.npz

# Statistical eval analysis
python scripts/analyze_eval_statistics.py --data data/viz/DATA.npz
```

### Ray Cleanup

```bash
# Kill all Ray processes (when training hangs)
ray stop
pkill -f "train_formation_rllib"
pkill -f "RolloutWorker"
```

## 🎓 ULTIMATE GOAL: ACADEMIC PAPER SUBMISSION
- **Core Research Objective**: Build a scalable framework for a high-quality academic paper on **"Multi-agent reinforcement learning for formation cooperative route decision and planning"**.
- **Physics Engine**: JSBSim 6-DOF F-16 flight dynamics model (FDM) for extreme aerodynamic fidelity.
- **Algorithm**: Multi-Agent PPO (MAPPO) with Centralized Training Decentralized Execution (CTDE), recently migrated from IPPO to Parameter-Shared MAPPO to eliminate non-stationarity.

## Architecture (Current — July 2026)

```
Parameter-Shared MAPPO (CTDE) + Self-Attention + MultiDiscrete Actions
─────────────────────────────────────────────────────────────────────

Shared Policy:       one AttentionFormationActor + AttentionCritic
                     for both p0 and p1 agents

Action Space:        MultiDiscrete([5 turn, 3 speed]) = 15 primitives
                     Turn:  HardLeft(-15°/s) SoftLeft(-5°) Straight(0°) SoftRight(+5°) HardRight(+15°)
                     Speed: Slow(180m/s) Cruise(250m/s) Fast(320m/s)

Observation:         Dict{"obs": Box(33), "global_state": Box(21), "action_mask": Box(8)}
                     Self(13) + Target(14) + Mate(6) → 3-token Self-Attention

Action Masking:      Stalls (<130m/s): forbid slow speed + hard turns
                     Ground proximity (<200m): forbid hard turns
                     Overspeed (>95% max): forbid fast

Decision Rate:       5 Hz (0.2s macro-action, 12 physics sub-steps at 60 Hz)

Cooperative:         Phase 1 [OR]: any pursuer < 200m → success
                     Phase 2 [AND]: both < dynamic_dist + pincer > 30° for 6 steps
                     Dynamic AND annealing: 2000m → 800m over training
                     Distance asymmetry penalty + time-sync pacing penalty
```

## Key Files

| File | Purpose |
|------|---------|
| `src/environment/formation_rllib_env.py` | RLlib MultiAgentEnv — Phase 5 cooperative 2v1 |
| `src/models/formation_rllib_model.py` | TorchModelV2 — Self-Attn Actor + Attn Critic + discrete heads |
| `src/models/attention_actor.py` | AttentionFormationActor + Tokenized AttentionCritic |
| `scripts/train_formation_rllib.py` | RLlib MAPPO training entry point |
| `scripts/collect_viz_data.py` | Trajectory + attention weight data collection |
| `scripts/viz_paper_figures.py` | Fig 1 (3D) + Fig 2 (attention timeline) |
| `scripts/viz_fig3_role_attention.py` | Fig 3 (role-grouped attention matrix) |
| `scripts/analyze_eval_statistics.py` | Eval episode statistical autopsy |

## Common Issues

1. **NaN in discrete model**: Caused by `float("-inf")` in action mask. Fixed by using `-1e9` instead.
2. **Training hangs**: Stale Ray workers from previous runs. Run `ray stop` + kill processes.
3. **`pyarrow.lib.ArrowInvalid: URI has empty scheme`**: Checkpoint path must be absolute.
4. **`No samples returned from remote workers`**: Too many Ray workers competing for CPU. Clean up and restart.
5. **BC weights incompatible with discrete model**: Continuous BC weights have `mean.weight` [2,256] but discrete needs `turn_head.weight` [5,256]. Skip `--load-bc` for discrete training.

## Experiment Results Summary

| Experiment | Architecture | Action Space | Best Eval | Key Finding |
|-----------|-------------|-------------|-----------|-------------|
| SB3 (ceiling) | 66-dim shared | Box(4) | +5,908 (92%) | Centralized upper bound |
| Exp 1 (MAPPO non-coop) | Shared Attn CTDE | Box(2) | -8,053 | Broke IPPO plateau but entropy diverged |
| Exp 2 (MAPPO OR-gate) | Shared Attn CTDE | Box(2) | +7,888 | 33% above SB3 ceiling |
| Exp 3 v1 (AND, 800m) | Shared Attn CTDE | Box(2) | -5,909 | First positive training reward |
| Exp 3 v3 (AND, 2000→800m) | Shared Attn CTDE | Box(2) | -1,171 | Dynamic annealing: eval improved 5,000 pts |
| Exp 4a (Discrete OR) | Shared MLP CTDE | MultiDiscrete(5,3) | TBD | Discrete migration verification |
