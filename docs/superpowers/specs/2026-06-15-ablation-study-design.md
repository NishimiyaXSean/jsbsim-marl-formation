# Ablation Study: Single-Pursuit Training Optimizations

**Date:** 2026-06-15  
**Status:** Approved  
**Goal:** Identify which of 3 optimization directions most improves Stage 1 → Stage 1.5 capture rate for single-pursuit training, via controlled small-scale ablation experiments.

---

## Background

Current baseline training (500K timesteps, 5-stage curriculum) achieves ~87% Stage 1 capture rate but drops to 3-27% at Stage 1.5 (gentle weaving target). This ablation study tests three targeted modifications at 200K timesteps (Stage 1.0 + 1.5 only) to find which direction yields the largest Stage 1.5 improvement.

## Three Ablation Directions

| Label | Name | What changes | Hypothesis |
|-------|------|-------------|------------|
| **BL** | Baseline | No modification — current `SinglePursuitEnv` | Control group |
| **RW** | Lead Pursuit Reward | Add velocity-alignment + lead-prediction reward terms | Agent learns to aim ahead of target, not directly at it |
| **FS** | Frame Stacking | Stack 4 consecutive observations → 76-dim input | Temporal awareness enables inertia/turn-rate prediction |
| **CA** | Cubic Action Mapping | Policy output `a` mapped via `a³` before physical conversion | Fine control near origin + full authority at extremes |

## Experiment Design

- **Total timesteps:** 200,000 (Stage 1.0: 0-100K, Stage 1.5: 100K-200K)
- **Curriculum stages:** `[1.0, 1.5]` only
- **Stage advancement threshold:** 40% capture rate + ≥40K steps in stage
- **Seeds:** 3 per variant (0, 1, 2) = 12 runs total
- **PPO hyperparameters:** Identical across all 4 variants (lr=3e-4, n_steps=2048, batch=256, n_epochs=10, net_arch=[128,128])

### Success Metrics

1. **Primary:** Peak capture rate at Stage 1.5 (first eval after min 40K steps in stage)
2. **Secondary:** Best Stage 1 capture rate before advancement, avg min distance

---

## Architecture: Wrapper Pattern

Each ablation is implemented as a `gym.Wrapper` composing with the existing `ResidualExpertWrapper`:

```
SinglePursuitEnv → ablation_wrapper (RW/FS/CA or None) → ResidualExpertWrapper → Monitor
```

### Wrapper 1: LeadPursuitRewardWrapper

Intercepts `step()`, lets the env compute default reward, then adds:

1. **Velocity alignment** — `cos(pursuer_vel, LOS_dir)` weighted at 2.0. Rewards the aircraft actually moving toward the target (accounts for AoA/sideslip discrepancy between nose pointing and velocity vector).
2. **Lead prediction** — `cos(pursuer_forward, LOS_to_future)` where `future = target_pos + target_vel × 1.0s` weighted at 3.0. Rewards pointing at the predicted intercept point.

Both terms multiplied by `dt` and added to the base reward.

### Wrapper 2: FrameStackWrapper

- Stacks last `N=4` observations into a flat vector
- Observation space: `Box(19,)` → `Box(76,)`
- Buffer initialized with copies of `obs_0` on `reset()`
- Uses `collections.deque(maxlen=N)` for the ring buffer
- Standard technique — no changes to reward, action, or termination logic

### Wrapper 3: CubicActionWrapper

- Maps policy output: `a_physical = sign(a) × |a|³`
- Observation space unchanged
- Action space unchanged (still Box(3,) ∈ [-1,1])
- The cubic mapping happens post-policy, pre-env, and applies to all 3 action channels (d_heading, d_alt, d_speed)
- Exploration noise also gets cubic-mapped through the same path

**Resolution around origin:**

| Policy output a | Physical fraction (a³) | Heading change | Effective precision |
|:---:|:---:|:---:|:---:|
| 0.0 | 0.000 | 0.0° | — |
| 0.1 | 0.001 | 0.03° | ~1000 steps to 30° |
| 0.3 | 0.027 | 0.81° | ~37 steps to 30° |
| 0.5 | 0.125 | 3.75° | ~8 steps to 30° |
| 0.7 | 0.343 | 10.3° | ~3 steps to 30° |
| 1.0 | 1.000 | 30.0° | 1 step to 30° |

---

## File Changes

### New files
| File | Purpose |
|------|---------|
| `src/environment/ablation_wrappers.py` | All 3 wrapper classes |
| `scripts/run_ablation_study.py` | Orchestrator: runs 12 training jobs + generates summary |

### Modified files
| File | Change |
|------|--------|
| `src/environment/__init__.py` | Export new wrappers |

### No changes to
- `train_single_pursuit.py` (baseline left untouched)
- `single_pursuit_env.py` (wrappers operate externally)
- `rewards.py`, `observations.py`, `bfm_actions.py`

---

## Runner Script: `run_ablation_study.py`

```
Usage: python scripts/run_ablation_study.py [--seeds 0 1 2] [--steps 200000]

Config (hardcoded):
    ABLATIONS = [
        {"name": "baseline",     "wrapper_cls": None},
        {"name": "lead_pursuit", "wrapper_cls": LeadPursuitRewardWrapper},
        {"name": "frame_stack",  "wrapper_cls": FrameStackWrapper},
        {"name": "cubic_action", "wrapper_cls": CubicActionWrapper},
    ]
    SEEDS   = [0, 1, 2]
    STAGES  = [1.0, 1.5]

Flow per run:
    1. Build env chain: SinglePursuitEnv → wrapper? → ResidualExpertWrapper → Monitor
    2. Build eval env identically
    3. PPO(model="MlpPolicy", ...) with identical hyperparams
    4. model.learn(total_timesteps=200000, callback=CurriculumCallback)
    5. Save to marl_runs/ablation/{label}_s{seed}/

Post-run:
    - Reads all eval_metrics.csv files
    - Computes per-variant: peak Stage 1 capture, best Stage 1.5 capture, Wilson CI
    - Saves marl_runs/ablation/summary.csv
    - Prints ranked table sorted by Stage 1.5 capture rate
```

---

## Evaluation Protocol

The `CurriculumCallback` at each `EVAL_FREQ=15000` checkpoint evaluates 30 episodes at the current curriculum stage. For the ablation comparison:

1. **Peak Stage 1 capture rate** — highest `capture_rate` from checkpoints where `stage == 1.0`
2. **Stage 1.5 transfer** — capture rate at the first checkpoint where `stage == 1.5` AND ≥40K steps in stage have elapsed
3. **Time-to-advance** — how many timesteps before the agent first reaches 40% capture and advances from Stage 1 → 1.5

The variant with the **highest Stage 1.5 transfer capture rate** wins, as this is the bottleneck the baseline fails at.

---

## Test Plan

Unit tests for each wrapper (in `tests/test_environment/`):

| Test | What it verifies |
|------|-----------------|
| `test_lead_pursuit_reward_wrapper` | Reward is non-zero when agent moves toward lead point; wrapper correctly passes through non-reward fields |
| `test_frame_stack_wrapper` | Output shape is (76,); reset fills buffer with first obs; successive frames differ |
| `test_cubic_action_wrapper` | a=0 → 0; a=0.5 → 0.125; a=1.0 → 1.0; a=-1.0 → -1.0; symmetry preserved |

No integration tests needed — the ablation runner itself is the integration test.

---

## Expected Timeline

- Wrapper implementation + unit tests: ~1 coding session
- Runner script: ~1 coding session
- Training: 12 runs × ~200K steps each, CPU-bound. Estimate 2-4 hours on a modern CPU
- Analysis: automated by runner

## Risks

- **Frame stacking + PPO**: The 76-dim observation is 4× baseline. The first layer (128-wide) should handle this fine, but if learning stalls, reduce to N=3 frames (57 dims)
- **Cubic mapping + exploration**: log_std_init=0.0 means σ=1.0 Gaussian noise on raw action. After cubic mapping, noise near center is even smaller. This is desirable but may slow early exploration — monitor Stage 1 convergence rate
- **Reward weights may need tuning**: The lead pursuit reward weights (vel_align=2.0, lead_pred=3.0) are initial guesses. If RW variant's reward scale is much larger than baseline, it could destabilize value estimation
