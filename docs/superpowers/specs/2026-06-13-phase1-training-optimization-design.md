# Phase 1 Training Optimization — Design Spec

**Date**: 2026-06-13  
**Goal**: Stage 3 capture rate ≥ 90%  
**Constraint**: Collision radius stays at 200m (not reduced)

---

## Overview

Two-phase approach:
- **Phase 1a**: PN guidance expert + reward function tuning (tightly coupled)
- **Phase 1b**: 5-stage curriculum + orthogonal hyperparameter sweep + multi-seed evaluation

---

## Phase 1a: PN Expert + Reward Tuning

### 1.1 PN Guidance Expert (`_compute_expert` upgrade)

**File**: `scripts/train_single_pursuit.py` (class `ResidualExpertWrapper`)

**Current**: `bearing * 0.3 -> aileron` (pure pursuit, no lead angle)

**New**: Call `compute_pn_heading()` from `src/utils/pn_guidance.py`, then P-controller on heading error:

```python
def _compute_expert(self, obs: np.ndarray) -> np.ndarray:
    # Reconstruct world-frame position/velocity from obs (inverse of _get_obs body-frame transform)
    # This requires access to env state → wrapper stores reference to base env,
    # or env extends info dict with expert-relevant world-frame data
    desired_heading = compute_pn_heading(
        pursuer_ned, pursuer_vel, target_ned, target_vel,
        current_heading_deg, dt=0.5, nav_constant=3.0, max_turn_rate_dps=15.0
    )
    heading_error = wrap_180(desired_heading - current_heading)
    ail = np.clip(heading_error * 0.05, -0.3, 0.3)  # K_p=0.05, clipped ±0.3
    return np.array([ail, 0.0, 1.0], dtype=np.float32)
```

**Implementation approach**: The simplest path — `ResidualExpertWrapper.__init__` stores a reference to `self.env` (the unwrapped `SinglePursuitEnv`), and reads `pursuer.position_ned`, `pursuer.velocity_ned`, `target_ac.position_ned`, `target_ac.velocity_ned` directly. No info dict extension needed.

**Key parameters**:
- `nav_constant = 3.0` (standard PN value)
- `max_turn_rate_dps = 15.0` (realistic for F-16 at combat speed)
- P-gain on heading error: 0.05 (gentle, avoids overshoot)

### 1.2 Reward Function Tuning

**File**: `src/environment/single_pursuit_env.py`

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `REWARD_PROGRESS` | 2.0 | 5.0 | Stronger primary pursuit signal |
| `REWARD_ATA` | 3.0 | 5.0 | Stronger pointing incentive |
| Terminal boost (NEW) | none | progress × 3 when dist < 500m | Encourages aggressive terminal phase |
| Time pressure (NEW) | none | `-0.5 × (t/120) × dt` per micro-step | Discourages leisurely approach |
| `REWARD_SUCCESS` | 500 | 500 | unchanged |
| `REWARD_CRASH` | -200 | -200 | unchanged |
| `REWARD_LOST_TARGET` | -200 | -200 | unchanged |
| `REWARD_GROUND_WARNING` | 2.0 | 2.0 | unchanged |

**Terminal boost implementation** — inside micro-step loop:
```python
if current_dist < 500.0:
    total_reward += REWARD_PROGRESS * delta_dist * 2.0  # additive on top of base progress
```

**Time pressure implementation** — inside micro-step loop:
```python
time_ratio = self._step_counter / (CTRL_FREQ * MAX_EPISODE_TIME)
total_reward -= 0.5 * time_ratio * dt
```

### 1.3 Phase 1a Validation

- Single seed (seed=0), 200k steps
- Confirm Stage 1 capture rate stays ≥ 50% (PN should not regress easy case)
- Check Stage 2-3 has improvement trend vs baseline
- If Stage 1 regresses: reduce nav_constant or increase P-gain damping

---

## Phase 1b: Curriculum + Hyperparams + Evaluation

### 2.1 Five-Stage Curriculum

**File**: `src/environment/single_pursuit_env.py`

Change `curriculum_stage` from `int` to `float` (1.0, 1.5, 2.0, 2.5, 3.0).

| Stage | Bearing | Init Dist (m) | Target Speed (m/s) | Heading Rate (°/s) | Alt Rate (m/s) |
|-------|---------|---------------|---------------------|---------------------|----------------|
| 1.0 | 0° | 800–1800 | 130 | 0 | 0 |
| 1.5 | ±7° | 900–2000 | 145 | ±3 | ±1.5 |
| 2.0 | ±15° | 1000–2500 | 160 | ±10 | ±3 |
| 2.5 | ±30° | 1200–2700 | 170 | ±15 | ±5 |
| 3.0 | ±45° | 1500–3000 | 180 | ±20 | ±8 |

**Advancement thresholds**:
- Stage 1.0→1.5, 1.5→2.0: capture rate ≥ 40%
- Stage 2.0→2.5, 2.5→3.0: capture rate ≥ 50%
- Reset best capture rate after each advancement (existing logic preserves this)

**Implementation**: Refactor `_generate_target_profile()` to accept float stage and use `np.interp` or dict lookup with rounding. Stage transitions happen in 0.5 increments only (not continuous).

**File**: `scripts/train_single_pursuit.py` (class `CurriculumCallback`)
- Update `CURRICULUM_STAGES = [1.0, 1.5, 2.0, 2.5, 3.0]`
- Advancement logic: use `np.isclose(stage, 3.0)` to detect terminal stage

### 2.2 Orthogonal Hyperparameter Sweep

**New file**: `scripts/run_hyperparam_sweep.py`

4 parameters, 2 levels each, fractional factorial (8 configs):

| Parameter | Low (-1) | High (+1) |
|-----------|----------|-----------|
| `lr` | 1e-4 | 3e-4 |
| `ent_coef` | 0.005 | 0.01 |
| `net_arch` | [128, 128] | [256, 128] |
| `n_steps` | 2048 | 4096 |

**Fixed**: `batch_size=256`, `gamma=0.99`, `clip_range=0.2`, `total_timesteps=500k`, `vf_coef=0.5`, `max_grad_norm=0.5`

**Procedure**:
1. Run all 8 configs with 1 seed (seed=0), 500k steps
2. Rank by final capture rate (Stage 5 eval)
3. Select top-2 configs
4. Run top-2 with 5 seeds each (0-4), 500k steps
5. Generate comparison report (mean ± std capture rate, min dist, intercept time)

**Output**: `results/sweep_YYYYMMDD_HHMM/report.csv`, `summary.txt`

### 2.3 Multi-Seed Evaluation Enhancement

**File**: `scripts/evaluate_and_visualize.py` (extend)

Add:
- `--multi-seed` flag: aggregate results across multiple model directories
- Wilson score confidence interval for capture rate
- CSV export with columns: seed, capture_rate, avg_min_dist, avg_intercept_time, std_min_dist
- Print formatted summary table

**File**: `scripts/train_single_pursuit.py`

- `EVAL_EPISODES` from 20 → 30 (reduce eval noise)
- `EVAL_FREQ` from 10_000 → 15_000 (reduce eval overhead with longer training)
- `TOTAL_TIMESTEPS` default from 200_000 → 500_000
- `CurriculumCallback` records eval metrics to CSV per-run for later aggregation

### 2.4 Updated Training Config

```python
TOTAL_TIMESTEPS = 500_000
CURRICULUM_STAGES = [1.0, 1.5, 2.0, 2.5, 3.0]
STAGE_TIMESTEPS = TOTAL_TIMESTEPS // len(CURRICULUM_STAGES)
EVAL_EPISODES = 30
EVAL_FREQ = 15_000
TARGET_CAPTURE_RATE_STAGE_1_2 = 0.40
TARGET_CAPTURE_RATE_STAGE_2_3 = 0.50
RESIDUAL_SCALE = 0.5
```

---

## Files Changed Summary

| File | Phase | Change Type |
|------|-------|-------------|
| `scripts/train_single_pursuit.py` | 1a, 1b | PN expert, curriculum stages, eval config |
| `src/environment/single_pursuit_env.py` | 1a, 1b | Reward tuning, 5-stage float curriculum |
| `scripts/run_hyperparam_sweep.py` | 1b | **New** — sweep runner |
| `scripts/evaluate_and_visualize.py` | 1b | Multi-seed aggregation, CI, CSV export |

---

## Success Criteria

1. **Phase 1a quick check**: Stage 1 capture rate ≥ 50% with PN expert (no regression)
2. **Phase 1b completion**: Top config achieves Stage 3 capture rate ≥ 90% (5-seed mean)
3. **Evaluation report**: CSV + summary with Wilson CI for top-2 configs

---

## Risks

| Risk | Mitigation |
|------|-----------|
| PN expert causes Stage 1 regression | Reduce nav_constant to 2.5, increase P-gain damping |
| 500k steps insufficient for Stage 3 | Can extend to 750k-1M for final config |
| JSBSim NaN crashes under aggressive PN turns | Clip aileron to ±0.3, monitor flight envelope |
| 5-stage slows training (more eval, more stages) | EVAL_FREQ adjusted up; stages naturally gate progress |
