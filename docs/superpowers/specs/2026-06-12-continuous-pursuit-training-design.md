# Continuous Pursuit Training — Design Spec

**Date**: 2026-06-12
**Goal**: Train an SB3 PPO agent using continuous control surfaces (throttle/elevator/aileron/rudder) to intercept a straight-flying target, achieving ≥50% capture rate on Stage 1.

## Background

- Engine bug is fixed (`set-running=1` in `Aircraft.reset()`)
- FlightController heading response too slow for pursuit turns
- BFM autopilot Nz PID sign/gain issues cause ground crashes
- Direct surface control bypasses all autopilot issues

## Design Decisions

### 1. Action Bias (Trim Centering)

Agent outputs `[d_thr, d_ele, d_ail, d_rud] ∈ [-1,1]^4`.
These are mapped to actual control surface commands via trim bias + scaling:

| Input | Formula | Range |
|-------|---------|-------|
| throttle | `clip(0.80 + 0.20 × d_thr, 0.0, 1.0)` | [0.60, 1.00] |
| elevator | `clip(-0.05 + 0.15 × d_ele, -1.0, 1.0)` | [-0.20, 0.10] |
| aileron | `clip(0.00 + 0.30 × d_ail, -1.0, 1.0)` | [-0.30, 0.30] |
| rudder | `0.10 × d_rud` | [-0.10, 0.10] |

Trim point (agent output zeros) gives ~176 m/s level flight at 3000m.

**Implementation**: `AirCombatEnv.step()` continuous branch.

### 2. Pursuit Spawn

New function `generate_pursuit_spawn()` in `scenario.py`:

- Target bearing: pursuer heading ±30° (front arc)
- Distance: 1000–2500 m
- Altitude delta: ±200 m from pursuer
- Target heading: pursuer heading ±15° (roughly same direction)
- Target speed: 130 m/s (~250 kts, well below pursuer capability)

Triggered via `curriculum_stage` in `AirCombatEnv.reset()`.

### 3. Training Setup

- **Algorithm**: SB3 PPO, MLP [128,128] ReLU, ortho_init
- **Total steps**: 200,000
- **LR**: 3e-4, **ent_coef**: 0.01, **n_steps**: 2048
- **Eval**: every 10,000 steps, 20 episodes, deterministic
- **Success**: ≥50% capture rate → advance to Stage 2
- **Curriculum**: Stage 1 (straight target, 130 m/s) → Stage 2 (target weaves gently, 160 m/s)

### 4. Files Changed

| File | Change |
|------|--------|
| `src/environment/scenario.py` | Add `generate_pursuit_spawn()` |
| `src/environment/air_combat_env.py` | Trim bias in continuous mode; use pursuit spawn |
| `scripts/train_continuous_pursuit.py` | **New** — training entry point |

## Success Criteria

- [ ] Training runs without NaN/crash errors
- [ ] Eval capture rate reaches ≥50% within 200k steps
- [ ] Tacview export shows realistic intercept trajectory
- [ ] Agent advances from Stage 1 → Stage 2 automatically
