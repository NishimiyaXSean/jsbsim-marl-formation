# Single-Pursuit Training Optimization Summary

**Date:** 2026-06-15  
**Status:** In Progress  
**Goal:** Transform Stage 1.5 capture rate from ~3% to a viable baseline via systematic optimization

---

## 1. Problem Diagnosis

### Baseline Performance (500K steps, 5-stage curriculum)
- Stage 1.0 (straight target): **87%** capture rate
- Stage 1.5 (weaving target): **0-13%** capture rate
- Failure mode: agent gets close (~700m) but cannot close the final gap within 120s
- Root cause: agent learns "pure pursuit" (chase distance) rather than "lead pursuit" (collision-course geometry)

### Key Diagnostic Tools Added
- **Termination distribution**: tracks succ/timeout/lost/crash/stall per eval
- **Reward decomposition**: per-component breakdown (progress, ATA, lead_vel, lead_pred, los_rate, step_penalty)
- **Physical telemetry**: `difficulty_physical` column verifies difficulty reaches JSBSim

---

## 2. Observation Space (19 → 25 dims)

| Indices | Feature | Source |
|---------|---------|--------|
| 0-2 | Target relative position (body frame) | — |
| 3-5 | Own velocity (body frame) | — |
| 6-8 | Own attitude rpy | — |
| 9-11 | Own angular velocity [p, q, r] | **Real JSBSim body rates** (was placeholder zeros) |
| 12 | Own height | — |
| 13-15 | Target velocity (body frame) | — |
| 16-18 | Target angular velocity | Finite-diff of target rpy |
| 19-21 | Tactical geometry cos(ATA), cos(AA), cos(HCA) | — |
| **22** | **Angle of Attack** | JSBSim `aero/alpha-deg` |
| **23** | **Airspeed** | JSBSim `velocities/vc-kts` |
| **24** | **Specific Excess Power (Ps)** | Computed: dh/dt + (V/g)·dV/dt |

---

## 3. Control Architecture

### Frequency: 2Hz → 10Hz
- `DECISION_DT`: 0.5s → 0.1s (6 micro-steps per decision)
- Blind-zone displacement: 90m → 18m (@180m/s)
- Gamma compensation: 0.99 → 0.998 (same per-second discount as 2Hz)

### Action Space
- `d_heading`: [-6°, +6°] per 0.1s → max 60°/s turn rate
- `d_alt`: [-3m, +3m] per 0.1s → max 30m/s climb rate
- `d_speed`: [-2, +2] m/s per 0.1s → max 20m/s² acceleration

### Cubic Action Mapping (CARW default)
- `physical = sign(a) × |a|³` — precision near origin, full authority at extremes
- Proven superior to linear-cubic blend at 10Hz (dead-zone acts as implicit regularizer)

### Blended Action Wrapper (BLRW, experimental)
- `physical = 0.2·a + 0.8·a³` — linear floor prevents dead-zone gradient collapse
- Available but not default: cubic outperformed at diff ≥ 0.15

---

## 4. Reward Architecture (v6 → v7)

### v6: Reward Rebalance (the "flip")
| Component | Weight (before) | Weight (after) | Purpose |
|-----------|:---:|:---:|------|
| `REWARD_PROGRESS` | 5.0 | **0.5** | Weak distance hint, not the goal |
| `VEL_ALIGN_WEIGHT` | 2.0 | **15.0** | Velocity aligned with LOS |
| `LEAD_PREDICT_WEIGHT` | 3.0 | **25.0** | Nose pointed at future intercept |
| `LOS_RATE_WEIGHT` | — | **20.0** 🆕 | Collision-course damping (λ̇ → 0) |
| `STEP_PENALTY` | — | **0.3** 🆕 | Per-decision survival cost |

**Result:** Guidance terms went from 4% → 88% of total reward signal.

### v7: Anti-Stall + Velocity Shaping + Speed Advantage
| Component | Config | Purpose |
|-----------|--------|---------|
| Anti-Stall Truncation | 5s window, Vc < 2m/s, dist > 300m → truncate -100 | Eliminate "comfortable drift" |
| Dynamic Velocity Shaping | cos(ATA) > 0.95 → reward high airspeed | "If aligned, go fast" |
| Speed Advantage | Pursuer max 270 m/s, Target max 160 m/s (69%) | Physical interceptability |

### LOS-Rate (λ̇) Reward — the core innovation
```
λ̇ = |v_rel_perp| / dist    (rad/s)
reward = exp(-|λ̇| × 5) × 20 × dt

|λ̇| = 0    → reward = 20·dt   (perfect collision course)
|λ̇| = 0.1  → reward = 12·dt   (gentle correction)
|λ̇| = 0.5  → reward = 1.6·dt  (significant deviation)
|λ̇| = 1.0  → reward ≈ 0       (pure pursuit, no intercept)
```

---

## 5. Curriculum: Auto-Curriculum with Sliding Window

### Architecture
- Replaced discrete 5-stage curriculum with continuous `difficulty_level ∈ [0, 1]`
- All target maneuver parameters interpolate smoothly: speed (130→160 m/s), heading rate (0→12°/s), altitude rate (0→4 m/s), spawn distance, bearing offset, heading difference
- Backward-compatible `curriculum_stage` property

### AutoCurriculumCallback
- **Sliding window**: deque of last 100 episode outcomes (success/failure)
- **Spring mechanism** (softened v2):
  - wr ≥ 80% → difficulty += 0.05 (aggressive push)
  - 50% ≤ wr < 80% → difficulty += 0.02 (gentle push)
  - 25% ≤ wr < 50% → maintain (wide flow zone)
  - 10% ≤ wr < 25% → difficulty -= 0.01 (gentle nudge)
  - wr < 10% → difficulty -= 0.02 (moderate retreat)
- **Physical telemetry**: `difficulty_physical` logged to verify env receives difficulty
- **Curriculum floor**: `MIN_DIFFICULTY = 0.15` — no trivial straight-line targets
- Minimum 20K steps between difficulty changes

### Critical Bug Found & Fixed
`ResidualExpertWrapper.difficulty_level` setter was writing to wrapper instance (creating shadow attribute) instead of delegating to `self.unwrapped`. All previous auto-curriculum results were at actual `difficulty=0.0` regardless of displayed value. **Fixed by using `self.unwrapped`.**

---

## 6. Model Saving

Per-stage best models with `capture_rate > 0` guard:
- `best_stage_1_0.zip` — best at Stage 1.0
- `best_stage_1_5.zip` — best at Stage 1.5
- `best_model.zip` — global best across all stages
- `final_model.zip` — weights at training end

---

## 7. Key Results

### Ablation Ranking (2Hz, 200K steps, ent_coef=0.0)
| Rank | Variant | S1.5 Best | S1.5 Avg Dist | Consistency |
|:---:|------|:---:|:---:|:---:|
| 1 | CA (Cubic Action) | 20% | 688m | 3/10 |
| 2 | FS (Frame Stack) | 10% | 909m | 1/10 |
| 3 | RW (Lead Pursuit) | 7% | 717m | 2/10 |
| 4 | BL (Baseline) | 3% | 1009m | 1/10 |

### CARW @ 10Hz + v7 Rewards (60K steps, real diff=0.15)
| Metric | Before (Bug, diff=0.0) | After (Fixed, diff=0.15) |
|--------|:---:|:---:|
| Best capture | "100%" (fake) | 63% (real) |
| Timeout rate | 0% | 0% |
| Stall rate | 0% | 37-67% |
| Difficulty advancement | "0.15→0.19" (fake) | 0.15→0.15 (stuck, correct) |

### Terminal-Phase Action Distribution (diff=0.15, <300m)
- Raw policy output: mean=+0.04, std=0.004, 100% within |a|<0.2
- Post-cubic: essentially no-op (~0.0001)
- **No Bang-Bang control** — but strategy collapse to fixed trim at low difficulty

---

## 8. Files Changed

| File | Changes |
|------|---------|
| `src/environment/single_pursuit_env.py` | Continuous difficulty, 25-dim obs, real JSBSim body rates, AoA/Ps, anti-stall truncation, velocity shaping, step penalty, reward rebalance |
| `src/environment/ablation_wrappers.py` | LeadPursuitRewardWrapper (λ̇ reward, rebalanced weights), BlendedActionWrapper, CubicActionWrapper |
| `src/environment/__init__.py` | Export new wrappers |
| `scripts/train_single_pursuit.py` | AutoCurriculumCallback, ResidualExpertWrapper fixes, physical telemetry, softened spring |
| `scripts/run_ablation_study.py` | Auto-curriculum integration, gamma fix, reward CSV columns |
| `scripts/evaluate_and_plot.py` | Tacview + guidance metrics + trajectory visualization |
| `tests/test_environment/test_ablation_wrappers.py` | Updated for 25-dim observation |

---

## 9. Next Steps

1. **Run full auto-curriculum training** (200K+ steps) with fixed difficulty delegation to observe real difficulty climb trajectory
2. **Monitor action variance** at higher difficulties — if policy collapses again, switch to BlendedActionWrapper
3. **Increase ent_coef** if policy degradation pattern persists
4. **Multi-seed validation** (seeds 0, 1, 2) once hyperparameters stabilize
5. **Expand reset geometry diversity** — add head-on and crossing engagement scenarios
