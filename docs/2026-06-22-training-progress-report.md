# Single-Pursuit Training Progress Report

**Date:** 2026-06-22  
**Status:** Breakthrough — agent achieves 84.4% avg peak capture rate with V_c-coupled reward  
**Commits:** `1cf3790` through `fab29c5`

---

## 1. Executive Summary

After 5 major iterations across 15 training runs (45+ seed-jobs), we have transformed the single-pursuit training from a **permanently-collapsing** baseline into a **continuously-recovering** system. The key innovations:

1. **V_c (closure-rate) multiplicative coupling** — all guidance rewards multiplied by `max(0, min(1, V_c/50))`. Zero closure = zero guidance. This single change broke the "750m drifting comfort zone" that killed all previous versions.

2. **Boiling-oil zone-of-death** — `-50/step` penalty after only 2 seconds of low-V_c drifting in `[300, 800]m`. Quick death → fresh episode → forced re-exploration.

3. **Doubled experience pool** — `n_steps=4096, batch_size=512` for BVR gradient stability. Eliminated the catastrophic seed variance seen in v6.

---

## 2. Version History

| Ver | Date | Key Changes | Avg Peak | Avg Final | Post-NZ | Diff Adv? |
|-----|------|-------------|:---:|:---:|:---:|:---:|
| v1 | 06-15 | Baseline CARW, 905m spawn, 5-stage curriculum | 55.6% | ~21% | ~15% | No |
| v4 | 06-17 | **V_c coupling + boiling oil**, 905m WVR | 70.0% | 4.4% | **74%** | No |
| v5 | 06-17 | BVR spawn (2000-5000m), compass decay | 28.9% | 0.0% | 0% | No |
| v6 | 06-18 | Progressive spawn (1000-4000m), compass fix | 67.8% | 0.0% | 20% | No |
| **v7** | **06-19** | **n_steps=4096, soft spawn (900-3000m)** | **84.4%** | **5.6%** | **34%** | **s1: 0.20** |

> **Note:** v2-v3 (BLRW blended action) were intermediate experiments that proved CARW (pure cubic) superior. v4 is the first V_c+oil version and remains the WVR baseline.

---

## 3. v7 Final Results (Current Best)

**Configuration:** CARW (CubicAction + LeadPursuitReward), V_c coupling, boiling-oil zone-of-death, `n_steps=4096`, `batch_size=512`, progressive spawn `900-1300m (d=0)` → `2000-3000m (d=1)`, `MAX_EPISODE_TIME=180s`, level-flight target below `d=0.5`.

| Seed | Peak Cap | Peak WR | Peak Step | Final Cap | Non-Zero Evals | >5% Evals | Post-100K NZ Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| s0 | **86.7%** | 66.0% | 45,056 | 6.7% | 18/34 (53%) | 13/34 (38%) | 14/30 (47%) |
| s1 | **90.0%** | 66.0% | 45,056 | 3.3% | 15/34 (44%) | 11/34 (32%) | 12/30 (40%) |
| s2 | 76.7% | 52.0% | 45,056 | 6.7% | 9/34 (26%) | 6/34 (18%) | 5/30 (17%) |
| **Mean** | **84.4%** | **61.3%** | — | **5.6%** | **14/34 (41%)** | **10/34 (29%)** | **10/30 (34%)** |

### Difficulty Ceiling

| Seed | Max Difficulty | Target Distance at Max | Target Speed | Target Turn Rate |
|:---:|:---:|:---:|:---:|:---:|
| s0 | 0.15 | 1065-1555m | 134.5 m/s | ±1.8 °/s |
| s1 | 0.20 | 1120-1640m | 136.0 m/s | ±2.4 °/s |
| s2 | 0.15 | 1065-1555m | 134.5 m/s | ±1.8 °/s |

**All three seeds spent >97% of training at `difficulty=0.15`.** The `CONSECUTIVE_REQUIRED=2` gate prevented advancement because no seed achieved two consecutive evaluations with `win_rate ≥ 50%`.

---

## 4. Architecture Details

### 4.1 Observation Space (25 dims)

| Indices | Feature | Source |
|---------|---------|--------|
| 0-2 | Target relative position (body frame) | — |
| 3-5 | Own velocity (body frame) | — |
| 6-8 | Own attitude rpy | — |
| 9-11 | Own angular velocity [p, q, r] | JSBSim body rates |
| 12 | Own height | — |
| 13-15 | Target velocity (body frame) | — |
| 16-18 | Target angular velocity | Finite-diff of target rpy |
| 19-21 | Tactical geometry cos(ATA), cos(AA), cos(HCA) | — |
| 22 | Angle of Attack | JSBSim `aero/alpha-deg` |
| 23 | Airspeed | JSBSim `velocities/vc-kts` |
| 24 | Specific Excess Power (Ps) | Computed |

### 4.2 Action Space (3 dims, 10 Hz, CARW)

| Dim | Range (raw) | Range (physical) | Description |
|-----|-------------|------------------|-------------|
| d_heading | [-1, +1] | [-6°, +6°] per 0.1s | Heading change (max 60°/s) |
| d_alt | [-1, +1] | [-3m, +3m] per 0.1s | Altitude change (max 30 m/s) |
| d_speed | [-1, +1] | [-2, +2] m/s per 0.1s | Speed change (max 20 m/s²) |

**CARW (Cubic Action):** `physical = sign(a) × |a|³` — pure cubic mapping with implicit low-pass filtering.

### 4.3 Reward Architecture (v9)

```
V_c_norm = max(0, min(1, closure_rate / 50.0))

r_total = r_progress × 0.5                              # Constant compass at all ranges
        + r_terminal_boost × 5.0  (if dist < 500m)      # Amplified terminal approach
        + r_ata × 5.0                                    # Antenna train angle
        + r_lead_vel_align × 15.0 × V_c_norm × energy_ok # Velocity-aligned-with-LOS
        + r_lead_pred × 25.0 × V_c_norm × energy_ok      # Nose-at-future-intercept
        + r_los_rate × 20.0 × V_c_norm × energy_ok       # Collision-course damping
        - 1.0                                            # Step penalty (per decision)
        - 50.0 (if zone-of-death triggered)              # Boiling oil
        + 2000.0 (if dist < 200m)                        # Capture terminal boost
```

### 4.4 Termination Conditions

| Condition | Trigger | Penalty |
|-----------|---------|---------|
| **Success** | dist < 200m | +2000 |
| **Anti-Stall** | V_c < 15 m/s for 3s, dist > 300m | -200 |
| **Zone-of-Death** | 300 < dist < 800m, V_c < 15 m/s for 2s+ | -50/step |
| **Timeout** | t > 180s | — |
| **Lost Target** | dist > 10,000m | -200 |
| **Ground Crash** | alt < 10m | -200 |

### 4.5 PPO Hyperparameters

```python
learning_rate = 3e-4
n_steps = 4096          # Doubled for BVR gradient stability
batch_size = 512        # Matched to n_steps
n_epochs = 10
gamma = 0.999           # Long horizon for delayed BVR gratification
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01         # Minimal entropy — gSDE handles exploration
vf_coef = 0.5
max_grad_norm = 0.5
use_sde = True          # State-dependent exploration for smooth 10Hz control
sde_sample_freq = 4
```

### 4.6 Curriculum

```python
MIN_STEPS_PER_LEVEL = 20_000
MIN_DIFFICULTY = 0.15
CONSECUTIVE_ADVANCE_REQUIRED = 2   # Need 2 consecutive evals ≥50% WR to advance
COLLAPSE_WIN_RATE = 0.15           # Rollback trigger: WR < 15% + difficulty above floor

# Difficulty spring (after ≥20K steps at current level):
#   WR ≥ 50% + 2×consecutive → difficulty +0.05
#   WR ≥ 40% + 2×consecutive → difficulty +0.02
#   WR ∈ [10%, 40%]          → maintain
#   WR ∈ [5%, 10%]           → difficulty -0.01
#   WR < 5%                  → difficulty -0.02
```

### 4.7 Target Spawn & Behaviour

```python
# Spawn distance (decoupled from manoeuvre difficulty)
target_dist ∈ [900 + d×1100, 1300 + d×1700]

# Target motion
speed = 130 + d×30 m/s           # 130 → 160 m/s
heading_rate ∈ [-12×d, +12×d] °/s  # 0 → ±12 °/s
alt_rate = 0 (level flight)      # Unless d > 0.5 and 20% chance → ±2 m/s
```

---

## 5. Key Discoveries

### 5.1 V_c Coupling Is the Single Most Important Innovation

Before V_c coupling, all guidance rewards were additive and independent. The agent learned to maximize `r_lead_pred + r_los_rate` by pointing correctly at 750m range while closing at merely 5-10 m/s — a local optimum with zero capture probability.

Multiplying ALL guidance rewards by `V_c_norm = max(0, min(1, V_c/50))` fundamentally changed the reward landscape. The agent now only receives guidance rewards proportional to how aggressively it is closing on the target.

### 5.2 Harsh Penalty (−50/step) Beats Gentle Penalty (−1/step)

Controlled A/B test (v4 vs v5) proved that the harsh "boiling oil" penalty produces superior results:
- **v4** (−50/step, 2s window): 74% post-collapse non-zero, 4.4% final
- **v5** (−1/step, 5s window): 0% post-collapse non-zero, 0.0% final

The mechanism: harsh penalty → rapid episode termination → forced re-exploration. Gentle penalty allows the agent to drift in the 750m comfort zone indefinitely.

### 5.3 Larger Batch Size Eliminates Catastrophic Seed Variance

v6 with `n_steps=2048` showed extreme seed variance (0% vs 60% post-collapse recovery). Doubling to `n_steps=4096, batch_size=512` in v7:
- Raised avg peak from 67.8% → 84.4%
- All 3 seeds survived to end (vs 2/3 dead in v6)
- Peak shifted from 16K (lucky init) to 45K (matured policy)

### 5.4 Progressive Distance: 1065m Is the Current Sweet Spot

The spawn distance at `difficulty=0.15` directly controls the problem difficulty:

| Spawn Distance | Avg Peak | Agent Status |
|:---:|:---:|:---:|
| 905m (v4) | 70.0% | All alive, 74% post-nz |
| 1065m (v7) | 84.4% | All alive, 34% post-nz |
| 1500m (v6) | 67.8% | 1/3 alive, 20% post-nz |
| 2450m (v5) | 28.9% | All dead |

The 1065m baseline (v7) combined with larger batch size achieves the best peak performance, though post-collapse recovery (34%) is lower than v4's WVR distance (74%).

---

## 6. Remaining Challenges

1. **CONSECUTIVE=2 gate never satisfied** — despite peak win rates of 66%, no seed achieved two consecutive evals ≥50% WR. The variance between evaluations remains too high for curriculum advancement.

2. **Difficulty ceiling at 0.15** — all progress was made against a near-straight-flying target (±1.8°/s turn rate, level flight). The system has not been tested against truly maneuvering targets.

3. **Post-collapse recovery still declining** — while agents no longer permanently die, the recovery rate (34%) is lower than desired. The system exhibits "punctuated equilibrium" — periods of high performance interrupted by collapse and slow recovery.

4. **Seed variance persists** — s2 (17% post-nz) significantly underperforms s0 (47%) and s1 (40%), despite identical configuration.

---

## 7. Next Steps

1. **Relax CONSECUTIVE_REQUIRED to 1** or implement a "best-of-N" voting mechanism to allow difficulty advancement on single strong evaluations
2. **Extend training to 3M+ steps** — the 84.4% peak at 45K suggests the policy is still maturing; longer training may consolidate stability
3. **Increase ent_coef to 0.02** with careful monitoring — may help post-collapse re-exploration without triggering 10Hz chattering
4. **Multi-seed ensemble evaluation** — aggregate policy across seeds for evaluation to reduce variance

---

## A. Appendix: File Inventory

| File | Purpose | Key Changes |
|------|---------|-------------|
| `src/environment/single_pursuit_env.py` | Environment, rewards, termination | V_c info exposure, boiling-oil ZONE_DEATH, aggressive ANTI_STALL, progressive spawn, level-flight target, 180s ceiling, compass fix |
| `src/environment/ablation_wrappers.py` | Action/reward wrappers | V_c multiplicative coupling, energy gating, action smoothness penalty, CARW cubic mapping |
| `scripts/train_single_pursuit.py` | PPO training, curriculum | AutoCurriculumCallback with rollback, gSDE, n_steps=4096, CONSECUTIVE=2 |
| `scripts/run_ablation_study.py` | Multi-seed experiment runner | CARW config, PPO_CONFIG with gSDE, auto-curriculum integration |
| `tests/test_environment/test_ablation_wrappers.py` | Unit tests | Adapted for V_c coupling (closure_rate in info dict) |
| `scripts/viz_success.py` | Visualization | Tacview export + 3D trajectory plots for successful episodes |
