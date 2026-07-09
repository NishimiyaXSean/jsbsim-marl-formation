# Continuous → Discrete Action Space Migration — Priority Action Plan

> **Date**: 2026-07-09
> **Context**: AFTER completing Experiment 3 v3 (dynamic AND annealing), the CTDE MAPPO showed clear improvement (eval -1,171 vs -5,909) but AND-gate success remains elusive at strict 800m. The hypothesis: **continuous action space creates exploration noise too high for the precise temporal synchronization AND-gate requires**. Discrete actions constrain exploration to semantically meaningful tactical primitives, reducing sample complexity.

---

## Why Discrete?

| Continuous (current) | Discrete (target) |
|---|---|
| Box(2): unbounded DiagGaussian | MultiDiscrete: categorical choice |
| Action clamping needed to prevent NaN | No clamping — actions always valid |
| Exploration diffuses in 2D manifold | Exploration is structured: choice among tactical primitives |
| PPO with Gaussian head | PPO with Categorical head |
| Gradients noisy → entropy runaway (3.6) | Gradients focused → entropy bounded by log(N) |
| SB3 succeeded with discrete actions | Aligns with proven SB3 architecture |

---

## Priority 0: Action Space Design (Design Doc — No Code)

**Proposed: MultiDiscrete([5, 3]) = 15 tactical primitives**

```
Turn dimension (5 choices):
  0: Hard left  (−15°/s)
  1: Soft left  (−5°/s)
  2: Straight   (0°/s)
  3: Soft right (+5°/s)
  4: Hard right (+15°/s)

Speed dimension (3 choices):
  0: Slow   (180 m/s ≈ 350 kts)  — energy-saving / loiter
  1: Cruise (250 m/s ≈ 486 kts)  — balanced pursuit
  2: Fast   (320 m/s ≈ 622 kts)  — afterburner chase
```

**Why 5×3 = 15 actions?**
- 5 turn options covers the full steering range without over-fragmentation
- 3 speed options maps to tactical regimes (conserve / cruise / burn)
- 15 total actions is small enough for Categorical PPO, large enough for tactical variety
- SB3's success used 4-dim Box — we're preserving tactical expressiveness while eliminating continuous noise

**Alternative considered**: Discrete(9) = single combined action. Rejected — decoupling turn and speed preserves the semantic structure of the continuous action space.

---

## Priority 1: Environment Adaptation (1 file, ~30 lines)

**File**: `src/environment/formation_rllib_env.py`

**Changes**:
1. Replace `single_act = Box(-1, 1, (2,))` with `MultiDiscrete([5, 3])`
2. Replace action parsing in `step()`:

```python
# OLD (continuous):
actions[aid] = {
    'turn': float(a[0]),
    'speed': float(a[1]),
    'cmd_turn_rate': float(a[0] * 15.0),
    'cmd_speed': float(250.0 + a[1] * 100.0),
}

# NEW (discrete):
TURN_RATES = [-15.0, -5.0, 0.0, 5.0, 15.0]   # deg/s
SPEEDS     = [180.0, 250.0, 320.0]             # m/s
turn_idx, speed_idx = int(a[0]), int(a[1])
actions[aid] = {
    'turn': 0.0,  # not used by FlightController
    'speed': 0.0,
    'cmd_turn_rate': TURN_RATES[turn_idx],
    'cmd_speed': SPEEDS[speed_idx],
}
```

3. Update observation/action space inspection (probe) — handle MultiDiscrete
4. Add `TURN_RATES` and `SPEEDS` module-level constants

**Estimated effort**: 15 min, 1 file

---

## Priority 2: Model Adaptation (1 file, ~10 lines)

**File**: `src/models/formation_rllib_model.py`

**Changes**:
1. `num_outputs` calculation: `2 * act_dim` → `sum(action_space.nvec)` = 5 + 3 = 8
2. Actor head: replace `mean(batch, 2) + log_std(1, 2)` with dual categorical logits
3. RLlib's `ActionDistribution` auto-switches from DiagGaussian to MultiCategorical when action space changes

```python
# In __init__:
assert num_outputs == sum(action_space.nvec), \
    f"Expected {sum(action_space.nvec)} outputs, got {num_outputs}"

# In forward() — replace loc/scale with:
turn_logits = self.turn_head(feat)    # [B, 5]
speed_logits = self.speed_head(feat)  # [B, 3]
logits = torch.cat([turn_logits, speed_logits], dim=1)  # [B, 8]
```

**Architecture change**: `AttentionFormationActor` MLP head splits into two output branches:
```
feat [B, 256]
  ├── turn_head: Linear(256, 5)  → turn_logits
  └── speed_head: Linear(256, 3) → speed_logits
```

**Estimated effort**: 20 min, 1 file

---

## Priority 3: BC Data Conversion (1 script, ~50 lines)

**Problem**: BC data (`attention_bc_2v1_filtered_pretrained.pth`) was trained on continuous actions. The pretrained AttentionFormationActor has:
- `mean.weight`: [2, 256] — continuous action mean
- `log_std`: [1, 2] — continuous action log_std

Discrete model needs:
- `turn_head.weight`: [5, 256]
- `speed_head.weight`: [3, 256]

**Approach**: Don't convert BC weights — they're fundamentally incompatible. Instead:

1. Train a NEW BC model on discretized expert data
2. Use SB3's 97.3% Phase 4.1 model to generate expert trajectories
3. Discretize the expert actions: map continuous [turn, speed] → closest (turn_idx, speed_idx)
4. Train BC on discretized data (same `train_attention_bc_2v1.py` pipeline)

**Quick alternative**: Skip BC pretraining for initial discrete experiments. RLlib PPO with Categorical head should explore efficiently from scratch for a 15-action space.

**Decision**: Start WITHOUT BC pretraining (Priority 4 first). If cold-start fails, add discretized BC.

**Estimated effort**: 1 hour if needed, 0 if skipped

---

## Priority 4: Training Script Adaptation (1 file, ~15 lines)

**File**: `scripts/train_formation_rllib.py`

**Changes**:
1. Action space probing: handle MultiDiscrete (not Box)
2. `--load-bc` flag: skip BC loading for initial experiments (shape mismatch)
3. Update print messages
4. Training hyperparameters: increase `entropy_coeff` slightly (0.01→0.02) to encourage discrete exploration

**Estimated effort**: 10 min, 1 file

---

## Priority 5: Visualization Script Adaptation (3 files, ~20 lines each)

**Files**: `collect_viz_data.py`, `viz_paper_figures.py`, `viz_fig3_role_attention.py`

**Changes**: Handle discrete action format in trajectory collection and plotting:
- Actions stored as `[turn_idx, speed_idx]` instead of `[turn, speed]`
- Plot labels: map indices to human-readable tactical labels

**Estimated effort**: 15 min, 3 files

---

## Priority 6: Smoke Test & Experiment 4 (Discrete OR-gate)

1. Smoke test: 5 iterations, verify no crashes
2. Experiment 4a: Discrete OR-gate (120 iters) — replicate Exp 2 baseline
3. Experiment 4b: Discrete AND-gate with dynamic annealing (300 iters)

**Estimated effort**: 5 min smoke + ~10 hours training

---

## Execution Order

```
Phase 1 (Today, ~1 hour):
  ✅ P1.1: Update formation_rllib_env.py (discrete action space)
  ✅ P1.2: Update formation_rllib_model.py (categorical heads)
  ✅ P1.3: Update train_formation_rllib.py (handle MultiDiscrete)
  ✅ P1.4: Smoke test (5 iterations)

Phase 2 (Today, ~2 hours):
  ✅ P2.1: Experiment 4a — Discrete OR-gate (120 iters)
           → Verify discrete MAPPO matches continuous baseline

Phase 3 (Today/Tomorrow, ~8 hours):
  ✅ P3.1: Experiment 4b — Discrete AND-gate + dynamic annealing
           → The REAL test: does discrete action space enable AND-gate success?
```

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Discrete too coarse for F-16 control | Low | FlightController PID smooths any command; 5×3 grid covers tactical range |
| BC pretraining incompatible | Medium | Skip BC for initial experiments; cold-start Categorical PPO should converge |
| RLlib MultiCategorical API issues | Medium | Smoke test catches early; fallback: Discrete(15) flatten |
| Discrete doesn't help AND-gate | Medium | If 4b fails, the bottleneck is architectural (33-dim obs), not action space |

## Success Criteria

- **Minimum**: Discrete OR-gate matches continuous baseline (+5,000 eval)
- **Target**: Discrete AND-gate eval > 0 (first-ever AND-gate positive eval)
- **Stretch**: Discrete AND-gate eval matches OR-gate level (+3,000+)
