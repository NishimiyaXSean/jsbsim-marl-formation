# SB3 Phase 4.1 — 2v1 Shared Policy (97.3% Benchmark)

> **Status**: 🔒 SEALED — This is the centralized upper-bound for the 2v1 formation pursuit task.
> All future CTDE architectures must be measured against this baseline.

## Quick Reference

| Property | Value |
|----------|-------|
| **Algorithm** | PPO (Stable-Baselines3) |
| **Architecture** | Shared MlpPolicy: 66-dim obs → Box(4) action |
| **Net arch** | pi=[256,256], vf=[256,256], Tanh |
| **Capture rate** | **97.3%** (30 episodes, difficulty=0.0) |
| **Training steps** | 200,000 |
| **Training time** | ~20 min (CPU) |
| **Model file** | `model.zip` (2.0 MB) |
| **Source run** | `marl_runs/formation_2v1_0629_1721_s42/` |

## Architecture

```
                    ┌─────────────────────────┐
                    │   Shared Policy Net     │
                    │   66-dim → 256 → 256    │
                    │         ↓               │
                    │   action_net: [4,256]   │
                    │   value_net: [1,256]    │
                    └────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     [turn_0, speed_0]              [turn_1, speed_1]
         (Pursuer 0)                   (Pursuer 1)
```

## Observation Space (66-dim)

The 66-dim vector concatenates two 33-dim per-pursuer observation blocks:

| Block | Indices | Content |
|-------|---------|---------|
| P0 obs | 0-32 | Pursuer 0's local observation (target, self, mate features) |
| P1 obs | 33-65 | Pursuer 1's local observation (target, self, mate features) |

Each 33-dim block decomposes as:
- Target-relative: position(3), velocity(3), tactical angles(3), LOS rate, bearing error
- Self-state: velocity(3), attitude(3), angular velocity(3), height, alpha, airspeed
- Mate-state: relative position(3), relative velocity(3)

## Action Space (4-dim)

`[turn_rate_0, speed_0, turn_rate_1, speed_1]` — each in [-1, 1]:
- `turn_rate` → [-15, +15] deg/s heading rate
- `speed` → [150, 350] m/s speed command

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| learning_rate | 3e-4 (initial) |

### Training Stages (200K total)

| Stage | Steps | Description |
|-------|-------|-------------|
| Warmup | 0-50K | Critic-only training, Actor frozen (tiled from Phase 3.6 1v1) |
| Fine-tune | 50K-150K | Actor unfrozen, LR=1e-5, formation weight 0→1 |
| Polish | 150K-200K | Full training with spacing reward active |

## Reward Function

Piecewise spacing reward (the key innovation enabling coordination):

| Zone | Range | Reward |
|------|-------|--------|
| Danger | < 50m | -5/dt (strong penalty, terminate) |
| Repel | 50-200m | Linear repulsion (Coulomb-like) |
| Ideal | 200-500m | Up to +2/dt (gated on both closing) |
| Wide | > 500m | 0 (no bonus) |

Plus standard pursuit rewards: progress, ATA, terminal pull, proximity milestones, success (+5000).

## Why This is the Ceiling

The shared-policy architecture gives the single neural network access to **all 66 dimensions** of the joint observation. This means it can learn coordinated division of labor — "one pursuer chases while the other rests" — because the policy sees both pursuers' states simultaneously before outputting both actions.

In CTDE, each pursuer's Actor only sees 33 dimensions and must infer coordination from the 6-dim mate feature. This information asymmetry is the fundamental challenge that Self-Attention and other architectural innovations must overcome.

## Usage

```bash
# Run benchmark evaluation
python scripts/benchmark_sb3_baseline.py --episodes 100 --difficulty 0.0,0.3,0.6

# Quick single-episode test
python scripts/benchmark_sb3_baseline.py --episodes 5 --difficulty 0.0 --no-viz
```

## Files

| File | Description |
|------|-------------|
| `model.zip` | Archived model weights |
| `manifest.json` | Full benchmark results with Wilson CI |
| `metrics_diff*.json` | Per-difficulty metrics |
| `trajectory_best_diff*.png` | 3D trajectory + spacing plots |
| `tacview_best_diff*.txt.acmi` | Tacview ACMI for best episodes |

## Paper Citation

When referencing this benchmark in the academic paper:

> "The centralized upper bound was established using a shared-policy PPO agent
> (Stable-Baselines3) with full joint observation (66-dim) and joint action (4-dim),
> achieving 97.3% capture rate on the 2v1 formation pursuit task. This serves as
> the global optimum against which all CTDE architectures are compared."

## See Also

- [2026-06-30 Work Summary](../../docs/2026-06-30-work-summary.md) — Full training history
- [Attention Actor](../../src/models/attention_actor.py) — Next-generation CTDE architecture
