# Phase 1 Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the single-pursuit training pipeline with PN guidance expert, tuned rewards, 5-stage curriculum, hyperparameter sweep runner, and multi-seed evaluation to achieve ≥90% Stage 3 capture rate.

**Architecture:** Four files modified, one new file created. Phase 1a upgrades `_compute_expert` in the training script to use PN guidance and tunes reward weights/micro-step rewards in the env. Phase 1b refactors curriculum to 5 float stages, builds a sweep runner script, and extends evaluation with multi-seed aggregation.

**Tech Stack:** Python 3.10+, Stable-Baselines3 PPO, JSBSim F-16 FDM, NumPy, Gymnasium

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/train_single_pursuit.py` | Modify | PN expert, curriculum config, eval config, CSV logging |
| `src/environment/single_pursuit_env.py` | Modify | Reward tuning, 5-stage float curriculum, terminal boost, time pressure |
| `scripts/run_hyperparam_sweep.py` | **Create** | Orthogonal sweep runner (8 configs × 500k, top-2 × 5 seeds) |
| `scripts/evaluate_and_visualize.py` | Modify | Multi-seed aggregation, Wilson CI, CSV export |

---

### Task 1: PN Guidance Expert in ResidualExpertWrapper

**Files:**
- Modify: `scripts/train_single_pursuit.py:35-72`

- [ ] **Step 1: Replace `_compute_expert` with PN guidance version**

Replace the current `_compute_expert` method and `__init__` in `ResidualExpertWrapper`:

```python
class ResidualExpertWrapper(gym.Wrapper):
    """Agent learns residual on top of a PN-guidance expert.

    Expert uses proportional navigation to compute desired heading,
    then a P-controller on heading error drives aileron.
    RL adds ±0.5 residual corrections on top.

    Stores reference to the underlying SinglePursuitEnv to access
    pursuer/target world-frame positions and velocities.
    """
    RESIDUAL_SCALE = 0.5

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        self._last_obs = np.zeros(19, dtype=np.float32)
        # Store reference to the base SinglePursuitEnv for world-frame data
        self._base_env = env   # unwrapped SinglePursuitEnv

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        expert = self._compute_expert()
        residual = np.asarray(action, dtype=np.float32) * self.RESIDUAL_SCALE
        combined = np.clip(expert + residual, -1.0, 1.0)
        obs, rew, term, trunc, info = self.env.step(combined)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, rew, term, trunc, info

    def _compute_expert(self) -> np.ndarray:
        """PN guidance expert: compute desired heading via PN, then P-controller."""
        from src.utils.pn_guidance import compute_pn_heading

        env = self._base_env
        pursuer_ned = env.pursuer.position_ned
        pursuer_vel = env.pursuer.velocity_ned
        target_ned = env.target_ac.position_ned
        target_vel = env.target_ac.velocity_ned
        current_heading = float(env.pursuer.state["yaw_deg"])

        desired_heading = compute_pn_heading(
            pursuer_ned, pursuer_vel, target_ned, target_vel,
            current_heading_deg=current_heading,
            dt=0.5, nav_constant=3.0, max_turn_rate_dps=15.0,
        )

        # Wrap heading error to [-180, 180]
        heading_error = (desired_heading - current_heading + 180.0) % 360.0 - 180.0
        ail = float(np.clip(heading_error * 0.05, -0.3, 0.3))
        return np.array([ail, 0.0, 1.0], dtype=np.float32)

    # Delegate curriculum_stage to underlying env
    @property
    def curriculum_stage(self):
        return self.env.curriculum_stage

    @curriculum_stage.setter
    def curriculum_stage(self, value):
        self.env.curriculum_stage = value
```

- [ ] **Step 2: Update the env construction so `_base_env` gets the base env, not the Monitor**

In `train()` function, make sure `base_env` is the naked `SinglePursuitEnv`:

```python
base_env = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)
env = ResidualExpertWrapper(base_env)   # _base_env = base_env (SinglePursuitEnv)
env = Monitor(env, log_dir)             # Monitor wraps ResidualExpertWrapper
```

For the eval env, the same pattern — but we need `eval_base` accessible:

```python
eval_base = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)
eval_env = ResidualExpertWrapper(eval_base)
```

The `eval_env` passed to `CurriculumCallback` is the `ResidualExpertWrapper` — `CurriculumCallback._eval_env.step()` will call `_compute_expert()` which reads from `_base_env`. This already works because the callback's eval loop calls `self._eval_env.step(action)`.

- [ ] **Step 3: Quick smoke test — verify PN expert doesn't crash**

Run: `cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && JSBSIM_DEBUG=0 python -c "
import sys, os
sys.path.insert(0, '.')
import numpy as np
from src.environment.single_pursuit_env import SinglePursuitEnv
from scripts.train_single_pursuit import ResidualExpertWrapper

env = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)
wrapper = ResidualExpertWrapper(env)
obs, _ = wrapper.reset()
expert = wrapper._compute_expert()
print(f'Expert action: {expert}')
print(f'Obs shape: {obs.shape}')
obs2, rew, term, trunc, info = wrapper.step(np.zeros(3))
print(f'Step: rew={rew:.2f}, term={term}, info={info}')
print('PN expert smoke test PASSED')
"`

Expected: Expert action printed (aileron between -0.3 and 0.3), no crashes, step completes.

---

### Task 2: Reward Function Tuning

**Files:**
- Modify: `src/environment/single_pursuit_env.py:85-93`

- [ ] **Step 1: Update reward weight constants**

Replace the reward weight block (lines 85-93):

```python
# Reward weights
REWARD_PROGRESS = 5.0         # stronger primary pursuit signal (was 2.0)
REWARD_ATA = 5.0              # stronger pointing incentive (was 3.0)
REWARD_ALTITUDE_BONUS = 0.0   # disabled
REWARD_ENERGY_PENALTY = 0.0   # disabled
REWARD_GROUND_WARNING = 2.0
REWARD_SUCCESS = 500.0        # unchanged
REWARD_CRASH = -200.0         # unchanged
REWARD_LOST_TARGET = -200.0   # unchanged
```

- [ ] **Step 2: Add terminal boost inside micro-step loop**

In `step()`, after the progress reward line (`total_reward += REWARD_PROGRESS * delta_dist`), add:

```python
            # Terminal boost: extra progress reward when close to target
            if current_dist < 500.0:
                total_reward += REWARD_PROGRESS * delta_dist * 2.0
```

Find the existing line in the micro-step loop:
```python
            total_reward += REWARD_PROGRESS * delta_dist
```

Replace it with:
```python
            total_reward += REWARD_PROGRESS * delta_dist
            # Terminal boost: extra closing reward within 500m
            if current_dist < 500.0:
                total_reward += REWARD_PROGRESS * delta_dist * 2.0
```

The exact old_string to match (at approximately line 345):

```python
            # Progress: closing distance (positive when closing)
            delta_dist = self._prev_dist - current_dist
            total_reward += REWARD_PROGRESS * delta_dist
```

Replace with:

```python
            # Progress: closing distance (positive when closing)
            delta_dist = self._prev_dist - current_dist
            total_reward += REWARD_PROGRESS * delta_dist
            # Terminal boost: extra closing reward within 500m to encourage aggressive terminal phase
            if current_dist < 500.0:
                total_reward += REWARD_PROGRESS * delta_dist * 2.0
```

- [ ] **Step 3: Add time pressure penalty inside micro-step loop**

After the energy penalty line (~line 354), add the time pressure penalty. Find:

```python
            # Energy: penalty for rapid throttle changes
            total_reward -= REWARD_ENERGY_PENALTY * abs(float(thr) - 0.8) * dt
```

Replace with:

```python
            # Energy: penalty for rapid throttle changes
            total_reward -= REWARD_ENERGY_PENALTY * abs(float(thr) - 0.8) * dt

            # Time pressure: small penalty that grows over time to discourage leisurely pursuit
            time_ratio = self._step_counter / (CTRL_FREQ * MAX_EPISODE_TIME)
            total_reward -= 0.5 * time_ratio * dt
```

- [ ] **Step 4: Verify reward changes with a smoke test**

Run: `cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && JSBSIM_DEBUG=0 python -c "
import sys, os
sys.path.insert(0, '.')
from src.environment.single_pursuit_env import SinglePursuitEnv
env = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)
obs, _ = env.reset()
total_rew = 0.0
for i in range(300):  # 5 seconds
    act = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(act)
    total_rew += rew
    if term or trunc:
        print(f'Episode ended at step {i}: {info[\"reason\"]}')
        break
print(f'Total reward after {i+1} steps: {total_rew:.2f}')
print('Reward smoke test PASSED')
"`

Expected: No crashes, reward values printed (should be larger in magnitude than before due to REWARD_PROGRESS=5.0).

---

### Task 3: 5-Stage Float Curriculum in Environment

**Files:**
- Modify: `src/environment/single_pursuit_env.py:110-129` (curriculum_stage type), `183-220` (reset spawning), `440-464` (_generate_target_profile)

- [ ] **Step 1: Change `curriculum_stage` from int to float in `__init__`**

In `__init__`, change signature and storage:

```python
    def __init__(
        self,
        curriculum_stage: float = 1.0,   # was int = 1
        jsbsim_data_dir: Optional[str] = None,
        record_tacview: bool = False,
    ):
```

The `self.curriculum_stage = curriculum_stage` line stays the same — it now stores a float.

- [ ] **Step 2: Update reset() target spawning for 5 stages**

Replace the curriculum-based spawning block in `reset()` (lines 183-220). The old block has `if self.curriculum_stage == 1:`, `elif self.curriculum_stage == 2:`, `else:` (stage 3). Replace with:

```python
        # --- Target spawn (5-stage float curriculum) ---
        stage = self.curriculum_stage

        # Use np.isclose for float stage matching
        if np.isclose(stage, 1.0):
            target_dist = rng.uniform(800, 1800)
            bearing_offset = 0.0
            target_alt_offset = rng.uniform(-50, 50)
            heading_diff = 0.0
        elif np.isclose(stage, 1.5):
            target_dist = rng.uniform(900, 2000)
            bearing_offset = rng.uniform(-7, 7)
            target_alt_offset = rng.uniform(-75, 75)
            heading_diff = rng.uniform(-10, 10)
        elif np.isclose(stage, 2.0):
            target_dist = rng.uniform(1000, 2500)
            bearing_offset = rng.uniform(-15, 15)
            target_alt_offset = rng.uniform(-150, 150)
            heading_diff = rng.uniform(-20, 20)
        elif np.isclose(stage, 2.5):
            target_dist = rng.uniform(1200, 2700)
            bearing_offset = rng.uniform(-30, 30)
            target_alt_offset = rng.uniform(-225, 225)
            heading_diff = rng.uniform(-25, 25)
        else:  # stage 3.0
            target_dist = rng.uniform(1500, 3000)
            bearing_offset = rng.uniform(-45, 45)
            target_alt_offset = rng.uniform(-300, 300)
            heading_diff = rng.uniform(-30, 30)

        target_bearing = (pursuer_hdg + bearing_offset) % 360.0
        target_bearing_rad = np.deg2rad(target_bearing)
        target_ned = np.array([
            pursuer_ned[0] + target_dist * np.cos(target_bearing_rad),
            pursuer_ned[1] + target_dist * np.sin(target_bearing_rad),
            pursuer_ned[2] + target_alt_offset,
        ])
        target_hdg = (pursuer_hdg + heading_diff) % 360.0
```

- [ ] **Step 3: Update _generate_target_profile for 5 stages**

Replace the `_generate_target_profile` method body (lines 440-464):

```python
    def _generate_target_profile(self, rng: np.random.Generator, spawn_heading: float = 90.0) -> TargetProfile:
        """Generate stage-dependent target motion for 5-stage float curriculum."""
        tp = TargetProfile()
        tp.alt_m = 3500.0
        stage = self.curriculum_stage

        if np.isclose(stage, 1.0):
            # Straight and level — no maneuvers
            tp.speed_mps = 130.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = 0.0
            tp.alt_rate_mps = 0.0
        elif np.isclose(stage, 1.5):
            # Very gentle weave
            tp.speed_mps = 145.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(1.5, 4.5) * rng.choice([-1, 1])
            tp.alt_rate_mps = rng.uniform(-1.5, 1.5)
        elif np.isclose(stage, 2.0):
            # Gentle weaving
            tp.speed_mps = 160.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(5, 15) * rng.choice([-1, 1])
            tp.alt_rate_mps = rng.uniform(-3, 3)
        elif np.isclose(stage, 2.5):
            # Moderate weaving
            tp.speed_mps = 170.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-15, 15)
            tp.alt_rate_mps = rng.uniform(-5, 5)
        else:  # stage 3.0
            # Evasive — full maneuvering
            tp.speed_mps = 180.0
            tp.heading_deg = spawn_heading
            tp.heading_rate_dps = rng.uniform(-20, 20)
            tp.alt_rate_mps = rng.uniform(-8, 8)

        return tp
```

- [ ] **Step 4: Commit**

```bash
git add src/environment/single_pursuit_env.py
git commit -m "feat: 5-stage float curriculum with reward tuning (Phase 1a)"
```

---

### Task 4: Update Training Config and CurriculumCallback

**Files:**
- Modify: `scripts/train_single_pursuit.py:88-94` (config), `101-159` (CurriculumCallback), `182-243` (train function)

- [ ] **Step 1: Update training config constants**

Replace the config block (lines 88-94):

```python
TOTAL_TIMESTEPS = 500_000
CURRICULUM_STAGES = [1.0, 1.5, 2.0, 2.5, 3.0]
STAGE_TIMESTEPS = TOTAL_TIMESTEPS // len(CURRICULUM_STAGES)  # 100k per stage

EVAL_EPISODES = 30
EVAL_FREQ = 15_000
TARGET_CAPTURE_RATE_STAGE_1_2 = 0.40   # stage 1.0→1.5 and 1.5→2.0
TARGET_CAPTURE_RATE_STAGE_2_3 = 0.50   # stage 2.0→2.5 and 2.5→3.0
```

- [ ] **Step 2: Update CurriculumCallback for 5 float stages**

Replace the `CurriculumCallback` class (lines 101-159):

```python
class CurriculumCallback(BaseCallback):
    """Handles 5-stage curriculum advancement with automatic evaluation."""

    def __init__(self, eval_env, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._log_dir = log_dir
        self._best_capture_rate = -1.0
        self._current_stage = 1.0
        self._eval_metrics: list[dict] = []  # per-eval metrics for CSV

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Don't eval every rollout — only at multiples of EVAL_FREQ
        if self.num_timesteps % EVAL_FREQ > 2048:
            return

        # Evaluate
        successes, min_dists, intercept_times = 0, [], []
        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done = False
            ep_min_dist = 8000.0
            ep_intercept_time = 120.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if "min_dist" in info:
                    ep_min_dist = min(ep_min_dist, info["min_dist"])
            if info.get("reason") == "success":
                successes += 1
                # Intercept time from env step counter
                base_env = self._eval_env.env  # ResidualExpertWrapper -> SinglePursuitEnv
                ep_intercept_time = base_env._step_counter / 60.0
            min_dists.append(ep_min_dist)
            intercept_times.append(ep_intercept_time)

        capture_rate = successes / EVAL_EPISODES
        avg_min_dist = np.mean(min_dists)
        avg_intercept_time = np.mean(intercept_times)

        self.logger.record("eval/capture_rate", capture_rate)
        self.logger.record("eval/avg_min_dist", avg_min_dist)

        print(f"\n  [Eval @ {self.num_timesteps:,} steps] "
              f"stage={self._current_stage:.1f} "
              f"capture_rate={capture_rate:.0%} "
              f"avg_min_dist={avg_min_dist:.0f}m "
              f"avg_intercept={avg_intercept_time:.1f}s")

        # Record metrics for CSV
        self._eval_metrics.append({
            "timesteps": self.num_timesteps,
            "stage": self._current_stage,
            "capture_rate": capture_rate,
            "avg_min_dist": avg_min_dist,
            "avg_intercept_time": avg_intercept_time,
        })

        # Save best model
        if capture_rate > self._best_capture_rate:
            self._best_capture_rate = capture_rate
            best_path = os.path.join(self._log_dir, "best_model")
            self.model.save(best_path)
            print(f"  → New best model saved: {best_path}")

        # Stage advancement with different thresholds
        threshold = (TARGET_CAPTURE_RATE_STAGE_1_2
                     if self._current_stage < 2.0
                     else TARGET_CAPTURE_RATE_STAGE_2_3)
        if capture_rate >= threshold and not np.isclose(self._current_stage, CURRICULUM_STAGES[-1]):
            # Advance to next stage (0.5 increment)
            current_idx = next(i for i, s in enumerate(CURRICULUM_STAGES)
                               if np.isclose(s, self._current_stage))
            self._current_stage = CURRICULUM_STAGES[current_idx + 1]
            print(f"  >> Advancing to curriculum stage {self._current_stage:.1f}")
            self._eval_env.curriculum_stage = self._current_stage
            self._best_capture_rate = -1.0  # reset for new stage
```

- [ ] **Step 3: Update train() function — PPO config, curriculum stages, save eval CSV**

In `train()`, update:

**PPO model creation** (lines 208-228) — replace net_arch, learning_rate, n_steps, batch_size:

```python
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,           # was 128
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # was [64, 64]
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )
```

**Callback instantiation** — store reference for CSV export:

```python
    curriculum_cb = CurriculumCallback(eval_env, log_dir)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=curriculum_cb,
        progress_bar=False,
    )
```

**After training, save eval CSV** (add after final model save):

```python
    # Save eval metrics CSV
    import csv
    eval_csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                                "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(curriculum_cb._eval_metrics)
    print(f"Eval metrics saved → {eval_csv_path}")
```

**Final evaluation with Tacview** — update eval env stages:

Change the tacview eval part to use the final stage from curriculum callback:
```python
    final_stage = eval_env.curriculum_stage
    tacview_env = SinglePursuitEnv(curriculum_stage=final_stage,
                                   record_tacview=True)
```

And update the eval loop to use `_base_env` for expert computation since `ResidualExpertWrapper._compute_expert` no longer takes `obs`:

```python
    expert_wrapper = ResidualExpertWrapper(tacview_env)
    RES_SCALE = ResidualExpertWrapper.RESIDUAL_SCALE

    for ep in range(EVAL_EPISODES):
        obs, _ = tacview_env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        while not done:
            expert = expert_wrapper._compute_expert()  # no arg now
            residual, _ = model.predict(obs, deterministic=True)
            action = np.clip(expert + np.asarray(residual) * RES_SCALE, -1.0, 1.0)
            obs, rew, terminated, truncated, info = tacview_env.step(action)
            done = terminated or truncated
            total_r += rew
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_single_pursuit.py
git commit -m "feat: 5-stage curriculum, updated PPO config, eval CSV logging (Phase 1b)"
```

---

### Task 5: Phase 1a Validation — Quick Training Run

**Files:** (none modified — verification only)

- [ ] **Step 1: Run short training with PN expert + reward tuning**

Run: `cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && JSBSIM_DEBUG=0 python scripts/train_single_pursuit.py --seed 0 --steps 200000`

Expected:
- No crashes throughout training
- Stage 1 capture rate ≥ 50% by end of stage
- Eval metrics printed at each eval point
- `marl_runs/single_pursuit_*_s0/` directory created with model files and eval_metrics.csv

Check eval_metrics.csv for stage progression:
```bash
cat marl_runs/single_pursuit_*_s0/eval_metrics.csv
```

- [ ] **Step 2: If Stage 1 capture rate < 50%**, reduce nav_constant to 2.5 or increase P-gain. Open a plan amendment if needed.

- [ ] **Step 3: Commit if no issues**

```bash
git add -A
git commit -m "verify: Phase 1a smoke test — PN expert + reward tuning stable"
```

---

### Task 6: Hyperparameter Sweep Runner

**Files:**
- Create: `scripts/run_hyperparam_sweep.py`
- Create (auto): `results/sweep_YYYYMMDD_HHMM/report.csv`, `summary.txt`

- [ ] **Step 1: Create the sweep runner script**

Write `scripts/run_hyperparam_sweep.py`:

```python
"""Orthogonal hyperparameter sweep for single-pursuit PPO training.

Runs a fractional factorial design (8 configs) × 1 seed (500k steps),
selects top-2 configs, then runs them with 5 seeds each.

Usage:
    conda activate jsbsim_rl
    JSBSIM_DEBUG=0 python scripts/run_hyperparam_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import warnings
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the training entry point — modified to accept hyperparams
from scripts.train_single_pursuit import train_with_config


# ═══════════════════════════════════════════════════════════════════════════════
#  Sweep design — 4 params, 2 levels each, fractional factorial = 8 configs
# ═══════════════════════════════════════════════════════════════════════════════

SWEEP_PARAMS = {
    "lr": [1e-4, 3e-4],
    "ent_coef": [0.005, 0.01],
    "net_arch_pi": [[128, 128], [256, 128]],
    "n_steps": [2048, 4096],
}

FIXED_CONFIG = {
    "total_timesteps": 500_000,
    "batch_size": 256,
    "gamma": 0.99,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def generate_configs() -> List[Dict]:
    """Generate 8 fractional factorial configs (even rows of full factorial)."""
    all_levels = [
        list(enumerate(SWEEP_PARAMS["lr"])),
        list(enumerate(SWEEP_PARAMS["ent_coef"])),
        list(enumerate(SWEEP_PARAMS["net_arch_pi"])),
        list(enumerate(SWEEP_PARAMS["n_steps"])),
    ]
    full = list(product(*all_levels))
    # Take rows where index parity matches — gives a resolvable fraction
    configs = []
    for combo in full:
        indices = [c[0] for c in combo]
        # Use the standard fractional factorial: sum of indices even
        if sum(indices) % 2 == 0:
            configs.append({
                "lr": combo[0][1],
                "ent_coef": combo[1][1],
                "net_arch_pi": combo[2][1],
                "n_steps": combo[3][1],
            })
    return configs


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score confidence interval for a proportion."""
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


def evaluate_config(model_path: str, stage: float = 3.0, n_episodes: int = 50) -> Dict:
    """Evaluate a trained model on a given curriculum stage."""
    from src.environment.single_pursuit_env import SinglePursuitEnv
    from scripts.train_single_pursuit import ResidualExpertWrapper
    from stable_baselines3 import PPO

    env = SinglePursuitEnv(curriculum_stage=stage, record_tacview=False)
    wrapper = ResidualExpertWrapper(env)
    model = PPO.load(model_path)

    successes = 0
    min_dists = []
    intercept_times = []

    for _ in range(n_episodes):
        obs, _ = wrapper.reset()
        done = False
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])
        if info.get("reason") == "success":
            successes += 1
            intercept_times.append(env._step_counter / 60.0)
        else:
            intercept_times.append(120.0)
        min_dists.append(ep_min_dist)

    p, lo, hi = wilson_ci(successes, n_episodes)
    return {
        "capture_rate": p,
        "ci_low": lo,
        "ci_high": hi,
        "avg_min_dist": float(np.mean(min_dists)),
        "std_min_dist": float(np.std(min_dists)),
        "avg_intercept_time": float(np.mean(intercept_times)),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without training")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(f"results/sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = generate_configs()
    print(f"{'='*60}")
    print(f"Hyperparameter Sweep — {len(configs)} configs")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"  Config {i+1}: {cfg}")
        return

    # ══ Phase 1: Run all 8 configs with seed=0 ═════════════════════════════
    phase1_results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'─'*50}")
        print(f"Phase 1 — Config {i+1}/{len(configs)}: {cfg}")
        print(f"{'─'*50}")

        run_name = f"sweep_cfg{i+1}_s0"
        log_dir = output_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        model_path = train_with_config(
            seed=0,
            log_dir=str(log_dir),
            learning_rate=cfg["lr"],
            ent_coef=cfg["ent_coef"],
            net_arch_pi=cfg["net_arch_pi"],
            n_steps=cfg["n_steps"],
            **FIXED_CONFIG,
        )

        # Evaluate on Stage 3
        result = evaluate_config(model_path, stage=3.0, n_episodes=50)
        result["config"] = cfg
        result["config_id"] = f"cfg{i+1}"
        phase1_results.append(result)
        print(f"  Result: capture_rate={result['capture_rate']:.2%} "
              f"[{result['ci_low']:.2%}, {result['ci_high']:.2%}] "
              f"min_dist={result['avg_min_dist']:.0f}±{result['std_min_dist']:.0f}m")

    # ══ Phase 2: Rank, pick top-2 ═══════════════════════════════════════════════
    phase1_results.sort(key=lambda r: r["capture_rate"], reverse=True)
    top2 = phase1_results[:2]
    print(f"\n{'='*60}")
    print("Top-2 Configs:")
    for r in top2:
        print(f"  {r['config_id']}: capture_rate={r['capture_rate']:.2%}  {r['config']}")

    # ══ Phase 3: Run top-2 with 5 seeds each ═══════════════════════════════════
    all_results = []
    for rank, result in enumerate(top2):
        cfg = result["config"]
        cfg_label = f"top{rank+1}"
        print(f"\n{'─'*50}")
        print(f"Phase 2 — {cfg_label}: {cfg} × 5 seeds")
        print(f"{'─'*50}")

        seed_results = []
        for seed in range(5):
            run_name = f"sweep_{cfg_label}_s{seed}"
            log_dir = output_dir / run_name
            log_dir.mkdir(parents=True, exist_ok=True)

            model_path = train_with_config(
                seed=seed,
                log_dir=str(log_dir),
                learning_rate=cfg["lr"],
                ent_coef=cfg["ent_coef"],
                net_arch_pi=cfg["net_arch_pi"],
                n_steps=cfg["n_steps"],
                **FIXED_CONFIG,
            )

            eval_result = evaluate_config(model_path, stage=3.0, n_episodes=50)
            eval_result["seed"] = seed
            eval_result["config_id"] = cfg_label
            seed_results.append(eval_result)
            print(f"    seed={seed}: capture_rate={eval_result['capture_rate']:.2%} "
                  f"min_dist={eval_result['avg_min_dist']:.0f}±{eval_result['std_min_dist']:.0f}m")

        # Aggregate over seeds
        cr_values = [r["capture_rate"] for r in seed_results]
        all_results.append({
            "config_id": cfg_label,
            "config": cfg,
            "mean_capture_rate": float(np.mean(cr_values)),
            "std_capture_rate": float(np.std(cr_values)),
            "mean_min_dist": float(np.mean([r["avg_min_dist"] for r in seed_results])),
            "mean_intercept_time": float(np.mean([r["avg_intercept_time"] for r in seed_results])),
            "seed_details": seed_results,
        })

    # ══ Write report CSV ═══════════════════════════════════════════════════════
    report_path = output_dir / "report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config_id", "seed", "capture_rate", "ci_low", "ci_high",
                          "avg_min_dist", "std_min_dist", "avg_intercept_time"])
        for top_result in all_results:
            for sd in top_result["seed_details"]:
                writer.writerow([
                    sd["config_id"], sd["seed"],
                    f"{sd['capture_rate']:.4f}",
                    f"{sd['ci_low']:.4f}", f"{sd['ci_high']:.4f}",
                    f"{sd['avg_min_dist']:.1f}", f"{sd['std_min_dist']:.1f}",
                    f"{sd['avg_intercept_time']:.1f}",
                ])

    # ══ Write summary ═══════════════════════════════════════════════════════════
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Hyperparameter Sweep Summary — {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Phase 1 Results (8 configs × seed=0):\n")
        for r in phase1_results:
            f.write(f"  {r['config_id']}: capture_rate={r['capture_rate']:.2%} "
                    f"[{r['ci_low']:.2%}, {r['ci_high']:.2%}]  "
                    f"min_dist={r['avg_min_dist']:.0f}m  {r['config']}\n")

        f.write(f"\nPhase 2 Results (top-2 × 5 seeds):\n")
        for top in all_results:
            p, lo, hi = wilson_ci(
                int(round(top["mean_capture_rate"] * 50)), 50)
            f.write(f"  {top['config_id']}: mean_capture_rate={top['mean_capture_rate']:.2%} "
                    f"±{top['std_capture_rate']:.2%}  "
                    f"mean_min_dist={top['mean_min_dist']:.0f}m  "
                    f"mean_intercept={top['mean_intercept_time']:.1f}s  "
                    f"config={top['config']}\n")
            f.write(f"    Per-seed: ")
            f.write(", ".join(
                f"s{sd['seed']}={sd['capture_rate']:.2%}"
                for sd in top["seed_details"]
            ))
            f.write("\n")

    print(f"\n{'='*60}")
    print(f"Sweep complete!")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    main()
```

- [ ] **Step 2: Add `train_with_config` export to train_single_pursuit.py**

At the end of `scripts/train_single_pursuit.py`, after the `train()` function, add a reusable entry point:

```python
def train_with_config(
    seed: int = 0,
    log_dir: str = "",
    learning_rate: float = 1e-4,
    ent_coef: float = 0.01,
    net_arch_pi: list | None = None,
    n_steps: int = 2048,
    total_timesteps: int = 500_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> str:
    """Train with explicit hyperparameters, return path to best model.

    Used by the sweep runner to inject hyperparameter values.
    """
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    if net_arch_pi is None:
        net_arch_pi = [128, 128]

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    base_env = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    env = ResidualExpertWrapper(base_env)
    env = Monitor(env, log_dir)

    eval_base = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=False)
    eval_env = ResidualExpertWrapper(eval_base)

    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=net_arch_pi, vf=net_arch_pi),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )

    # Use global config values
    global EVAL_FREQ, EVAL_EPISODES, TARGET_CAPTURE_RATE_STAGE_1_2, TARGET_CAPTURE_RATE_STAGE_2_3
    global CURRICULUM_STAGES

    # Temporarily override
    old_total = TOTAL_TIMESTEPS
    globals()["TOTAL_TIMESTEPS"] = total_timesteps

    cb = CurriculumCallback(eval_env, log_dir)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    best_path = os.path.join(log_dir, "best_model")
    model.save(best_path)
    model.save(os.path.join(log_dir, "model"))

    # Save eval CSV
    import csv
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                                "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(cb._eval_metrics)

    globals()["TOTAL_TIMESTEPS"] = old_total
    return best_path + ".zip"
```

**Note**: We use `globals()` to temporarily override TOTAL_TIMESTEPS since the current code uses module-level globals. A cleaner approach would be to refactor, but this is the minimal-change path.

- [ ] **Step 3: Verify the sweep runner parses correctly**

Run: `cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && python scripts/run_hyperparam_sweep.py --dry-run`

Expected: Prints 8 configs (or fewer depending on fractional factorial filtering) without training.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_hyperparam_sweep.py scripts/train_single_pursuit.py
git commit -m "feat: hyperparameter sweep runner with train_with_config export"
```

---

### Task 7: Multi-Seed Evaluation Enhancement

**Files:**
- Modify: `scripts/evaluate_and_visualize.py` (add `--multi-seed` flag, Wilson CI, CSV export, summary table)

- [ ] **Step 1: Add --multi-seed, --stage, and --csv flags to evaluate script**

At the top of `main()`, extend the argument parser. Find `parser.add_argument("--output-dir"` block (~line 344) and add new arguments after it:

```python
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Aggregate across multiple model directories (seeds) for statistical summary",
    )
    parser.add_argument(
        "--stage", type=float, default=1.0,
        help="Curriculum stage to evaluate on (default: 1.0)",
    )
    parser.add_argument(
        "--model-dirs", nargs="+", default=None,
        help="One or more model directories (used with --multi-seed)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export results to CSV file",
    )
```

- [ ] **Step 2: Add multi-seed evaluation function**

Add before `main()`:

```python
def run_multi_seed_evaluation(model_dirs: List[str], stage: float, n_episodes: int = 30):
    """Evaluate models from multiple seeds and aggregate with Wilson CI.

    Args:
        model_dirs: List of paths to model directories (each contains best_model.zip or model.zip).
        stage: Curriculum stage for evaluation.
        n_episodes: Episodes per model directory.

    Returns:
        Dict with per-seed and aggregate results.
    """
    from src.environment.single_pursuit_env import SinglePursuitEnv
    from scripts.train_single_pursuit import ResidualExpertWrapper
    from stable_baselines3 import PPO

    per_seed = []
    for seed_idx, model_dir in enumerate(model_dirs):
        # Find model file
        model_path = None
        for name in ["best_model", "model", "final_model"]:
            candidate = os.path.join(model_dir, f"{name}.zip")
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            print(f"  WARNING: No model found in {model_dir}, skipping")
            continue

        model = PPO.load(model_path)
        env = SinglePursuitEnv(curriculum_stage=stage, record_tacview=False)
        wrapper = ResidualExpertWrapper(env)

        successes = 0
        min_dists = []
        intercept_times = []

        for _ in range(n_episodes):
            obs, _ = wrapper.reset()
            done = False
            ep_min_dist = 8000.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = wrapper.step(action)
                done = terminated or truncated
                if "min_dist" in info:
                    ep_min_dist = min(ep_min_dist, info["min_dist"])
            if info.get("reason") == "success":
                successes += 1
                intercept_times.append(env._step_counter / 60.0)
            else:
                intercept_times.append(120.0)
            min_dists.append(ep_min_dist)

        p, lo, hi = wilson_ci(successes, n_episodes)
        per_seed.append({
            "model_dir": model_dir,
            "capture_rate": p,
            "ci_low": lo,
            "ci_high": hi,
            "successes": successes,
            "n_episodes": n_episodes,
            "avg_min_dist": float(np.mean(min_dists)),
            "std_min_dist": float(np.std(min_dists)),
            "avg_intercept_time": float(np.mean(intercept_times)),
        })
        print(f"  {os.path.basename(model_dir)}: capture_rate={p:.2%} "
              f"[{lo:.2%}, {hi:.2%}]  min_dist={np.mean(min_dists):.0f}m")

    # Aggregate
    cr_list = [r["capture_rate"] for r in per_seed]
    agg = {
        "mean_capture_rate": float(np.mean(cr_list)),
        "std_capture_rate": float(np.std(cr_list)),
        "mean_min_dist": float(np.mean([r["avg_min_dist"] for r in per_seed])),
        "mean_intercept_time": float(np.mean([r["avg_intercept_time"] for r in per_seed])),
        "total_episodes": sum(r["n_episodes"] for r in per_seed),
        "total_successes": sum(r["successes"] for r in per_seed),
    }
    agg_p, agg_lo, agg_hi = wilson_ci(agg["total_successes"], agg["total_episodes"])
    agg["pooled_capture_rate"] = agg_p
    agg["pooled_ci_low"] = agg_lo
    agg["pooled_ci_high"] = agg_hi

    return {"per_seed": per_seed, "aggregate": agg}
```

- [ ] **Step 3: Add Wilson CI helper and summary printer**

Add the helper before `run_multi_seed_evaluation`:

```python
def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    from math import sqrt
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


def print_multi_seed_summary(results: dict) -> None:
    """Print a formatted summary table for multi-seed results."""
    agg = results["aggregate"]
    print(f"\n{'='*70}")
    print(f"Multi-Seed Evaluation Summary")
    print(f"{'='*70}")
    print(f"  Seeds evaluated:       {len(results['per_seed'])}")
    print(f"  Total episodes:        {agg['total_episodes']}")
    print(f"  Pooled capture rate:   {agg['pooled_capture_rate']:.2%} "
          f"[{agg['pooled_ci_low']:.2%}, {agg['pooled_ci_high']:.2%}]")
    print(f"  Mean capture rate:     {agg['mean_capture_rate']:.2%} "
          f"± {agg['std_capture_rate']:.2%}")
    print(f"  Mean min distance:     {agg['mean_min_dist']:.0f}m")
    print(f"  Mean intercept time:   {agg['mean_intercept_time']:.1f}s")
    print(f"\n  Per-seed details:")
    print(f"  {'Seed':<25s} {'Capture Rate':>12s} {'95% CI':>20s} {'Min Dist':>10s}")
    print(f"  {'─'*25} {'─'*12} {'─'*20} {'─'*10}")
    for r in results["per_seed"]:
        dir_name = os.path.basename(r["model_dir"])
        print(f"  {dir_name:<25s} {r['capture_rate']:>11.2%}  "
              f"[{r['ci_low']:.2%}, {r['ci_high']:.2%}]  "
              f"{r['avg_min_dist']:>7.0f}m")
```

- [ ] **Step 4: Wire --multi-seed into main()**

In `main()`, after argument parsing and before model loading, add the multi-seed branch:

```python
    # ── Multi-seed mode ───────────────────────────────────────────────────
    if args.multi_seed:
        if not args.model_dirs:
            # Auto-discover from marl_runs/
            import glob as _glob
            candidates = sorted(_glob.glob("marl_runs/single_pursuit_*"))
            if not candidates:
                print("No model directories found in marl_runs/")
                return
            # Use all directories (they are from different seeds of the same config)
            model_dirs = candidates
        else:
            model_dirs = args.model_dirs

        results = run_multi_seed_evaluation(model_dirs, stage=args.stage,
                                            n_episodes=args.episodes)
        print_multi_seed_summary(results)

        # CSV export
        csv_path = args.csv or os.path.join(output_dir, f"multi_seed_eval_stage{args.stage:.1f}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_dir", "capture_rate", "ci_low", "ci_high",
                             "avg_min_dist", "std_min_dist", "avg_intercept_time"])
            for r in results["per_seed"]:
                writer.writerow([
                    r["model_dir"],
                    f"{r['capture_rate']:.4f}",
                    f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}",
                    f"{r['avg_min_dist']:.1f}", f"{r['std_min_dist']:.1f}",
                    f"{r['avg_intercept_time']:.1f}",
                ])
        print(f"\nCSV exported: {csv_path}")
        return
```

- [ ] **Step 5: Add --stage support to single-model evaluation**

In the existing single-pursuit evaluation path, use `args.stage` instead of hardcoded stage 1:

Find `run_pursuit_evaluate` call in `main()` — there's the `if cfg["action_mode"] == "pursuit":` branch. Update it to pass stage:

```python
    if cfg["action_mode"] == "pursuit":
        episodes, env = run_pursuit_evaluate(model, n_episodes=args.episodes,
                                              stage=args.stage)
```

And update `run_pursuit_evaluate` signature:

```python
def run_pursuit_evaluate(model, n_episodes: int = 5, stage: float = 1.0):
    """Evaluate a SinglePursuitEnv-trained model."""
    from src.environment.single_pursuit_env import SinglePursuitEnv
    env = SinglePursuitEnv(curriculum_stage=stage, record_tacview=True)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_and_visualize.py
git commit -m "feat: multi-seed evaluation with Wilson CI and CSV export"
```

---

### Task 8: End-to-End Integration Test

**Files:** (none modified — verification only)

- [ ] **Step 1: Run a short end-to-end sweep test**

Run a minimal sweep test (2 configs, 50k steps each):

```bash
cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && JSBSIM_DEBUG=0 python -c "
import sys, os
sys.path.insert(0, '.')
from scripts.train_single_pursuit import train_with_config

# Quick test of train_with_config
path = train_with_config(
    seed=0, log_dir='./marl_runs/test_sweep_cfg',
    learning_rate=1e-4, ent_coef=0.01,
    net_arch_pi=[64, 64], n_steps=512,
    total_timesteps=5000, batch_size=128,
)
print(f'Model saved to: {path}')
print('train_with_config integration test PASSED')
"
```

- [ ] **Step 2: Verify multi-seed evaluation works**

```bash
cd /mnt/c/Users/Sean/Documents/GitHub/jsbsim-marl-formation && source jsbsim_rl/bin/activate && JSBSIM_DEBUG=0 python scripts/evaluate_and_visualize.py --model single_pursuit --stage 1.0 --episodes 5
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "verify: end-to-end integration tests pass for Phase 1 optimization"
```

---

## Summary

| Task | Files | Key Changes |
|------|-------|-------------|
| 1 | `train_single_pursuit.py` | PN guidance expert in `_compute_expert` via `_base_env` |
| 2 | `single_pursuit_env.py` | REWARD_PROGRESS=5.0, REWARD_ATA=5.0, terminal boost, time pressure |
| 3 | `single_pursuit_env.py` | Float curriculum stages 1.0-3.0, 5-stage spawning + target profiles |
| 4 | `train_single_pursuit.py` | Updated config, CurriculumCallback for 5 stages, eval CSV, PPO net_arch |
| 5 | (verification) | Phase 1a smoke test — Stage 1 capture ≥ 50% |
| 6 | `run_hyperparam_sweep.py` (new), `train_single_pursuit.py` | Sweep runner, `train_with_config` export |
| 7 | `evaluate_and_visualize.py` | `--multi-seed`, Wilson CI, CSV export, `--stage` flag |
| 8 | (verification) | End-to-end integration test |
