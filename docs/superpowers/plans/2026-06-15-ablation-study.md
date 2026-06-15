# Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 3 ablation wrappers + a 12-run study orchestrator to determine which optimization most improves Stage 1→1.5 single-pursuit capture rate.

**Architecture:** Three `gym.Wrapper` subclasses compose with the existing `ResidualExpertWrapper` chain. A runner script coordinates all training jobs and produces a comparison CSV.

**Tech Stack:** Python 3, Gymnasium, Stable-Baselines3 PPO, NumPy. All code runs inside `jsbsim_rl` virtual environment.

---

## File Map

| File | Role | Change |
|------|------|--------|
| `src/environment/ablation_wrappers.py` | All 3 wrapper classes: LeadPursuitRewardWrapper, FrameStackWrapper, CubicActionWrapper | **Create** |
| `src/environment/__init__.py` | Export new wrapper classes | **Modify** |
| `scripts/run_ablation_study.py` | Orchestrator: config → run 12 jobs → generate summary CSV | **Create** |
| `tests/test_environment/test_ablation_wrappers.py` | Unit tests for all 3 wrappers | **Create** |

---

### Task 1: Create `tests/test_environment/test_ablation_wrappers.py` — FrameStackWrapper tests

**Rationale:** TDD — write tests first, verify they fail, then implement.

**Files:**
- Create: `tests/test_environment/test_ablation_wrappers.py`

- [ ] **Step 1: Write the test file with FrameStackWrapper tests**

```python
"""Unit tests for ablation wrappers."""
import numpy as np
import pytest
import gymnasium as gym


# ── FrameStackWrapper tests ─────────────────────────────────────────────

class DummyEnvForFrameStack(gym.Env):
    """Minimal env that returns obs equal to step count."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.observation_space = gym.spaces.Box(-1, 1, (3,))
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return np.full((3,), float(self._step), dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full((3,), float(self._step), dtype=np.float32)
        reward = float(self._step)
        terminated = self._step >= 5
        truncated = False
        info = {"step": self._step}
        return obs, reward, terminated, truncated, info


def test_frame_stack_output_shape():
    """FrameStackWrapper outputs (obs_dim * N,) stacked observations."""
    from src.environment.ablation_wrappers import FrameStackWrapper
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    assert env.observation_space.shape == (12,)
    assert env.observation_space.dtype == np.float32


def test_frame_stack_reset_fills_buffer():
    """On reset, all N frames equal the initial observation."""
    from src.environment.ablation_wrappers import FrameStackWrapper
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    obs, _ = env.reset()
    assert obs.shape == (12,)
    # All 4 frames should be [0, 0, 0]
    expected = np.zeros(12, dtype=np.float32)
    np.testing.assert_array_equal(obs, expected)


def test_frame_stack_step_returns_stacked():
    """After step, observation is concatenation of last N frames."""
    from src.environment.ablation_wrappers import FrameStackWrapper
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    obs0, _ = env.reset()
    # obs0 = [0,0,0, 0,0,0, 0,0,0, 0,0,0]
    obs1, _, _, _, _ = env.step(np.zeros(2))
    # obs1 should be [0,0,0, 0,0,0, 0,0,0, 1,1,1] — 3 frames of 0s + 1 frame of 1s
    expected = np.array([0,0,0, 0,0,0, 0,0,0, 1,1,1], dtype=np.float32)
    np.testing.assert_array_equal(obs1, expected)


def test_frame_stack_3_frames():
    """n_frames=3 produces correct output shape."""
    from src.environment.ablation_wrappers import FrameStackWrapper
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=3)
    assert env.observation_space.shape == (9,)
    obs, _ = env.reset()
    assert obs.shape == (9,)


def test_frame_stack_preserves_info():
    """Info dict from base env is passed through unchanged."""
    from src.environment.ablation_wrappers import FrameStackWrapper
    base = DummyEnvForFrameStack()
    env = FrameStackWrapper(base, n_frames=4)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["step"] == 1
```

- [ ] **Step 2: Run tests to verify they fail (FrameStackWrapper not defined)**

```bash
cd C:/Users/Sean/Documents/GitHub/jsbsim-marl-formation
source jsbsim_rl/bin/activate 2>/dev/null || conda activate jsbsim_rl
python -m pytest tests/test_environment/test_ablation_wrappers.py -v
```

Expected: 5 FAIL (ModuleNotFoundError or ImportError for FrameStackWrapper)

- [ ] **Step 3: Commit**

```bash
git add tests/test_environment/test_ablation_wrappers.py
git commit -m "test: add FrameStackWrapper unit tests (TDD red)"
```

---

### Task 2: Create `src/environment/ablation_wrappers.py` — FrameStackWrapper

**Files:**
- Create: `src/environment/ablation_wrappers.py`

- [ ] **Step 1: Create the file with FrameStackWrapper only (tests fail)**

```python
"""Ablation experiment wrappers for single-pursuit training.

Each wrapper modifies exactly one concern, composes with ResidualExpertWrapper,
and is independently testable.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame Stack Wrapper — temporal awareness via stacked observations
# ═══════════════════════════════════════════════════════════════════════════════

class FrameStackWrapper(gym.Wrapper):
    """Stack the last N observations into a flat vector.

    The policy sees [obs_{t-N+1}, ..., obs_t] giving it implicit velocity and
    inertia information through consecutive position changes.

    On reset, the buffer is filled with copies of the first observation.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        base_shape = env.observation_space.shape
        base_dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(base_shape[0] * n_frames,),
            dtype=base_dtype,
        )
        self._buffer: deque = deque(maxlen=n_frames)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        # Fill the buffer with copies of the initial observation
        self._buffer.clear()
        for _ in range(self.n_frames):
            self._buffer.append(obs.copy())
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        self._buffer.append(obs.copy())
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self._buffer)).astype(np.float32)
```

- [ ] **Step 2: Run FrameStackWrapper tests — all should pass**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v -k "frame_stack"
```

Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add src/environment/ablation_wrappers.py
git commit -m "feat: add FrameStackWrapper for temporal observation stacking"
```

---

### Task 3: Add CubicActionWrapper tests to test file

**Files:**
- Modify: `tests/test_environment/test_ablation_wrappers.py` (append tests)

- [ ] **Step 1: Append CubicActionWrapper tests**

```python
# ── CubicActionWrapper tests ────────────────────────────────────────────

class DummyEnvForCubicAction(gym.Env):
    """Captures the action received by the base env."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (4,))
        self.last_action = None

    def reset(self, seed=None, options=None):
        self.last_action = None
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_cubic_action_zero_passes_zero():
    """a=0 maps to 0 through cubic."""
    from src.environment.ablation_wrappers import CubicActionWrapper
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [0.0, 0.0, 0.0])


def test_cubic_action_half_maps_to_eighth():
    """a=0.5 maps to 0.125 through cubic."""
    from src.environment.ablation_wrappers import CubicActionWrapper
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [0.125, 0.125, 0.125])


def test_cubic_action_one_passes_one():
    """a=1.0 maps to 1.0 through cubic."""
    from src.environment.ablation_wrappers import CubicActionWrapper
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [1.0, 1.0, 1.0])


def test_cubic_action_negative_preserves_sign():
    """a=-0.5 maps to -0.125 through cubic — sign preserved."""
    from src.environment.ablation_wrappers import CubicActionWrapper
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    env.reset()
    env.step(np.array([-0.5, -1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(base.last_action, [-0.125, -1.0, 0.0])


def test_cubic_action_space_unchanged():
    """CubicActionWrapper preserves the action space definition."""
    from src.environment.ablation_wrappers import CubicActionWrapper
    base = DummyEnvForCubicAction()
    env = CubicActionWrapper(base)
    assert env.action_space.shape == (3,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
```

- [ ] **Step 2: Run tests — should fail (CubicActionWrapper not defined)**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v -k "cubic"
```

Expected: 5 FAIL

- [ ] **Step 3: Commit**

```bash
git add tests/test_environment/test_ablation_wrappers.py
git commit -m "test: add CubicActionWrapper unit tests (TDD red)"
```

---

### Task 4: Add CubicActionWrapper to `ablation_wrappers.py`

**Files:**
- Modify: `src/environment/ablation_wrappers.py` (append class)

- [ ] **Step 1: Append CubicActionWrapper class**

```python
# ═══════════════════════════════════════════════════════════════════════════════
#  Cubic Action Wrapper — nonlinear action mapping for origin precision
# ═══════════════════════════════════════════════════════════════════════════════

class CubicActionWrapper(gym.Wrapper):
    """Map raw policy output a ∈ [-1,1] through a cubic function: a³.

    This gives the policy fine-grained control near the origin while preserving
    full authority at the extremes:

        a=0.0 → 0.000  (dead zone for small jitter)
        a=0.1 → 0.001  (~1000 steps to max, extremely precise)
        a=0.5 → 0.125  (~8 steps to max)
        a=1.0 → 1.000  (full authority preserved)

    The mapping applies to all action dimensions uniformly. Exploration noise
    from PPO's log_std also passes through this mapping.
    """

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        mapped = np.sign(action) * np.power(np.abs(action), 3.0)
        return self.env.step(mapped)
```

- [ ] **Step 2: Run all CubicActionWrapper tests**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v -k "cubic"
```

Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add src/environment/ablation_wrappers.py
git commit -m "feat: add CubicActionWrapper for nonlinear action precision"
```

---

### Task 5: Add LeadPursuitRewardWrapper tests to test file

**Files:**
- Modify: `tests/test_environment/test_ablation_wrappers.py` (append tests)

- [ ] **Step 1: Append LeadPursuitRewardWrapper tests**

The reward wrapper needs to read pursuer/target state from `self.env`. We create a dummy env that exposes the necessary attributes.

```python
# ── LeadPursuitRewardWrapper tests ──────────────────────────────────────

class DummyEnvForLeadPursuit(gym.Env):
    """Minimal env that exposes pursuer/target NED state for reward calc."""

    REWARD_PROGRESS = 5.0
    REWARD_ATA = 5.0
    REWARD_GROUND_WARNING = 2.0
    REWARD_SUCCESS = 500.0
    REWARD_CRASH = -200.0
    REWARD_LOST_TARGET = -200.0
    PROXIMITY_TIERS = []
    MAX_DIST = 10000.0
    MAX_VEL = 400.0
    CTRL_FREQ = 60.0
    DECISION_STEPS = 30
    PHYSICS_DT = 1.0 / 60.0
    MAX_EPISODE_TIME = 120.0

    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (3,))
        self.observation_space = gym.spaces.Box(-1, 1, (19,))
        self.pursuer = None
        self.target_ac = None
        self._step_counter = 0
        self._prev_dist = 1000.0
        self._proximity_awarded = set()
        self._tacview_frames = []
        self._record_tacview_frames = False

    def reset(self, seed=None, options=None):
        # Create mock aircraft-like objects with .position_ned and .velocity_ned
        self._step_counter = 0
        self._prev_dist = 1000.0
        self._proximity_awarded.clear()

        # Mock pursuer
        self.pursuer = type('obj', (object,), {})()
        self.pursuer.position_ned = np.array([0.0, 0.0, 3000.0], dtype=np.float64)
        self.pursuer.velocity_ned = np.array([180.0, 0.0, 0.0], dtype=np.float64)
        self.pursuer.rpy_rad = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.pursuer.state = {
            "n_z_g": 1.0, "airspeed_mps": 180.0, "alt_m": 3000.0,
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
            "beta_deg": 0.0,
        }

        # Mock target
        self.target_ac = type('obj', (object,), {})()
        self.target_ac.position_ned = np.array([1000.0, 0.0, 3000.0], dtype=np.float64)
        self.target_ac.velocity_ned = np.array([180.0, 10.0, 0.0], dtype=np.float64)  # moving slightly right
        self.target_ac.rpy_rad = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target_ac.state = {
            "n_z_g": 1.0, "airspeed_mps": 180.0, "alt_m": 3000.0,
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
            "beta_deg": 0.0,
        }

        obs = np.zeros(19, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Move pursuer forward (simple translation)
        self.pursuer.position_ned = self.pursuer.position_ned + np.array([3.0, 0.0, 0.0])

        # Target moves as well (same as initial velocity)
        self.target_ac.position_ned = self.target_ac.position_ned + np.array([3.0, 0.167, 0.0])

        self._step_counter += 1
        prev = self._prev_dist
        current = float(np.linalg.norm(self.pursuer.position_ned - self.target_ac.position_ned))
        self._prev_dist = current

        # Compute simple base reward matching SinglePursuitEnv pattern
        reward = 0.0
        delta_dist = prev - current
        reward += self.REWARD_PROGRESS * delta_dist

        from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles
        a_forward = compute_forward_vector(self.pursuer.rpy_rad)
        t_forward = compute_forward_vector(self.target_ac.rpy_rad)
        _, los_dir, _ = compute_los(self.pursuer.position_ned, self.target_ac.position_ned)
        geo = compute_tactical_angles(a_forward, t_forward, los_dir)
        reward += self.REWARD_ATA * max(geo["cos_ata"], -0.2) * self.PHYSICS_DT

        terminated = self._step_counter >= 30
        obs = np.zeros(19, dtype=np.float32)
        return obs, reward, terminated, False, {"reason": "timeout" if terminated else ""}


def test_lead_pursuit_wrapper_shape_unchanged():
    """LeadPursuitRewardWrapper preserves observation space."""
    from src.environment.ablation_wrappers import LeadPursuitRewardWrapper
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    assert env.observation_space.shape == (19,)


def test_lead_pursuit_wrapper_adds_reward():
    """Reward is strictly larger when velocity points at target (positive LOS alignment)."""
    from src.environment.ablation_wrappers import LeadPursuitRewardWrapper
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(3))
    # Pursuer moving [180,0,0], target ahead → velocity aligns with LOS
    # So reward should be larger than baseline (positive lead terms)
    assert reward != 0.0  # Sanity check


def test_lead_pursuit_wrapper_includes_lead_prediction():
    """When target has lateral velocity, lead prediction reward is non-zero."""
    from src.environment.ablation_wrappers import LeadPursuitRewardWrapper
    base = DummyEnvForLeadPursuit()
    env = LeadPursuitRewardWrapper(base)
    env.reset()
    # Pursuer and target initially offset; target has lateral velocity
    _, reward1, _, _, _ = env.step(np.zeros(3))
    # Step again — lead point should differ from current target position
    _, reward2, _, _, _ = env.step(np.zeros(3))
    # Both steps should produce rewards; lead prediction term contributes
    assert np.isfinite(reward1)
    assert np.isfinite(reward2)
```

- [ ] **Step 2: Run tests — should fail (LeadPursuitRewardWrapper not defined)**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v -k "lead"
```

Expected: 3 FAIL

- [ ] **Step 3: Commit**

```bash
git add tests/test_environment/test_ablation_wrappers.py
git commit -m "test: add LeadPursuitRewardWrapper unit tests (TDD red)"
```

---

### Task 6: Add LeadPursuitRewardWrapper to `ablation_wrappers.py`

**Files:**
- Modify: `src/environment/ablation_wrappers.py` (append class after CubicActionWrapper)

- [ ] **Step 1: Append LeadPursuitRewardWrapper class**

```python
# ═══════════════════════════════════════════════════════════════════════════════
#  Lead Pursuit Reward Wrapper — guide toward predicted intercept point
# ═══════════════════════════════════════════════════════════════════════════════

class LeadPursuitRewardWrapper(gym.Wrapper):
    """Add lead pursuit reward terms on top of the base environment reward.

    Two new components:
    1. Velocity alignment — cos(pursuer_vel_dir, LOS_dir) × 2.0 × dt
       Rewards the aircraft actually MOVING toward the target (not just
       pointing at it — accounts for AoA/sideslip).

    2. Lead prediction — cos(pursuer_forward, LOS_to_future) × 3.0 × dt
       Rewards pointing at where the target WILL be (1 second ahead),
       not where it currently is. This is the core of lead pursuit.
    """

    VEL_ALIGN_WEIGHT = 2.0
    LEAD_PREDICT_WEIGHT = 3.0
    LEAD_TIME_SEC = 1.0          # look-ahead time for lead point

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only add lead pursuit bonus during normal flight (not on termination)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Access underlying SinglePursuitEnv state via unwrapped chain
        env = self.env

        pursuer_pos = env.pursuer.position_ned
        pursuer_vel = env.pursuer.velocity_ned
        pursuer_rpy = env.pursuer.rpy_rad
        target_pos = env.target_ac.position_ned
        target_vel = env.target_ac.velocity_ned

        from src.utils.geometry import compute_forward_vector, compute_los

        dt = env.PHYSICS_DT

        # 1. Velocity alignment: is the aircraft MOVING toward the target?
        _, los_dir, _ = compute_los(pursuer_pos, target_pos)
        vel_norm = float(np.linalg.norm(pursuer_vel))
        if vel_norm > 1.0:
            vel_dir = pursuer_vel / vel_norm
            cos_vel_los = float(np.clip(np.dot(vel_dir, los_dir), -0.5, 1.0))
            reward += cos_vel_los * self.VEL_ALIGN_WEIGHT * dt

        # 2. Lead prediction: point at future target position
        future_pos = target_pos + target_vel * self.LEAD_TIME_SEC
        _, future_los_dir, _ = compute_los(pursuer_pos, future_pos)
        pursuer_forward = compute_forward_vector(pursuer_rpy)
        cos_lead = float(np.clip(np.dot(pursuer_forward, future_los_dir), -0.5, 1.0))
        reward += cos_lead * self.LEAD_PREDICT_WEIGHT * dt

        return obs, reward, terminated, truncated, info
```

- [ ] **Step 2: Run all LeadPursuitRewardWrapper tests**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v -k "lead"
```

Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add src/environment/ablation_wrappers.py
git commit -m "feat: add LeadPursuitRewardWrapper for velocity-aligned lead pursuit"
```

---

### Task 7: Run all wrapper tests together

- [ ] **Step 1: Run full test suite for ablation wrappers**

```bash
python -m pytest tests/test_environment/test_ablation_wrappers.py -v
```

Expected: 13 PASS (5 frame_stack + 5 cubic + 3 lead_pursuit)

- [ ] **Step 2: Commit if any cleanups needed, otherwise skip**

---

### Task 8: Update `src/environment/__init__.py` to export new wrappers

**Files:**
- Modify: `src/environment/__init__.py`

- [ ] **Step 1: Add exports**

```python
from src.environment.ablation_wrappers import (
    FrameStackWrapper,
    CubicActionWrapper,
    LeadPursuitRewardWrapper,
)
```

- [ ] **Step 2: Verify imports work**

```bash
python -c "from src.environment.ablation_wrappers import FrameStackWrapper, CubicActionWrapper, LeadPursuitRewardWrapper; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/environment/__init__.py
git commit -m "feat: export ablation wrappers from environment package"
```

---

### Task 9: Create `scripts/run_ablation_study.py` — the orchestrator

**Files:**
- Create: `scripts/run_ablation_study.py`

- [ ] **Step 1: Create the full runner script**

```python
"""Ablation study orchestrator for single-pursuit training.

Runs 4 configurations × 3 seeds = 12 training jobs at 200K timesteps each,
then produces a summary CSV comparing Stage 1 → 1.5 transfer performance.

Usage:
    conda activate jsbsim_rl
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --seeds 0 1 2 3 4
    python scripts/run_ablation_study.py --steps 100000
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import (
    CubicActionWrapper,
    FrameStackWrapper,
    LeadPursuitRewardWrapper,
)
from scripts.train_single_pursuit import (
    CurriculumCallback,
    CURRICULUM_STAGES,
    EVAL_EPISODES,
    EVAL_FREQ,
    TARGET_CAPTURE_RATE_STAGE_1_2,
    ResidualExpertWrapper,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Ablation configuration
# ═══════════════════════════════════════════════════════════════════════════════

ABLATIONS = [
    {"name": "baseline",     "label": "BL", "wrapper_cls": None},
    {"name": "lead_pursuit", "label": "RW", "wrapper_cls": LeadPursuitRewardWrapper},
    {"name": "frame_stack",  "label": "FS", "wrapper_cls": FrameStackWrapper},
    {"name": "cubic_action", "label": "CA", "wrapper_cls": CubicActionWrapper},
]

STAGES_FOR_ABLATION = [1.0, 1.5]  # Only Stage 1.0 and 1.5

# PPO hyperparameters — identical across all variants
PPO_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
        ortho_init=True,
        log_std_init=0.0,
    ),
)


def build_env(ablation_config: dict, record_tacview: bool = False):
    """Build the full env chain: SinglePursuitEnv → ablation_wrapper? → ResidualExpertWrapper → Monitor?"""
    base = SinglePursuitEnv(curriculum_stage=1.0, record_tacview=record_tacview)
    if ablation_config["wrapper_cls"] is not None:
        base = ablation_config["wrapper_cls"](base)
    wrapped = ResidualExpertWrapper(base)
    return wrapped


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def run_one(ablation_config: dict, seed: int, total_steps: int, output_dir: str):
    """Run one training job. Returns path to eval_metrics.csv."""
    label = ablation_config["label"]
    name = ablation_config["name"]
    run_name = f"{label}_s{seed}"
    log_dir = os.path.join(output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{label}] {name}  |  seed={seed}  |  steps={total_steps:,}")
    print(f"  Log: {log_dir}")
    print(f"{'='*60}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build envs
    train_env = build_env(ablation_config, record_tacview=False)
    train_env = Monitor(train_env, log_dir)
    eval_env = build_env(ablation_config, record_tacview=False)

    # PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir, **PPO_CONFIG)

    # Curriculum callback
    curriculum_cb = CurriculumCallback(eval_env, log_dir)
    # Override CURRICULUM_STAGES for ablation (2-stage only)
    import scripts.train_single_pursuit as train_mod
    original_stages = train_mod.CURRICULUM_STAGES
    train_mod.CURRICULUM_STAGES = STAGES_FOR_ABLATION
    try:
        model.learn(total_timesteps=total_steps, callback=curriculum_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("\n  Interrupted — saving checkpoint...")
    finally:
        train_mod.CURRICULUM_STAGES = original_stages

    # Save model
    model.save(os.path.join(log_dir, "model"))
    model.save(os.path.join(log_dir, "best_model"))

    # Save eval metrics CSV
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "stage", "capture_rate",
                                               "avg_min_dist", "avg_intercept_time"])
        writer.writeheader()
        writer.writerows(curriculum_cb._eval_metrics)

    print(f"  ✓ Done → {csv_path}")
    return csv_path


def summarize(output_dir: str):
    """Read all eval_metrics.csv files and produce a comparison summary."""
    rows = []
    for ablation in ABLATIONS:
        label = ablation["label"]
        name = ablation["name"]
        for seed in range(10):  # scan for existing seed dirs
            run_dir = os.path.join(output_dir, f"{label}_s{seed}")
            csv_path = os.path.join(run_dir, "eval_metrics.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                metrics = list(reader)
            if not metrics:
                continue

            # Peak Stage 1 capture rate
            stage1_rates = [float(r["capture_rate"]) for r in metrics
                           if float(r["stage"]) == 1.0]
            peak_s1 = max(stage1_rates) if stage1_rates else 0.0

            # Stage 1.5 transfer: first eval after ≥40K steps in stage 1.5
            stage15_metrics = [r for r in metrics if float(r["stage"]) == 1.5]
            s15_rate = 0.0
            s15_avg_dist = 0.0
            s15_total_evals = len(stage15_metrics)
            if stage15_metrics:
                # Use best Stage 1.5 capture rate
                s15_rates = [float(r["capture_rate"]) for r in stage15_metrics]
                s15_rate = max(s15_rates)
                s15_dists = [float(r["avg_min_dist"]) for r in stage15_metrics]
                s15_avg_dist = np.mean(s15_dists)

            # Time-to-advance (first timestep where stage ≥ 1.5)
            advance_step = int(float(metrics[0]["timesteps"]))
            for r in metrics:
                if float(r["stage"]) >= 1.5:
                    advance_step = int(float(r["timesteps"]))
                    break

            rows.append({
                "label": label,
                "name": name,
                "seed": seed,
                "peak_stage1": peak_s1,
                "stage15_best": s15_rate,
                "stage15_avg_dist": s15_avg_dist,
                "stage15_evals": s15_total_evals,
                "advance_step": advance_step,
            })

    if not rows:
        print("  No results found.")
        return

    # Save detailed CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    fieldnames = ["label", "name", "seed", "peak_stage1", "stage15_best",
                  "stage15_avg_dist", "stage15_evals", "advance_step"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary CSV → {summary_path}")

    # Print ranked table by Stage 1.5 capture rate
    print(f"\n{'='*80}")
    print("ABLATION RESULTS — Ranked by Stage 1.5 capture rate")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Var':<6} {'Name':<16} {'Seeds':<8} {'Peak S1':<10} {'Best S1.5':<12} {'95% CI':<20} {'Adv@':<10}")
    print("-" * 80)

    # Aggregate per variant
    variants = {}
    for r in rows:
        v = r["label"]
        if v not in variants:
            variants[v] = {"name": r["name"], "rows": []}
        variants[v]["rows"].append(r)

    # Sort by mean Stage 1.5 best
    ranked = sorted(variants.items(),
                    key=lambda kv: np.mean([r["stage15_best"] for r in kv[1]["rows"]]),
                    reverse=True)

    for rank, (label, vdata) in enumerate(ranked, 1):
        vrows = vdata["rows"]
        name = vdata["name"]
        n_seeds = len(vrows)
        peak_s1 = np.mean([r["peak_stage1"] for r in vrows])
        best_s15 = np.mean([r["stage15_best"] for r in vrows])
        total_successes = sum(int(r["stage15_best"] * EVAL_EPISODES) for r in vrows)
        p, lo, hi = wilson_ci(total_successes, n_seeds * EVAL_EPISODES)
        avg_adv = np.mean([r["advance_step"] for r in vrows])

        print(f"{rank:<6} {label:<6} {name:<16} {n_seeds:<8} "
              f"{peak_s1:<10.1%} {best_s15:<12.1%} "
              f"[{lo:.1%}, {hi:.1%}]  "  # CI on its own line-ish
              f"{avg_adv:>8.0f}")

    print("-" * 80)
    print(f"  Winner: {ranked[0][0]} ({ranked[0][1]['name']})")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for single-pursuit training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                       help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--steps", type=int, default=200_000,
                       help="Total timesteps per run (default: 200000)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, just regenerate summary from existing CSVs")
    parser.add_argument("--ablation", type=str, nargs="+",
                       choices=["BL", "RW", "FS", "CA"],
                       help="Run only specific ablations (e.g. --ablation BL RW)")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.abspath(f"./marl_runs/ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ABLATION STUDY: Single-Pursuit Training Optimizations")
    print(f"  Variants:    {len(ABLATIONS)}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Total runs:  {len(ABLATIONS) * len(args.seeds)}")
    print(f"  Steps/run:   {args.steps:,}")
    print(f"  Stages:      {STAGES_FOR_ABLATION}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # Filter ablations if --ablation specified
    active = ABLATIONS
    if args.ablation:
        active = [a for a in ABLATIONS if a["label"] in args.ablation]
        print(f"  Running only: {[a['label'] for a in active]}")

    if not args.skip_training:
        for ablation_config in active:
            for seed in args.seeds:
                run_one(ablation_config, seed, args.steps, output_dir)

    # Generate summary
    print(f"\n{'='*60}")
    print("Generating summary...")
    summarize(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    main()
```

- [ ] **Step 2: Smoke test — import the runner (no training)**

```bash
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from scripts.run_ablation_study import ABLATIONS, build_env, PPO_CONFIG
print(f'Ablations: {[a[\"label\"] for a in ABLATIONS]}')
print(f'PPO lr={PPO_CONFIG[\"learning_rate\"]}')
env = build_env(ABLATIONS[0])
print(f'Baseline obs: {env.observation_space.shape}')
env = build_env(ABLATIONS[2])
print(f'FrameStack obs: {env.observation_space.shape}')
print('Smoke test PASSED')
"
```

Expected: prints ablation labels and observation shapes, then "Smoke test PASSED"

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ablation_study.py
git commit -m "feat: add ablation study orchestrator script"
```

---

### Task 10: Quick integration test — single short run (BL, 1 seed, 20K steps)

- [ ] **Step 1: Run a single tiny baseline job to verify end-to-end**

```bash
cd C:/Users/Sean/Documents/GitHub/jsbsim-marl-formation
source jsbsim_rl/Scripts/activate
python scripts/run_ablation_study.py --seeds 0 --steps 20000 --ablation BL
```

Expected: training runs, saves model + CSV to `marl_runs/ablation_<ts>/BL_s0/`

- [ ] **Step 2: Verify outputs exist**

```bash
ls marl_runs/ablation_*/BL_s0/eval_metrics.csv
ls marl_runs/ablation_*/BL_s0/summary.csv
```

Expected: both files exist; summary.csv has 1 row

- [ ] **Step 3: Commit (if integration test revealed issues that needed fixing)**

Otherwise skip — this is verification.

---

### Task 11: Run short smoke tests for all 4 variants (1 seed, 20K steps each)

Validate that all 4 wrappers train without crashing before committing to the full 200K × 12 run.

- [ ] **Step 1: Run all 4 variants at 20K steps**

```bash
python scripts/run_ablation_study.py --seeds 0 --steps 20000
```

Expected: 4 training runs complete. Check for any crash or NaN.

- [ ] **Step 2: Verify summary was generated**

```bash
python scripts/run_ablation_study.py --skip-training
```

Expected: summary printed. All 4 variants present with non-NaN metrics.

---

### Task 12: Launch full ablation study (12 runs × 200K steps)

- [ ] **Step 1: Start the full training**

```bash
python scripts/run_ablation_study.py --seeds 0 1 2 --steps 200000
```

This will run for several hours. Monitoring: check `marl_runs/ablation_<ts>/` for CSVs.

---

## Self-Review Notes

Before finalizing: checked spec coverage, no placeholders, all types consistent:
- `FrameStackWrapper(n_frames=4)` — matches spec
- `CubicActionWrapper` — `sign(a) * |a|³` matches spec
- `LeadPursuitRewardWrapper` — vel_align=2.0, lead_pred=3.0, LEAD_TIME=1.0s matches spec
- Runner uses `train_single_pursuit.CurriculumCallback` directly (reuse, not duplication)
- Summary metrics match spec: peak Stage 1, Stage 1.5 transfer, Wilson CI
- Output dir `marl_runs/ablation_{timestamp}/` matches spec
