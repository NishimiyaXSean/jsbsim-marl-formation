"""Unit tests for the seedable/configurable combat geometry (2026-08-06).

Covers:
  * reset(seed=s) reproduces the same initial geometry (same env & fresh env)
  * different seeds -> different geometry
  * max_heading_bias_deg buckets (30 / 60 / 120)
  * chase_dist_min/max range
  * alt_diff_m pursuer offset
  * separating closure via p_speed_range < t_speed_range
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask


def make_env(**cfg):
    base = {"difficulty_level": 0.0, "obs_include_closure": True}
    base.update(cfg)
    return BaseEnv(task=SingleCombatShootTask(base))


def first_obs(env, seed):
    obs, _ = env.reset(seed=seed)
    return obs["p0"]


def bias_est(obs):
    return abs(float(obs[21])) * 180.0


def dist_m(obs):
    return float(obs[19]) * 15000.0


def closure_mps(obs):
    return float(obs[22]) * 300.0


def test_same_env_same_seed_reproducible():
    env = make_env()
    o1 = first_obs(env, 7)
    o2 = first_obs(env, 7)
    assert np.array_equal(o1, o2) or np.allclose(o1, o2, atol=1e-6)


def test_fresh_env_same_seed_reproducible():
    o1 = first_obs(make_env(), 7)
    o2 = first_obs(make_env(), 7)
    assert np.array_equal(o1, o2) or np.allclose(o1, o2, atol=1e-6)


def test_different_seed_differs():
    o1 = first_obs(make_env(), 7)
    o2 = first_obs(make_env(), 8)
    assert not np.allclose(o1, o2, atol=1e-3)


def test_bias_30_bucket():
    env = make_env(max_heading_bias_deg=30.0)
    biases = [bias_est(first_obs(env, s)) for s in range(5)]
    assert all(b <= 48.0 for b in biases), biases
    assert max(biases) >= 12.0, biases


def test_bias_60_bucket():
    env = make_env(max_heading_bias_deg=60.0)
    biases = [bias_est(first_obs(env, s)) for s in range(5)]
    assert all(20.0 <= b <= 78.0 for b in biases), biases


def test_bias_120_reaches_wide():
    env = make_env(max_heading_bias_deg=120.0)
    biases = [bias_est(first_obs(env, s)) for s in range(8)]
    assert max(biases) >= 60.0, biases


def test_chase_dist_range():
    env = make_env(chase_dist_min=5000.0, chase_dist_max=8000.0)
    dists = [dist_m(first_obs(env, s)) for s in range(5)]
    assert all(4200.0 <= d <= 9200.0 for d in dists), dists


def test_alt_diff_offset():
    env = make_env(alt_diff_m=300.0)
    p_alts, t_alts = [], []
    for s in range(4):
        first_obs(env, s)
        env.reset(seed=s)
        p_alts.append(env.pursuers[0].aircraft.state["alt_m"])
        t_alts.append(env.targets[0].aircraft.state["alt_m"])
    assert all(abs(a - 3000.0) < 30.0 for a in t_alts), t_alts
    assert all(2600.0 <= a <= 3400.0 for a in p_alts), p_alts
    assert all(abs(a - 3000.0) > 200.0 for a in p_alts), p_alts


def test_closure_separating():
    env = make_env(p_speed_range=(150.0, 180.0),
                   t_speed_range=(210.0, 240.0))
    closures = [closure_mps(first_obs(env, s)) for s in range(5)]
    assert all(c > 0.0 for c in closures), closures


def test_closure_closing():
    env = make_env(max_heading_bias_deg=15.0,
                   p_speed_range=(280.0, 300.0),
                   t_speed_range=(180.0, 200.0))
    closures = [closure_mps(first_obs(env, s)) for s in range(5)]
    assert all(c < 0.0 for c in closures), closures

