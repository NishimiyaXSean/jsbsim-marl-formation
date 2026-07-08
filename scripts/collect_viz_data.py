"""Collect 3D trajectories + Self-Attention weights for paper visualizations.

Loads the best Experiment 2 checkpoint (OR-gate MAPPO), runs evaluation
episodes, and captures:
  1. 3D positions (P0, P1, Target) at every decision step — for spatial trajectory plot
  2. Multi-Head Self-Attention weights (3×3 per head) + Pool weights (1×3) — for heatmap
  3. Per-step rewards, actions, tactical geometry — for contextual analysis

Output: data/viz/exp2_best_trajectory.npz

Usage:
    conda activate marl_env
    python scripts/collect_viz_data.py --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.formation_rllib_env import FormationRLlibEnv
from src.models.formation_rllib_model import RLlibAttentionActor

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention Hook — captures weights during forward pass
# ═══════════════════════════════════════════════════════════════════════════════

_attn_storage: Dict[str, list] = {
    "attn_weights": [],    # list of [n_heads, 3, 3] per step
    "pool_weights": [],    # list of [1, 3] per step
}


def _make_attn_hook(actor_module):
    """Register a forward hook on actor.attention to capture attention weights."""

    def hook(module, input, output):
        # MultiheadAttention.forward() returns (attn_output, attn_weights)
        # attn_weights: [B, n_heads, 3, 3] or [B, 3, 3] depending on batch_first
        if isinstance(output, tuple) and len(output) == 2:
            attn_out, attn_w = output
            _attn_storage["attn_weights"].append(
                attn_w.detach().cpu().numpy().copy()
            )

    actor_module.attention.register_forward_hook(hook)


def _make_pool_hook(actor_module):
    """Capture pool weights by hooking into the pool computation.

    Since pool_weights are computed inline (not a module), we hook the
    mlp_head input to capture the last computed pool_weights from
    the actor's state. Simpler: monkey-patch the forward to stash pool_weights.
    """
    original_forward = actor_module.forward

    def patched_forward(obs, return_attention=False):
        result = original_forward(obs, return_attention=True)
        if isinstance(result, tuple) and len(result) == 2:
            (loc, scale), attn_info = result
            _attn_storage["pool_weights"].append(
                attn_info["pool_weights"].detach().cpu().numpy().copy()
            )
            return loc, scale
        return result

    actor_module.forward = patched_forward


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_episodes(
    ckpt_path: str,
    n_episodes: int = 5,
    difficulty: float = 0.0,
    seed: int = 42,
) -> List[dict]:
    """Run eval episodes and collect trajectory + attention data.

    Returns list of episode dicts, each with keys:
      - positions: dict of {agent_id: ndarray[steps, 3]} + target
      - actions: dict of {agent_id: ndarray[steps, 2]}
      - attn_weights: ndarray[steps, n_heads, 3, 3]
      - pool_weights: ndarray[steps, 1, 3]
      - rewards: dict of {agent_id: ndarray[steps]}
      - total_reward: float
      - success: bool
      - reason: str
    """
    # ── Setup ────────────────────────────────────────────────────────────
    ModelCatalog.register_custom_model("attention_formation", RLlibAttentionActor)
    register_env("formation_2v1_rllib", lambda c: FormationRLlibEnv(c))

    ckpt_abs = os.path.abspath(ckpt_path)
    print(f"Loading checkpoint: {ckpt_abs}")
    algo = PPO.from_checkpoint(ckpt_abs)

    policy = algo.get_policy("shared_policy")
    model = policy.model  # RLlibAttentionActor
    model.eval()

    # Register attention hooks
    _make_attn_hook(model.actor)
    _make_pool_hook(model.actor)

    device = next(model.parameters()).device
    print(f"Model device: {device}")

    # ── Run episodes ─────────────────────────────────────────────────────
    episodes = []

    for ep_idx in range(n_episodes):
        # Clear attention storage
        _attn_storage["attn_weights"].clear()
        _attn_storage["pool_weights"].clear()

        env = FormationRLlibEnv({
            "difficulty_level": difficulty,
            "lock_altitude": True,
            "record_tacview": False,
            "cooperative_mode": True,
        })
        env._difficulty = difficulty

        obs_dict, _ = env.reset(seed=seed + ep_idx)
        done = False

        # Per-step storage
        ep_data = {
            "positions": {"p0": [], "p1": [], "target": []},
            "actions": {"p0": [], "p1": []},
            "rewards": {"p0": [], "p1": []},
            "distances": {"p0": [], "p1": []},
            "pincer_angles": [],
        }

        step = 0
        while not done:
            actions = {}
            for aid in env._agent_ids:
                if aid in obs_dict:
                    raw_act = algo.compute_single_action(
                        obs_dict[aid],
                        policy_id="shared_policy",
                        explore=False,
                    )
                    actions[aid] = np.clip(np.asarray(raw_act, dtype=np.float32), -1.0, 1.0)

            obs_dict, rewards, terminateds, truncateds, _ = env.step(actions)

            # Record positions
            for i, aid in enumerate(env._agent_ids):
                ep_data["positions"][aid].append(
                    env.pursuers[i].aircraft.position_ned.copy()
                )
            ep_data["positions"]["target"].append(
                env.targets[0].aircraft.position_ned.copy()
            )

            # Record actions, rewards
            for aid in env._agent_ids:
                ep_data["actions"][aid].append(actions[aid].copy())
                ep_data["rewards"][aid].append(float(rewards.get(aid, 0.0)))

            # Record distances
            t_pos = env.targets[0].aircraft.position_ned
            for i, aid in enumerate(env._agent_ids):
                d = float(np.linalg.norm(
                    env.pursuers[i].aircraft.position_ned - t_pos))
                ep_data["distances"][aid].append(d)

            # Record pincer angle
            if env.N >= 2:
                p0_pos = env.pursuers[0].aircraft.position_ned
                p1_pos = env.pursuers[1].aircraft.position_ned
                los0 = (t_pos - p0_pos)[:2]
                los1 = (t_pos - p1_pos)[:2]
                n0, n1 = float(np.linalg.norm(los0)), float(np.linalg.norm(los1))
                if n0 > 1.0 and n1 > 1.0:
                    cos_a = np.clip(float(np.dot(los0, los1)) / (n0 * n1), -1.0, 1.0)
                    ep_data["pincer_angles"].append(
                        float(np.degrees(np.arccos(cos_a))))
                else:
                    ep_data["pincer_angles"].append(0.0)

            step += 1
            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        env.close()

        # Convert lists to arrays
        for aid in env._agent_ids:
            for key in ["positions", "actions", "rewards", "distances"]:
                ep_data[key][aid] = np.array(ep_data[key][aid])
        ep_data["positions"]["target"] = np.array(ep_data["positions"]["target"])
        ep_data["pincer_angles"] = np.array(ep_data["pincer_angles"])
        ep_data["n_steps"] = step
        ep_data["total_reward"] = sum(
            ep_data["rewards"][aid].sum() for aid in env._agent_ids
        )

        # Attention weights
        if _attn_storage["attn_weights"]:
            ep_data["attn_weights"] = np.stack(_attn_storage["attn_weights"], axis=0)
        else:
            ep_data["attn_weights"] = np.array([])
        if _attn_storage["pool_weights"]:
            ep_data["pool_weights"] = np.stack(
                [w.squeeze(1) for w in _attn_storage["pool_weights"]], axis=0)
        else:
            ep_data["pool_weights"] = np.array([])

        episodes.append(ep_data)

        # Determine success
        total_r = ep_data["total_reward"]
        min_d = min(ep_data["distances"][aid].min() for aid in env._agent_ids)
        pincer_max = ep_data["pincer_angles"].max() if len(ep_data["pincer_angles"]) > 0 else 0
        print(f"  Ep {ep_idx+1}/{n_episodes}: "
              f"steps={step:3d}  rew={total_r:8.1f}  "
              f"min_d={min_d:6.0f}m  pincer_max={pincer_max:5.1f}°")

    algo.stop()
    ray.shutdown()
    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════════════

def save_episodes(episodes: List[dict], output_path: str):
    """Save collected episodes to compressed numpy archive."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Flatten episodes into a structured dict
    n_eps = len(episodes)
    save_dict = {"n_episodes": n_eps}

    for ep_idx, ep in enumerate(episodes):
        prefix = f"ep{ep_idx}_"
        save_dict[f"{prefix}n_steps"] = ep["n_steps"]
        save_dict[f"{prefix}total_reward"] = ep["total_reward"]

        for aid in ["p0", "p1"]:
            save_dict[f"{prefix}{aid}_positions"] = ep["positions"][aid]
            save_dict[f"{prefix}{aid}_actions"] = ep["actions"][aid]
            save_dict[f"{prefix}{aid}_rewards"] = ep["rewards"][aid]
            save_dict[f"{prefix}{aid}_distances"] = ep["distances"][aid]

        save_dict[f"{prefix}target_positions"] = ep["positions"]["target"]
        save_dict[f"{prefix}pincer_angles"] = ep["pincer_angles"]

        if ep["attn_weights"].size > 0:
            save_dict[f"{prefix}attn_weights"] = ep["attn_weights"]    # [T, n_heads, 3, 3]
        if ep["pool_weights"].size > 0:
            save_dict[f"{prefix}pool_weights"] = ep["pool_weights"]    # [T, 3]

    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved {n_eps} episodes to {output_path}")
    _print_save_summary(save_dict)


def _print_save_summary(save_dict):
    """Print a summary of what was saved."""
    n_eps = save_dict["n_episodes"]
    for ep_idx in range(n_eps):
        prefix = f"ep{ep_idx}_"
        steps = save_dict[f"{prefix}n_steps"]
        rew = save_dict[f"{prefix}total_reward"]
        min_d0 = save_dict[f"{prefix}p0_distances"].min()
        min_d1 = save_dict[f"{prefix}p1_distances"].min()
        pincer_max = save_dict[f"{prefix}pincer_angles"].max()
        has_attn = f"{prefix}attn_weights" in save_dict

        print(f"  Ep {ep_idx}: {steps} steps, rew={rew:.0f}, "
              f"d0_min={min_d0:.0f}m, d1_min={min_d1:.0f}m, "
              f"pincer_max={pincer_max:.0f}°, attn={'YES' if has_attn else 'NO'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect trajectory + attention data for paper visualizations")
    parser.add_argument("--ckpt", type=str,
                        default="marl_runs/rllib_formation_0708_1445_s42/checkpoints/best_iter_0040_rew_7888",
                        help="Path to RLlib checkpoint directory")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of eval episodes to run")
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Difficulty level [0, 1]")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str,
                        default="data/viz/exp2_best_trajectory.npz",
                        help="Output path for .npz file")
    args = parser.parse_args()

    episodes = collect_episodes(
        ckpt_path=args.ckpt,
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    save_episodes(episodes, args.output)
