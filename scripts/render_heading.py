"""Render a trained HeadingTrackingTask policy to Tacview ACMI.

Usage:
    conda activate marl_env
    python scripts/render_heading.py --ckpt marl_runs/rllib_heading_XXXX_s42/checkpoints/best
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import logging

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.base_env import BaseEnv
from src.environment.heading_task import HeadingTrackingTask

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

ENV_NAME = "heading_tracking_v1"


def make_env(env_config: dict | None = None):
    config = env_config or {}
    task = HeadingTrackingTask(config)
    return BaseEnv(task=task, env_config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to RLlib checkpoint")
    parser.add_argument("--output", type=str, default="heading_tracking_result.acmi")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--controller", type=str, default="pid")
    parser.add_argument("--target-heading", type=float, default=90.0)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=1)
    register_env(ENV_NAME, make_env)

    env_config = {
        "target_heading": args.target_heading,
        "target_altitude": 5000.0,
        "target_speed": 250.0,
        "controller_type": args.controller,
    }

    # Load policy
    print(f"Loading checkpoint: {args.ckpt}")
    algo = PPO.from_checkpoint(os.path.abspath(args.ckpt))

    # Create env, reset first, THEN enable ACMI (so registration has correct positions)
    task = HeadingTrackingTask(env_config)
    env = BaseEnv(task=task, env_config=env_config)
    obs, _ = env.reset(seed=42)
    env.enable_acmi_logging(args.output)
    env.log_acmi_step()

    total_reward = 0.0
    for step in range(args.max_steps):
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            action_dict[agent_id] = algo.compute_single_action(
                agent_obs, explore=False)

        obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
        env.log_acmi_step()

        for r in rewards.values():
            total_reward += r

        if terminateds.get("__all__") or truncateds.get("__all__"):
            reason = infos.get("p0", {}).get("termination_reason", "unknown")
            print(f"Episode ended at step {step+1}: {reason}")
            break
    else:
        print(f"Episode completed {args.max_steps} steps (timeout)")

    print(f"Total reward: {total_reward:.2f}")
    print(f"ACMI saved to: {args.output}")
    env.close()


if __name__ == "__main__":
    main()
