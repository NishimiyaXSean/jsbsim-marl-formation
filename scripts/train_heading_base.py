"""RLlib PPO — HeadingTrackingTask smoke test.

Quick single-worker training to verify the Task-Based architecture
with the simplest possible task (heading hold).

Usage:
    conda activate marl_env
    python scripts/train_heading_base.py --iterations 50 --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
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
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-heading", type=float, default=90.0)
    parser.add_argument("--controller", type=str, default="pid")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"rllib_heading_{timestamp}_s{args.seed}"
    project_root = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(f"{project_root}/checkpoints", exist_ok=True)

    ray.init(ignore_reinit_error=True, num_cpus=2)
    register_env(ENV_NAME, make_env)

    env_config = {
        "target_heading": args.target_heading,
        "target_altitude": 5000.0,
        "target_speed": 250.0,
        "controller_type": args.controller,
    }

    temp_env = make_env(env_config)
    print(f"Observation space: {temp_env.observation_space}")
    print(f"Action space:      {temp_env.action_space}")
    temp_env.close()

    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .training(
            lr=3e-4, gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, vf_clip_param=1000.0, grad_clip=0.5,
            train_batch_size=1024, minibatch_size=128, num_epochs=10,
            model={"use_lstm": True, "lstm_cell_size": 128, "max_seq_len": 50},
        )
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .resources(num_gpus=1)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .debugging(log_level="WARN", seed=args.seed)
    )

    algo = config.build()
    print(f"\nTraining: {run_name}  target_heading={args.target_heading}°")

    best_reward = -float("inf")
    best_ckpt = None

    for i in range(args.iterations):
        result = algo.train()
        reward = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        length = result.get("env_runners", {}).get("episode_len_mean", 0)
        print(f"[iter {i:3d}] reward={reward:+.4f}  len={length:.0f}")

        if not np.isnan(reward) and reward > best_reward:
            best_reward = reward
            best_ckpt = algo.save(f"{project_root}/checkpoints/best")

        if (i + 1) % 20 == 0:
            algo.save(f"{project_root}/checkpoints/checkpoint_{i:04d}")

    algo.save(f"{project_root}/checkpoints/checkpoint_final")
    print(f"\nBest reward: {best_reward:.4f}  → {best_ckpt}")
    ray.shutdown()


if __name__ == "__main__":
    main()
