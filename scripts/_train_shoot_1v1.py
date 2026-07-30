"""Minimal RLlib PPO training for 1v1 shoot task — end-to-end verification.

Usage:
  # Stage 1: difficulty=0.0 (target flies straight, learn to fire)
  python scripts/_train_shoot_1v1.py --iterations 100 --difficulty 0.0

  # Stage 2: difficulty=0.3 (target evades, forced to close range)
  python scripts/_train_shoot_1v1.py --iterations 300 --difficulty 0.3 \
      --checkpoint marl_runs/shoot_stage1_s42/checkpoints/best
"""

import os, sys, warnings, logging, argparse, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from ray.rllib.models import ModelCatalog
from src.models.shoot_mask_model import ShootMaskModel

ENV_NAME = "jsbsim_shoot_v101"


def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()

    register_env(ENV_NAME, lambda c: env_creator(c))
    ModelCatalog.register_custom_model("shoot_mask_model", ShootMaskModel)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")
    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")

    env_config = {"difficulty_level": args.difficulty}

    # Use custom model with real action mask support
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.03,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            train_batch_size=1024,
            minibatch_size=128,
            num_epochs=10,
            model={"custom_model": "shoot_mask_model"},
        )
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            sample_timeout_s=120,
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN", seed=args.seed)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    if args.checkpoint:
        algo.restore(os.path.abspath(args.checkpoint))
        print(f"Resumed from {args.checkpoint}")

    best_reward = -float("inf")
    output_dir = args.output or f"marl_runs/shoot_1v1_d{args.difficulty:.1f}_s{args.seed}"
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    print(f"Training 1v1 shoot — {args.iterations} iters, difficulty={args.difficulty:.1f}, "
          f"lr={args.lr}, entropy=0.03")
    print(f"Output: {output_dir}")

    for i in range(args.iterations):
        result = algo.train()
        rew = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        length = result.get("env_runners", {}).get("episode_len_mean", 0)
        # Fire rate: fraction of actions where fire==1 (flat index 12 of 13)
        # Note: this is an approximation — RLlib doesn't expose action histograms easily

        # Log every iteration for fine-grained per-episode analysis
        print(f"  iter {i:3d}: rew={rew:+.0f}  len={length:.0f}")

        if not np.isnan(rew) and rew > best_reward:
            best_reward = rew
            algo.save(f"{output_dir}/checkpoints/best")

        # Early success detection
        if rew > 2200:
            print(f"  *** HIT likely detected at iter {i}: rew={rew:+.0f} ***")

    algo.save(f"{output_dir}/checkpoints/final")
    print(f"\nDone. Best reward: {best_reward:+.0f}")
    print(f"Checkpoints: {output_dir}/checkpoints/")

    ray.shutdown()


if __name__ == "__main__":
    main()
