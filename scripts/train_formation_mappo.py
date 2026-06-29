"""Phase 5: RLlib MAPPO 2v1 formation training (CTDE).

Each pursuer has its own policy (shared weights), Box(2) action,
and 33-dim local observation.  Centralized critic sees global state.

Phase 5.0: diff=0.0 baseline — validate MAPPO matches SB3 97% success.
Phase 5.1: difficulty scaling + target maneuvers.

Usage:
    python scripts/train_formation_mappo.py
    python scripts/train_formation_mappo.py --iterations 200 --difficulty 0.0
"""

from __future__ import annotations

import argparse, os, sys, warnings, logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext

from src.models.formation_mappo_model import FormationMAPPOModel
from src.environment.formation_mappo_env import FormationMAPPOEnv


def env_creator(config: EnvContext):
    return FormationMAPPOEnv(config)


def train(iterations: int = 200, difficulty: float = 0.0, checkpoint_freq: int = 50):
    ray.init(ignore_reinit_error=True, num_cpus=4)

    # Register env + model
    tune.register_env("formation_2v1", env_creator)

    config = (
        PPOConfig()
        .environment("formation_2v1", env_config={
            "difficulty_level": difficulty,
            "lock_altitude": True,
            "record_tacview": False,
        })
        .framework("torch")
        .training(
            lr=3e-4,
            train_batch_size=8192,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            model={"custom_model": "formation_mappo", "vf_share_layers": False},
        )
        .multi_agent(
            policies={"shared_pursuer"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_pursuer",
            policies_to_train=["shared_pursuer"],
        )
        .resources(num_gpus=0, num_cpus_per_env_runner=1)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
    )

    print(f"[MAPPO 2v1] Starting training")
    print(f"  Iterations: {iterations}, Difficulty: {difficulty:.2f}")
    print(f"  Architecture: CTDE — shared Actor, centralized Critic")
    print(f"  Per-agent action: Box(2) via FlightController")

    from ray.rllib.algorithms.ppo import PPO
    algo = config.build()

    for i in range(iterations):
        result = algo.train()
        if i % 10 == 0:
            ep_rew = result.get("env_runners", {}).get("episode_reward_mean", 0)
            ep_len = result.get("env_runners", {}).get("episode_len_mean", 0)
            print(f"[MAPPO] iter={i:4d}  ep_rew={ep_rew:8.1f}  ep_len={ep_len:6.1f}")

        if checkpoint_freq and i > 0 and i % checkpoint_freq == 0:
            ckpt = algo.save()
            print(f"[MAPPO] checkpoint saved: {ckpt}")

    final = algo.save()
    print(f"[MAPPO] Training complete. Final model: {final}")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(iterations=args.iterations, difficulty=args.difficulty, checkpoint_freq=args.checkpoint_freq)
