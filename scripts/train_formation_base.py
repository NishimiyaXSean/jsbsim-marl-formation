"""RLlib MAPPO training entry point — Task-Based architecture (BaseEnv + FormationTask).

Replaces train_formation_rllib.py for the refactored architecture.
Uses the hierarchical action space (MultiDiscrete[3 speed, 5 heading, 3 altitude])
and pliable controller interface (PID or Neural + SafetyInterceptor).

Usage:
    conda activate marl_env
    python scripts/train_formation_base.py --iterations 500 --seed 42
    python scripts/train_formation_base.py --controller neural --iterations 200 --seed 42

Step 5: start with --num-workers 1 for smoke validation.
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
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.base_env import BaseEnv
from src.environment.formation_task import FormationTask
from src.models.formation_rllib_model import RLlibAttentionActor

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment creator for RLlib registration
# ═══════════════════════════════════════════════════════════════════════════════

ENV_NAME = "jsbsim_formation_base_v1"

def make_env(env_config: dict | None = None):
    """Factory function for RLlib tune.register_env()."""
    config = env_config or {}
    task = FormationTask(config)
    return BaseEnv(task=task, env_config=config)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RLlib MAPPO — Task-Based Formation Training")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller", type=str, default="pid",
                        choices=["pid", "neural"])
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Rollout workers (1 for smoke test)")
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.03)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"rllib_base_{timestamp}_s{args.seed}"
    project_root = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(f"{project_root}/checkpoints", exist_ok=True)

    ray.init(ignore_reinit_error=True, num_cpus=args.num_workers + 2)

    # Register env + custom model with new env name
    register_env(ENV_NAME, make_env)
    ModelCatalog.register_custom_model("attention_formation", RLlibAttentionActor)

    # Probe env for spaces
    env_config = {
        "difficulty_level": args.difficulty,
        "controller_type": args.controller,
        "cooperative_mode": True,
    }
    temp_env = make_env(env_config)
    obs_space_p0 = temp_env.observation_space["p0"]
    act_space_p0 = temp_env.action_space["p0"]
    print(f"Controller: {args.controller}")
    print(f"Observation space (p0): {obs_space_p0}")
    print(f"Action space (p0):      {act_space_p0}")
    temp_env.close()

    # ── RLlib Config ─────────────────────────────────────────────────────
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=env_config)
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=args.entropy_coeff,
            vf_clip_param=1000.0,
            grad_clip=0.5,
            train_batch_size=4096,
            minibatch_size=512,
            num_epochs=10,
            model={"custom_model": "attention_formation", "vf_share_layers": False},
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
        )
        .resources(
            num_gpus=1,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_duration=args.eval_episodes,
            evaluation_config={"env_config": env_config},
        )
        .debugging(log_level="WARN")
    )

    print(f"\nStarting training: {run_name}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Workers:    {args.num_workers}")
    print(f"  Controller: {args.controller}")
    print(f"  LR:         {args.lr}")
    print(f"  Entropy:    {args.entropy_coeff}")
    print()

    # ── Build / Resume ───────────────────────────────────────────────────
    algo = config.build()

    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        algo.restore(args.resume_from)

    # ── Train ────────────────────────────────────────────────────────────
    for i in range(args.iterations):
        result = algo.train()

        train_reward = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        print(f"[iter {i:4d}] reward={train_reward:+.1f}  "
              f"steps={result.get('env_runners', {}).get('num_env_steps_sampled', 0)}  "
              f"fps={result.get('env_runners', {}).get('num_env_steps_sampled_per_second', 0):.0f}")

        # Checkpoint
        if (i + 1) % args.checkpoint_freq == 0 or i == args.iterations - 1:
            ckpt_dir = algo.save(f"{project_root}/checkpoints/checkpoint_{i:06d}")
            print(f"  → checkpoint: {ckpt_dir}")

    algo.save(f"{project_root}/checkpoints/checkpoint_final")
    print(f"\nTraining complete. Results: {project_root}")
    ray.shutdown()


if __name__ == "__main__":
    main()
