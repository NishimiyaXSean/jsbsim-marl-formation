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

ENV_NAME = "jsbsim_shoot_1v1"
# Legacy env names used by older checkpoints (P0-1): keep them registered so old
# checkpoints can be resumed/rendered after the env-name unification.
ENV_ALIASES = ["jsbsim_shoot_v101", "jsbsim_shoot_1v1_v1"]


def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


for _name in [ENV_NAME] + ENV_ALIASES:
    register_env(_name, lambda c: env_creator(c))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--legacy-obs", action="store_true",
                        help="use 36-dim legacy observation (no closure/LOS-rate); required when resuming pre-P0-2 checkpoints")
    args = parser.parse_args()

    register_env(ENV_NAME, lambda c: env_creator(c))
    ModelCatalog.register_custom_model("shoot_mask_model", ShootMaskModel)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

    # P0-4: persist launch-quality stats inside the run dir.  Set before ray.init
    # so env-runner workers inherit LAUNCH_STATS_FILE.
    output_dir = args.output or f"marl_runs/shoot_1v1_d{args.difficulty:.1f}_s{args.seed}"
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    launch_stats_path = os.path.join(output_dir, "launch_stats.log")
    os.environ["LAUNCH_STATS_FILE"] = launch_stats_path
    with open(launch_stats_path, "w") as _stats_f:
        _stats_f.write("# launch quality stats: good/bad/premium + closure/ATA/range\n")

    ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")

    # Always persist obs_include_closure in the checkpoint's env_config so
    # PPO.from_checkpoint rebuilds the identical obs dim (38 new / 36 legacy).
    env_config = {"difficulty_level": args.difficulty,
                  "obs_include_closure": not args.legacy_obs}

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
            entropy_coeff=0.03,  # annealed manually in the loop (old API stack)
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

    # P1: manual lr/entropy annealing (dict schedules unsupported in the old
    # API stack).  Linear decay to 20% of the initial values over the run —
    # v14 showed a late-training collapse with fixed lr/entropy.
    _policy = algo.get_policy("default_policy")
    _optim = _policy._optimizers[0] if getattr(_policy, "_optimizers", None) else None
    _ENTROPY0, _ENTROPY1 = 0.03, 0.005

    best_reward = -float("inf")

    print(f"Training 1v1 shoot — {args.iterations} iters, difficulty={args.difficulty:.1f}, "
          f"lr={args.lr}->{args.lr*0.2:.1e}, entropy=0.03->0.005")
    print(f"Output: {output_dir}")
    print(f"Launch stats: {launch_stats_path}")

    for i in range(args.iterations):
        frac = i / max(args.iterations - 1, 1)
        cur_lr = args.lr * (1.0 - 0.8 * frac)
        cur_entropy = _ENTROPY0 - (_ENTROPY0 - _ENTROPY1) * frac
        _policy.config["entropy_coeff"] = cur_entropy
        if _optim is not None:
            for _g in _optim.param_groups:
                _g["lr"] = cur_lr
        result = algo.train()
        rew = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
        length = result.get("env_runners", {}).get("episode_len_mean", 0)
        # Fire rate: fraction of actions where fire==1 (flat index 10 of 11)
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
