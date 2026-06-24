"""V10.5 formal training batch: 3 seeds × 5M steps with anti-dolphin fixes.

Changes from V10:
  - Quadratic altitude penalty (gravity well) — kills dive-for-V_c arbitrage
  - Lowered V_c sigmoid saturation ceiling (V_c=30 m/s → multiplier ≈ 0.99)
  - AutoCurriculumCallback starting at difficulty=0.15
  - Per-difficulty model checkpointing

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/run_v10_5_training.py
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/run_v10_5_training.py --seeds 0 1 2
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/run_v10_5_training.py --steps 1000000
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment.single_pursuit_env import SinglePursuitEnv
from src.environment.ablation_wrappers import (
    ActionRepeatWrapper,
    BlendedActionWrapper,
    LeadPursuitRewardWrapper,
)
from scripts.train_single_pursuit import (
    AutoCurriculumCallback,
    EVAL_EPISODES,
    EVAL_FREQ,
    ResidualExpertWrapper,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 5_000_000
SEEDS = [0, 1, 2]
BATCH_TIMESTAMP = datetime.datetime.now().strftime("%m%d_%H%M")

# PPO hyperparameters (V10 baseline)
PPO_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=10,
    gamma=0.998,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.015,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=10,
    device="cpu",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
        ortho_init=True,
        log_std_init=0.0,
    ),
)


def build_env():
    """Build the V10.5 env chain with anti-dolphin reward wrapper.

    SinglePursuitEnv(diff=0.15) → BlendedAction(a=0.02) → LeadPursuitReward
        → ResidualExpert → ActionRepeat(×5, 2Hz) → Monitor
    """
    base = SinglePursuitEnv(difficulty_level=0.15, record_tacview=False)
    base = BlendedActionWrapper(base, alpha=0.02)
    base = LeadPursuitRewardWrapper(base)
    wrapped = ResidualExpertWrapper(base)
    wrapped = ActionRepeatWrapper(wrapped, repeat_frames=5)
    return wrapped


def run_seed(seed: int, total_steps: int, output_dir: str):
    """Run one training job for a single seed."""
    run_name = f"s{seed}"
    log_dir = os.path.join(output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  V10.5 Training  |  seed={seed}  |  steps={total_steps:,}")
    print(f"  Log: {log_dir}")
    print(f"{'=' * 60}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build envs
    train_env = build_env()
    train_env = Monitor(train_env, log_dir)
    eval_env = build_env()

    # Print setup
    print(f"  Action space:   {train_env.action_space}")
    print(f"  Observation:    {train_env.observation_space.shape}")
    print(f"  Difficulty:     starting at 0.15")
    print(f"  V_c sigmoid:    K=0.3, MID=15 (saturates at ~30 m/s)")
    print(f"  Alt penalty:    quadratic (dh/1000)^2 x 30.0\n")

    # PPO model
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir, **PPO_CONFIG)

    # Auto-curriculum callback
    auto_cb = AutoCurriculumCallback(eval_env, log_dir, train_env=train_env)

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=auto_cb, progress_bar=False)
    except KeyboardInterrupt:
        print(f"\n  [{run_name}] Interrupted — saving checkpoint...")

    elapsed = time.time() - start_time
    print(f"\n  [{run_name}] Training completed in {elapsed / 3600:.1f} hours")

    # Save final models
    model.save(os.path.join(log_dir, "model"))
    model.save(os.path.join(log_dir, "final_model"))

    # Save eval metrics CSV
    import csv
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    if auto_cb._eval_metrics:
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(auto_cb._eval_metrics[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(auto_cb._eval_metrics)

    print(f"  [{run_name}] Done → {csv_path}")
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="V10.5 formal training batch")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to run (default: 0 1 2)")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total timesteps per seed (default: 5_000_000)")
    args = parser.parse_args()

    output_dir = os.path.abspath(f"./marl_runs/v10_5_{BATCH_TIMESTAMP}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"  V10.5 FORMAL TRAINING BATCH")
    print(f"  Timestamp:  {BATCH_TIMESTAMP}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Steps/seed: {args.steps:,}")
    print(f"  Total:      {len(args.seeds)} x {args.steps:,} = {len(args.seeds) * args.steps:,}")
    print(f"  Output:     {output_dir}")
    print(f"  Anti-dolphin fixes: quadratic alt penalty + lowered V_c saturation")
    print(f"{'=' * 65}")

    results = []
    for seed in args.seeds:
        log_dir = run_seed(seed, args.steps, output_dir)
        results.append((seed, log_dir))

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  BATCH COMPLETE")
    print(f"{'=' * 65}")
    for seed, log_dir in results:
        print(f"  seed={seed}: {log_dir}")
    print(f"\n  TensorBoard:  tensorboard --logdir {output_dir}")
    print(f"  Models saved per difficulty level in each seed dir.\n")


if __name__ == "__main__":
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)

    main()
