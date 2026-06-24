"""Train PPO with Discrete(9) BFM action space on BFMPursuitEnv.

The agent selects from 9 PURSUIT_ACTIONS — tactical-level commands like
"turn right", "climb", "accelerate" — fed through the Phase 3.5 autopilot
pipeline (FlightEnvelope → BFMAutopilot+GainScheduler → JSBSim FCS).

This replaces the continuous [d_heading, d_alt, d_speed] interface.
The low-level flight stability is handled entirely by classical control;
RL focuses purely on tactical decision-making.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/train_bfm_pursuit.py
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/train_bfm_pursuit.py --seed 0 --steps 1000000
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment.bfm_pursuit_env import BFMPursuitEnv
from src.environment.ablation_wrappers import (
    ActionRepeatWrapper, BlendedActionWrapper, LeadPursuitRewardWrapper,
)
from src.environment.masked_policy import MaskableActorCriticPolicy
from scripts.train_single_pursuit import (
    AutoCurriculumCallback, EVAL_EPISODES, EVAL_FREQ,
)


TOTAL_TIMESTEPS = 5_000_000
DECISION_HZ = 2  # discrete BFM decisions at 2 Hz (0.5s per maneuver)


def build_env(difficulty: float = 0.15):
    """Build the BFM pursuit env chain with V11 reward wrappers."""
    base = BFMPursuitEnv(difficulty_level=difficulty, record_tacview=False)
    base = BlendedActionWrapper(base, alpha=0.02)
    base = LeadPursuitRewardWrapper(base)
    # No ActionRepeatWrapper — BFMPursuitEnv already runs at 2 Hz decision rate
    # (DECISION_DT=0.5s).  The autopilot stabilises through all 30 micro-steps.
    return base


def train(seed: int = 0, total_steps: int = TOTAL_TIMESTEPS):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"bfm_pursuit_{timestamp}_s{seed}"
    log_dir = os.path.abspath(f"./marl_runs/{run_name}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    train_env = build_env(difficulty=0.15)
    train_env = Monitor(train_env, log_dir)
    eval_env = build_env(difficulty=0.15)

    print(f"{'='*55}")
    print(f"BFM Discrete Pursuit Training  |  JSBSim F-16 + Phase 3.5 Autopilot")
    print(f"  Action space:   Discrete(9) — PURSUIT_ACTIONS")
    print(f"  Decision rate:  {DECISION_HZ} Hz (0.5s per maneuver)")
    print(f"  Autopilot:      BFMAutopilot + GainScheduler + TrimSchedule")
    print(f"  V_c:            minimum-wage floor (V11 anti-collapse)")
    print(f"  Alt penalty:    quadratic gravity well (V10.5 anti-dolphin)")
    print(f"  Action mask:    5 safety rules (hard deck, stall, overspeed, alpha)")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Log dir:        {log_dir}")
    print(f"{'='*55}\n")

    model = PPO(
        MaskableActorCriticPolicy, train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,             # shorter rollouts for discrete actions
        batch_size=256,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,            # V11 pacemaker
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )

    auto_cb = AutoCurriculumCallback(eval_env, log_dir, train_env=train_env)
    try:
        model.learn(total_timesteps=total_steps, callback=auto_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")

    model.save(os.path.join(log_dir, "final_model"))
    model.save(os.path.join(log_dir, "model"))

    import csv
    csv_path = os.path.join(log_dir, "eval_metrics.csv")
    if auto_cb._eval_metrics:
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(auto_cb._eval_metrics[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(auto_cb._eval_metrics)

    print(f"\nTraining complete.  tensorboard --logdir {log_dir}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(seed=args.seed, total_steps=args.steps)
