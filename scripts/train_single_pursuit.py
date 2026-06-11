"""Train a single-agent PPO policy for F-16 pursuit using Stable-Baselines3.

The agent controls high-level flight targets (heading, altitude, speed)
through a stabilised FlightController.  Training follows a 3-stage
curriculum with increasing target difficulty.

Usage:
    conda activate jsbsim_rl
    JSBSIM_DEBUG=0 python scripts/train_single_pursuit.py
"""

from __future__ import annotations

import datetime
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.single_pursuit_env import SinglePursuitEnv

# ═══════════════════════════════════════════════════════════════════════════════
#  Training config
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 500_000
CURRICULUM_STAGES = [1, 2, 3]
STAGE_TIMESTEPS = TOTAL_TIMESTEPS // len(CURRICULUM_STAGES)

EVAL_EPISODES = 20
EVAL_FREQ = 50_000
TARGET_CAPTURE_RATE = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumCallback(BaseCallback):
    """Handles stage advancement with automatic evaluation."""

    def __init__(self, eval_env: SinglePursuitEnv, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._log_dir = log_dir
        self._best_capture_rate = -1.0
        self._current_stage = 1

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Check if it's time for evaluation
        if self.num_timesteps % EVAL_FREQ > 0 and self.num_timesteps < TOTAL_TIMESTEPS:
            return

        # Evaluate
        successes, min_dists = 0, []
        for _ in range(EVAL_EPISODES):
            obs, _ = self._eval_env.reset()
            done = False
            ep_min_dist = 8000.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if "min_dist" in info:
                    ep_min_dist = min(ep_min_dist, info["min_dist"])
            if info.get("reason") == "success":
                successes += 1
            min_dists.append(ep_min_dist)

        capture_rate = successes / EVAL_EPISODES
        avg_min_dist = np.mean(min_dists)

        self.logger.record("eval/capture_rate", capture_rate)
        self.logger.record("eval/avg_min_dist", avg_min_dist)

        print(f"\n  [Eval @ {self.num_timesteps:,} steps] "
              f"stage={self._current_stage} "
              f"capture_rate={capture_rate:.0%} "
              f"avg_min_dist={avg_min_dist:.0f}m")

        # Save best model
        if capture_rate > self._best_capture_rate:
            self._best_capture_rate = capture_rate
            best_path = os.path.join(self._log_dir, "best_model")
            self.model.save(best_path)
            print(f"  → New best model saved: {best_path}")

        # Stage advancement
        new_stage = 1
        for s in CURRICULUM_STAGES:
            new_stage = s
            if capture_rate < TARGET_CAPTURE_RATE:
                break

        if new_stage != self._current_stage:
            print(f"  >> Advancing to curriculum stage {new_stage}")
            self._current_stage = new_stage
            self._eval_env.curriculum_stage = new_stage
            self._best_capture_rate = -1.0  # reset for new stage


class TacviewEvalCallback(BaseCallback):
    """Generate Tacview + trajectory plots for the best episode during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._last_eval_step = -EVAL_FREQ

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval_step < EVAL_FREQ:
            return
        self._last_eval_step = self.num_timesteps


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.abspath(f"./marl_runs/single_pursuit_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("data/tacview", exist_ok=True)

    # Environment
    env = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)
    env = Monitor(env, log_dir)

    eval_env = SinglePursuitEnv(curriculum_stage=1, record_tacview=False)

    # Print setup
    print(f"{'='*55}")
    print(f"Single-Agent Pursuit Training  |  JSBSim F-16")
    print(f"  Action space:   {env.action_space}")
    print(f"  Observation:    {env.observation_space.shape}")
    print(f"  Total steps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Monitor:        tensorboard --logdir {log_dir}")
    print(f"{'='*55}\n")

    # PPO model
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
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

    # Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CurriculumCallback(eval_env, log_dir),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")

    # Save final
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    # ── Final evaluation with Tacview ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final Evaluation with Tacview (20 episodes)")

    tacview_env = SinglePursuitEnv(curriculum_stage=eval_env.curriculum_stage,
                                   record_tacview=True)
    os.makedirs("results/single_pursuit", exist_ok=True)

    successes = 0
    min_dists = []
    best_ep_reward = -float("inf")
    best_ep_idx = 0

    for ep in range(EVAL_EPISODES):
        obs, _ = tacview_env.reset()
        done = False
        total_r = 0.0
        ep_min_dist = 8000.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = tacview_env.step(action)
            done = terminated or truncated
            total_r += rew
            if "min_dist" in info:
                ep_min_dist = min(ep_min_dist, info["min_dist"])

        reason = info.get("reason", "timeout")
        if reason == "success":
            successes += 1
        min_dists.append(ep_min_dist)

        if total_r > best_ep_reward:
            best_ep_reward = total_r
            best_ep_idx = ep

        print(f"  Ep {ep+1:2d}: {reason:15s}  reward={total_r:+7.1f}  "
              f"min_dist={info.get('min_dist', -1):.0f}m")

    # Export Tacview (frames from last episode run)
    tacview_path = os.path.join("results", "single_pursuit",
                                "single_pursuit_engagement.txt.acmi")
    tacview_env.export_tacview(tacview_path)

    capture_rate = successes / EVAL_EPISODES
    print(f"\n  Capture rate: {capture_rate:.0%}")
    print(f"  Avg min dist: {np.mean(min_dists):.0f} ± {np.std(min_dists):.0f}m")
    print(f"  Tacview → {os.path.abspath(tacview_path)}")

    # Generate trajectory plot using matplotlib
    try:
        _plot_best_episode(tacview_env, best_ep_idx)
    except Exception as e:
        print(f"  Plotting skipped: {e}")

    print("\nTraining complete.  Monitor:  tensorboard --logdir", log_dir)


def _plot_best_episode(env: SinglePursuitEnv, ep_idx: int) -> None:
    """Quick 3D trajectory + altitude plot for the best episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = env._tacview_frames
    if not frames:
        return

    a_lat = np.array([f["pursuer"]["lat_deg"] for f in frames])
    a_lon = np.array([f["pursuer"]["lon_deg"] for f in frames])
    a_alt = np.array([f["pursuer"]["alt_m"] for f in frames])
    t_lat = np.array([f["target"]["lat_deg"] for f in frames])
    t_lon = np.array([f["target"]["lon_deg"] for f in frames])
    t_alt = np.array([f["target"]["alt_m"] for f in frames])

    # Convert to approximate meters from reference
    ref_lat, ref_lon = 30.0, 120.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref_lat))

    a_x = (a_lat - ref_lat) * m_per_deg_lat
    a_y = (a_lon - ref_lon) * m_per_deg_lon
    t_x = (t_lat - ref_lat) * m_per_deg_lat
    t_y = (t_lon - ref_lon) * m_per_deg_lon

    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(16, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(a_x, a_y, a_alt, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax3d.plot(t_x, t_y, t_alt, "b-", lw=1.5, alpha=0.8, label="Target")
    ax3d.scatter(a_x[0], a_y[0], a_alt[0], color="darkred", s=80, marker="o")
    ax3d.scatter(a_x[-1], a_y[-1], a_alt[-1], color="red", s=80, marker="x")
    ax3d.scatter(t_x[0], t_y[0], t_alt[0], color="darkblue", s=80, marker="o")
    ax3d.scatter(t_x[-1], t_y[-1], t_alt[-1], color="blue", s=80, marker="x")
    ax3d.set_xlabel("North (m)")
    ax3d.set_ylabel("East (m)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

    ax2d.plot(a_x, a_y, "r-", lw=1.5, alpha=0.8, label="Pursuer")
    ax2d.plot(t_x, t_y, "b-", lw=1.5, alpha=0.8, label="Target")
    ax2d.scatter(a_x[0], a_y[0], color="darkred", s=80, marker="o")
    ax2d.scatter(a_x[-1], a_y[-1], color="red", s=80, marker="x")
    ax2d.scatter(t_x[0], t_y[0], color="darkblue", s=80, marker="o")
    ax2d.scatter(t_x[-1], t_y[-1], color="blue", s=80, marker="x")
    ax2d.set_xlabel("North (m)")
    ax2d.set_ylabel("East (m)")
    ax2d.set_title("Top-Down View")
    ax2d.legend()
    ax2d.axis("equal")

    out_path = "results/single_pursuit/single_pursuit_trajectory_best.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    # Altitude profile
    fig2, ax_alt = plt.subplots(figsize=(10, 4))
    times = np.arange(len(frames)) * 0.5
    ax_alt.plot(times, a_alt, "r-", lw=1.5, label="Pursuer")
    ax_alt.plot(times, t_alt, "b-", lw=1.5, label="Target")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude Profile")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)
    alt_path = "results/single_pursuit/single_pursuit_trajectory_best_altitude.png"
    fig2.savefig(alt_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Plots saved → {out_path}")


if __name__ == "__main__":
    import logging
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    train()
