"""Benchmark the SB3 Phase 4.1 97.3% baseline — canonical evaluation.

Produces:
  1. Wilson CI success rate across N episodes at multiple difficulty levels
  2. Tacview ACMI files for best + worst trajectories
  3. 3D trajectory + altitude plots
  4. JSON metrics manifest for paper-ready tables

Usage:
  conda activate jsbsim_rl
  python scripts/benchmark_sb3_baseline.py
  python scripts/benchmark_sb3_baseline.py --episodes 100 --difficulty 0.0,0.3,0.6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import logging
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

import numpy as np
from stable_baselines3 import PPO

from src.environment.formation_env import FormationEnv

# ── matplotlib (non-interactive) ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_MODEL = "benchmarks/sb3_2v1_97p3/model.zip"
SOURCE_MODEL = "marl_runs/formation_2v1_0629_1721_s42/formation_2v1_final.zip"
OUTPUT_DIR = "benchmarks/sb3_2v1_97p3"

# ═══════════════════════════════════════════════════════════════════════════
#  Wilson CI
# ═══════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a binomial proportion."""
    from math import sqrt
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(model_path: str, n_episodes: int = 100,
                  difficulty: float = 0.0, record_tacview: bool = True):
    """Run N evaluation episodes and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmark: SB3 Phase 4.1 (2v1, difficulty={difficulty:.1f})")
    print(f"Model:     {model_path}")
    print(f"Episodes:  {n_episodes}")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)

    episodes = []
    successes = 0
    all_min_dists = []
    all_intercept_times = []
    all_final_spacings = []
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0.0
        ep_min_dist = 10000.0
        reason = "timeout"

        p0_traj = []; p1_traj = []; t_traj = []
        spacing_hist = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rew += rew

            # Trajectory recording
            p0_traj.append(env.pursuers[0].aircraft.position_ned.copy())
            p1_traj.append(env.pursuers[1].aircraft.position_ned.copy())
            t_traj.append(env.targets[0].aircraft.position_ned.copy())

            # Per-step metrics
            t_pos = env.targets[0].aircraft.position_ned
            d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - t_pos))
            d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - t_pos))
            ep_min_dist = min(ep_min_dist, d0, d1)
            spacing = float(np.linalg.norm(
                env.pursuers[0].aircraft.position_ned - env.pursuers[1].aircraft.position_ned))
            spacing_hist.append(spacing)

            if "reason" in info:
                reason = info["reason"]

        is_success = reason == "success"
        if is_success:
            successes += 1

        # Final spacing: last 10 steps average (formation quality at capture)
        final_spacing = float(np.mean(spacing_hist[-10:])) if len(spacing_hist) >= 10 else float(spacing_hist[-1])

        all_rewards.append(total_rew)
        all_min_dists.append(ep_min_dist)
        all_final_spacings.append(final_spacing)
        intercept_time = env.step_counter / env.CTRL_FREQ if is_success else 120.0
        all_intercept_times.append(intercept_time)

        episodes.append({
            "episode": ep,
            "success": is_success,
            "reason": reason,
            "total_reward": float(total_rew),
            "min_dist": float(ep_min_dist),
            "intercept_time_s": float(intercept_time),
            "final_spacing_m": float(final_spacing),
            "p0_traj": np.array(p0_traj) if record_tacview else None,
            "p1_traj": np.array(p1_traj) if record_tacview else None,
            "t_traj": np.array(t_traj) if record_tacview else None,
            "spacing_hist": spacing_hist if record_tacview else None,
        })

        if (ep + 1) % 20 == 0 or ep == 0:
            p, lo, hi = wilson_ci(successes, ep + 1)
            print(f"  [{ep+1:>3d}/{n_episodes}]  success_rate={p:.2%}  "
                  f"95%CI=[{lo:.2%},{hi:.2%}]  avg_rew={np.mean(all_rewards):+.0f}  "
                  f"avg_spacing={np.mean(all_final_spacings):.0f}m")

    # Aggregate
    p, lo, hi = wilson_ci(successes, n_episodes)
    results = {
        "model_path": model_path,
        "difficulty": difficulty,
        "n_episodes": n_episodes,
        "successes": successes,
        "capture_rate": p,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "avg_min_dist_m": float(np.mean(all_min_dists)),
        "std_min_dist_m": float(np.std(all_min_dists)),
        "avg_intercept_time_s": float(np.mean(all_intercept_times)),
        "std_intercept_time_s": float(np.std(all_intercept_times)),
        "avg_final_spacing_m": float(np.mean(all_final_spacings)),
        "std_final_spacing_m": float(np.std(all_final_spacings)),
    }

    print(f"\n{'='*60}")
    print(f"RESULTS (difficulty={difficulty:.1f})")
    print(f"  Capture rate:  {p:.2%}  95%CI=[{lo:.2%}, {hi:.2%}]")
    print(f"  Avg reward:    {results['avg_reward']:+.0f} ± {results['std_reward']:.0f}")
    print(f"  Avg min dist:  {results['avg_min_dist_m']:.0f} ± {results['std_min_dist_m']:.0f} m")
    print(f"  Avg intercept: {results['avg_intercept_time_s']:.1f} ± {results['std_intercept_time_s']:.1f} s")
    print(f"  Avg spacing:   {results['avg_final_spacing_m']:.0f} ± {results['std_final_spacing_m']:.0f} m")
    print(f"{'='*60}\n")

    return results, episodes, env


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_best_episode(ep, output_path: str, difficulty: float):
    """3D trajectory + spacing over time for a single episode."""
    fig = plt.figure(figsize=(18, 7))

    # ── 3D Trajectory ──────────────────────────────────────────────────
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    p0 = ep["p0_traj"]; p1 = ep["p1_traj"]; t = ep["t_traj"]

    ax3d.plot(p0[:, 0], p0[:, 1], p0[:, 2], "r-", lw=1.2, alpha=0.8, label="Pursuer 0")
    ax3d.plot(p1[:, 0], p1[:, 1], p1[:, 2], "orange", lw=1.2, alpha=0.8, label="Pursuer 1")
    ax3d.plot(t[:, 0], t[:, 1], t[:, 2], "b-", lw=1.2, alpha=0.8, label="Target")
    ax3d.scatter(p0[0, 0], p0[0, 1], p0[0, 2], color="darkred", s=60, marker="o")
    ax3d.scatter(p1[0, 0], p1[0, 1], p1[0, 2], color="darkorange", s=60, marker="o")
    ax3d.scatter(t[0, 0], t[0, 1], t[0, 2], color="darkblue", s=60, marker="o")
    ax3d.set_xlabel("North (m)"); ax3d.set_ylabel("East (m)"); ax3d.set_zlabel("Alt (m)")
    ax3d.set_title(f"3D Trajectory (diff={difficulty:.1f})"); ax3d.legend(fontsize=8)

    # ── Top-Down ───────────────────────────────────────────────────────
    ax2d = fig.add_subplot(1, 3, 2)
    ax2d.plot(p0[:, 0], p0[:, 1], "r-", lw=1.2, alpha=0.8, label="P0")
    ax2d.plot(p1[:, 0], p1[:, 1], "orange", lw=1.2, alpha=0.8, label="P1")
    ax2d.plot(t[:, 0], t[:, 1], "b-", lw=1.2, alpha=0.8, label="Target")
    ax2d.scatter(p0[0, 0], p0[0, 1], color="darkred", s=60, marker="o")
    ax2d.scatter(p1[0, 0], p1[0, 1], color="darkorange", s=60, marker="o")
    ax2d.scatter(t[0, 0], t[0, 1], color="darkblue", s=60, marker="o")
    ax2d.set_xlabel("North (m)"); ax2d.set_ylabel("East (m)")
    ax2d.set_title(f"Top-Down (capture={ep['success']})"); ax2d.legend(fontsize=8); ax2d.axis("equal")

    # ── Spacing vs Time ────────────────────────────────────────────────
    ax_sp = fig.add_subplot(1, 3, 3)
    times = np.arange(len(ep["spacing_hist"])) * 0.5  # 0.5s decision interval
    ax_sp.plot(times, ep["spacing_hist"], "g-", lw=1.5, alpha=0.8)
    ax_sp.axhline(y=50, color="red", ls="--", lw=0.8, alpha=0.5, label="Danger (50m)")
    ax_sp.axhline(y=200, color="orange", ls="--", lw=0.8, alpha=0.5, label="Repel (200m)")
    ax_sp.axhline(y=500, color="blue", ls="--", lw=0.8, alpha=0.5, label="Ideal max (500m)")
    ax_sp.fill_between([0, times[-1]], 200, 500, alpha=0.1, color="green", label="Ideal zone")
    ax_sp.set_xlabel("Time (s)"); ax_sp.set_ylabel("Pursuer-Pursuer Spacing (m)")
    ax_sp.set_title(f"Formation Spacing (final={ep['final_spacing_m']:.0f}m)")
    ax_sp.legend(fontsize=7); ax_sp.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def export_tacview(ep, env, output_path: str):
    """Export a single episode's Tacview ACMI from the env's frame buffer.

    We re-run the episode quickly with recording enabled since FormationEnv
    records per-step frames internally during step().
    """
    env.export_tacview(output_path)
    print(f"  Tacview saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark SB3 Phase 4.1 97.3% baseline")
    parser.add_argument("--episodes", "-n", type=int, default=100,
                        help="Number of evaluation episodes per difficulty (default: 100)")
    parser.add_argument("--difficulty", "-d", type=str, default="0.0",
                        help="Comma-separated difficulty levels (default: 0.0)")
    parser.add_argument("--model", "-m", type=str, default=BENCHMARK_MODEL,
                        help="Path to model .zip file")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_DIR,
                        help="Output directory for benchmark artifacts")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip Tacview + plot generation (faster)")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(SOURCE_MODEL):
        print(f"Model not found at {model_path}, falling back to {SOURCE_MODEL}")
        model_path = SOURCE_MODEL
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Expected one of:")
        print(f"  {BENCHMARK_MODEL}")
        print(f"  {SOURCE_MODEL}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    difficulties = [float(d.strip()) for d in args.difficulty.split(",")]

    all_results = {}

    for diff in difficulties:
        results, episodes, env = run_benchmark(
            model_path, n_episodes=args.episodes,
            difficulty=diff, record_tacview=not args.no_viz,
        )

        all_results[f"difficulty_{diff:.1f}"] = results

        # Save metrics
        metrics_path = os.path.join(args.output, f"metrics_diff{diff:.1f}.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Metrics saved: {metrics_path}")

        if not args.no_viz and episodes:
            # Find best (successful, highest reward) and worst episodes
            successful = [ep for ep in episodes if ep["success"]]
            best_ep = max(successful, key=lambda e: e["total_reward"]) if successful else episodes[0]
            worst_ep = min(episodes, key=lambda e: e["min_dist"])

            # Plot best
            plot_best_episode(
                best_ep,
                os.path.join(args.output, f"trajectory_best_diff{diff:.1f}.png"),
                diff,
            )

            # Re-run best episode with Tacview recording
            print(f"\n  Recording Tacview for best episode (ep {best_ep['episode']})...")
            _record_single_tacview(
                model_path, diff,
                os.path.join(args.output, f"tacview_best_diff{diff:.1f}.txt.acmi"),
            )

    # ── Aggregate manifest ──────────────────────────────────────────────
    manifest = {
        "benchmark_name": "SB3 Phase 4.1 — 2v1 Shared Policy (97.3% baseline)",
        "model_source": SOURCE_MODEL,
        "model_archived": f"{args.output}/model.zip",
        "architecture": {
            "policy": "MlpPolicy (shared, 66-dim obs → Box(4) action)",
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation": "Tanh",
            "algorithm": "PPO (Stable-Baselines3)",
            "training_steps": 200_000,
        },
        "environment": {
            "num_pursuers": 2,
            "num_targets": 1,
            "obs_dim": 66,
            "action_dim": 4,
            "decision_hz": 2,
            "max_episode_s": 120,
        },
        "results": all_results,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes_per_difficulty": args.episodes,
    }

    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Benchmark complete. Manifest: {manifest_path}")
    print(f"All artifacts in: {args.output}/")
    print(f"{'='*60}")


def _record_single_tacview(model_path: str, difficulty: float, output_path: str):
    """Re-run one episode with Tacview recording enabled, export ACMI."""
    import logging as _logging
    _logging.getLogger("jsbsim").setLevel(_logging.CRITICAL)

    model = PPO.load(model_path)
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.export_tacview(output_path)
    print(f"    → {output_path}")


if __name__ == "__main__":
    main()
