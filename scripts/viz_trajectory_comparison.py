"""Trajectory Comparison: Best (+2994) vs Worst (-8511) vs BC Baseline (+381).

Generates a 4-panel figure:
  Top-left:  Best checkpoint — best episode 3D trajectory
  Top-right: Worst checkpoint — worst episode 3D trajectory
  Bottom:    BC baseline — representative episode 3D trajectory
  Right panel: Distance + pincer angle timeline comparison

Usage:
    conda activate marl_env
    python scripts/viz_trajectory_comparison.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# ── Academic styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def load_data(npz_path: str) -> dict:
    """Load and parse .npz trajectory data."""
    data = np.load(npz_path, allow_pickle=True)
    n_eps = int(data["n_episodes"])

    episodes = []
    for ep_idx in range(n_eps):
        prefix = f"ep{ep_idx}_"
        ep = {
            "n_steps": int(data[f"{prefix}n_steps"]),
            "total_reward": float(data[f"{prefix}total_reward"]),
            "p0_positions": data[f"{prefix}p0_positions"],
            "p1_positions": data[f"{prefix}p1_positions"],
            "target_positions": data[f"{prefix}target_positions"],
            "p0_distances": data[f"{prefix}p0_distances"],
            "p1_distances": data[f"{prefix}p1_distances"],
            "pincer_angles": data[f"{prefix}pincer_angles"],
        }
        episodes.append(ep)
    return episodes


def plot_trajectory_comparison(
    best_eps: list, worst_eps: list, bc_eps: list,
    save_path: str
):
    """Generate 4-panel trajectory comparison figure."""
    # Select best/worst episodes from each dataset
    best_from_best = max(best_eps, key=lambda e: e["total_reward"])
    worst_from_best = min(best_eps, key=lambda e: e["total_reward"])
    best_from_worst = max(worst_eps, key=lambda e: e["total_reward"])
    worst_from_worst = min(worst_eps, key=lambda e: e["total_reward"])
    best_from_bc = max(bc_eps, key=lambda e: e["total_reward"])
    worst_from_bc = min(bc_eps, key=lambda e: e["total_reward"])

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35,
                  height_ratios=[1, 1, 1], width_ratios=[2, 2, 1.5])

    def plot_3d_traj(ax, ep, title, title_color="black"):
        """Plot 3D trajectory of a single episode."""
        p0 = ep["p0_positions"][:, :3]  # NED
        p1 = ep["p1_positions"][:, :3]
        tgt = ep["target_positions"][:, :3]
        T = min(len(p0), len(tgt))

        # Trajectories
        ax.plot(p0[:T, 1], p0[:T, 0], -p0[:T, 2],
                color="#e41a1c", linewidth=1.5, alpha=0.9, label="P0 (Striker)")
        ax.plot(p1[:T, 1], p1[:T, 0], -p1[:T, 2],
                color="#377eb8", linewidth=1.5, alpha=0.9, label="P1 (Interceptor)")
        ax.plot(tgt[:T, 1], tgt[:T, 0], -tgt[:T, 2],
                color="#333333", linewidth=1.2, alpha=0.7, linestyle="--", label="Target")

        # Start/end markers
        ax.scatter(*[p0[0, 1]], *[p0[0, 0]], *[-p0[0, 2]],
                   color="#e41a1c", s=80, marker="o", edgecolors="black", linewidth=0.5, zorder=5)
        ax.scatter(*[p1[0, 1]], *[p1[0, 0]], *[-p1[0, 2]],
                   color="#377eb8", s=80, marker="o", edgecolors="black", linewidth=0.5, zorder=5)
        ax.scatter(*[tgt[0, 1]], *[tgt[0, 0]], *[-tgt[0, 2]],
                   color="#333333", s=60, marker="s", edgecolors="black", linewidth=0.5, zorder=5)

        # End markers (X)
        ax.scatter(*[p0[min(T-1, len(p0)-1), 1]], *[p0[min(T-1, len(p0)-1), 0]],
                   *[-p0[min(T-1, len(p0)-1), 2]],
                   color="#e41a1c", s=40, marker="X", edgecolors="black", linewidth=0.5, zorder=5)
        ax.scatter(*[p1[min(T-1, len(p1)-1), 1]], *[p1[min(T-1, len(p1)-1), 0]],
                   *[-p1[min(T-1, len(p1)-1), 2]],
                   color="#377eb8", s=40, marker="X", edgecolors="black", linewidth=0.5, zorder=5)
        ax.scatter(*[tgt[min(T-1, len(tgt)-1), 1]], *[tgt[min(T-1, len(tgt)-1), 0]],
                   *[-tgt[min(T-1, len(tgt)-1), 2]],
                   color="#333333", s=30, marker="X", edgecolors="black", linewidth=0.5, zorder=5)

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title(title, fontsize=10, color=title_color, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left")
        ax.view_init(elev=25, azim=-60)

    def plot_timeline(ax, ep, label, color):
        """Plot distance + pincer angle timeline."""
        T = ep["n_steps"]
        t = np.arange(T) * 0.2  # 5 Hz decision rate

        d0 = ep["p0_distances"][:T] / 1000
        d1 = ep["p1_distances"][:T] / 1000
        pincer = ep["pincer_angles"][:T]

        ax2 = ax.twinx()
        ax.plot(t, d0, color=color, linewidth=1.5, alpha=0.8, linestyle="-")
        ax.plot(t, d1, color=color, linewidth=1.5, alpha=0.5, linestyle="--")
        ax.fill_between(t, 0, 0.2, alpha=0.1, color="green", label="OR success zone (<200m)")
        ax2.plot(t, pincer, color="#ff7f00", linewidth=1.0, alpha=0.6, linestyle=":")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (km)", color=color)
        ax2.set_ylabel("Pincer Angle (°)", color="#ff7f00")
        ax.set_ylim(0, max(max(d0.max(), d1.max()) * 1.1, 3))
        ax2.set_ylim(0, 180)
        ax.grid(True, alpha=0.2)

    # ═══════════════════════════════════════════════════════════════════════
    # Row 1: Best checkpoint — best vs worst episode
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_3d_traj(ax1, best_from_best,
                 f"Best Ckpt (+2994 eval) — Best Ep\n"
                 f"Rew={best_from_best['total_reward']:.0f}, "
                 f"d0_min={min(best_from_best['p0_distances']):.0f}m, "
                 f"d1_min={min(best_from_best['p1_distances']):.0f}m, "
                 f"pincer_max={max(best_from_best['pincer_angles']):.0f}°",
                 title_color="#2ca02c")

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    plot_3d_traj(ax2, worst_from_best,
                 f"Best Ckpt (+2994 eval) — Worst Ep\n"
                 f"Rew={worst_from_best['total_reward']:.0f}, "
                 f"d0_min={min(worst_from_best['p0_distances']):.0f}m, "
                 f"d1_min={min(worst_from_best['p1_distances']):.0f}m, "
                 f"pincer_max={max(worst_from_best['pincer_angles']):.0f}°",
                 title_color="#d62728")

    # Timelines for Row 1
    ax_t1 = fig.add_subplot(gs[0, 2])
    plot_timeline(ax_t1, best_from_best, "Best Ep", "#2ca02c")

    # ═══════════════════════════════════════════════════════════════════════
    # Row 2: Worst checkpoint — best vs worst episode
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_3d_traj(ax3, best_from_worst,
                 f"Worst Ckpt (−8511 eval) — Best Ep\n"
                 f"Rew={best_from_worst['total_reward']:.0f}, "
                 f"d0_min={min(best_from_worst['p0_distances']):.0f}m, "
                 f"d1_min={min(best_from_worst['p1_distances']):.0f}m, "
                 f"pincer_max={max(best_from_worst['pincer_angles']):.0f}°",
                 title_color="#2ca02c")

    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    plot_3d_traj(ax4, worst_from_worst,
                 f"Worst Ckpt (−8511 eval) — Worst Ep\n"
                 f"Rew={worst_from_worst['total_reward']:.0f}, "
                 f"d0_min={min(worst_from_worst['p0_distances']):.0f}m, "
                 f"d1_min={min(worst_from_worst['p1_distances']):.0f}m, "
                 f"pincer_max={max(worst_from_worst['pincer_angles']):.0f}°",
                 title_color="#d62728")

    # Timelines for Row 2
    ax_t2 = fig.add_subplot(gs[1, 2])
    plot_timeline(ax_t2, worst_from_worst, "Worst Ep", "#d62728")

    # ═══════════════════════════════════════════════════════════════════════
    # Row 3: BC Baseline — best vs worst episode
    # ═══════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 0], projection="3d")
    plot_3d_traj(ax5, best_from_bc,
                 f"BC Baseline (+381 eval) — Best Ep\n"
                 f"Rew={best_from_bc['total_reward']:.0f}, "
                 f"d0_min={min(best_from_bc['p0_distances']):.0f}m, "
                 f"d1_min={min(best_from_bc['p1_distances']):.0f}m, "
                 f"pincer_max={max(best_from_bc['pincer_angles']):.0f}°",
                 title_color="#2ca02c")

    ax6 = fig.add_subplot(gs[2, 1], projection="3d")
    plot_3d_traj(ax6, worst_from_bc,
                 f"BC Baseline (+381 eval) — Worst Ep\n"
                 f"Rew={worst_from_bc['total_reward']:.0f}, "
                 f"d0_min={min(worst_from_bc['p0_distances']):.0f}m, "
                 f"d1_min={min(worst_from_bc['p1_distances']):.0f}m, "
                 f"pincer_max={max(worst_from_bc['pincer_angles']):.0f}°",
                 title_color="#d62728")

    # Timelines for Row 3
    ax_t3 = fig.add_subplot(gs[2, 2])
    plot_timeline(ax_t3, best_from_bc, "BC Best Ep", "#2ca02c")

    # ── Super-title ───────────────────────────────────────────────────────
    fig.suptitle(
        "Trajectory Comparison: Best (+2994) vs Worst (−8511) vs BC Baseline (+381)\n"
        "Rows: checkpoints | Left: best episode | Middle: worst episode | Right: distance+pincer timeline\n"
        "◦ = start, × = end, solid = P0 (Striker), dashed = P1 (Interceptor), dotted = Target",
        fontsize=9, y=1.01
    )

    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Trajectory comparison saved: {save_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Episode Comparison Summary")
    print(f"{'='*60}")
    for label, ep in [
        ("Best Ckpt / Best Ep", best_from_best),
        ("Best Ckpt / Worst Ep", worst_from_best),
        ("Worst Ckpt / Best Ep", best_from_worst),
        ("Worst Ckpt / Worst Ep", worst_from_worst),
        ("BC Baseline / Best Ep", best_from_bc),
        ("BC Baseline / Worst Ep", worst_from_bc),
    ]:
        print(f"  {label:30s}: rew={ep['total_reward']:8.0f}  "
              f"steps={ep['n_steps']:4d}  "
              f"d0_min={min(ep['p0_distances']):6.0f}m  "
              f"d1_min={min(ep['p1_distances']):6.0f}m  "
              f"pincer_max={max(ep['pincer_angles']):6.1f}°")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trajectory comparison: Best vs Worst vs BC Baseline")
    parser.add_argument("--best-npz", type=str,
                        default="data/viz/exp5_best_2994_traj.npz",
                        help="Path to best checkpoint trajectory data")
    parser.add_argument("--worst-npz", type=str,
                        default="data/viz/exp5_worst_8511_traj.npz",
                        help="Path to worst checkpoint trajectory data")
    parser.add_argument("--bc-npz", type=str,
                        default="data/viz/exp5_bc_baseline_381_traj.npz",
                        help="Path to BC baseline trajectory data")
    parser.add_argument("--output", type=str,
                        default="results/viz/fig_trajectory_comparison.pdf",
                        help="Output path for figure")
    args = parser.parse_args()

    best_eps = load_data(args.best_npz)
    worst_eps = load_data(args.worst_npz)
    bc_eps = load_data(args.bc_npz)

    print(f"Best checkpoint:    {len(best_eps)} episodes, "
          f"rew range [{min(e['total_reward'] for e in best_eps):.0f}, "
          f"{max(e['total_reward'] for e in best_eps):.0f}]")
    print(f"Worst checkpoint:   {len(worst_eps)} episodes, "
          f"rew range [{min(e['total_reward'] for e in worst_eps):.0f}, "
          f"{max(e['total_reward'] for e in worst_eps):.0f}]")
    print(f"BC baseline:        {len(bc_eps)} episodes, "
          f"rew range [{min(e['total_reward'] for e in bc_eps):.0f}, "
          f"{max(e['total_reward'] for e in bc_eps):.0f}]")

    plot_trajectory_comparison(best_eps, worst_eps, bc_eps, args.output)
