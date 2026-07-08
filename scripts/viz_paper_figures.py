"""Generate paper-ready visualizations from collected trajectory + attention data.

Two figures:
  1. 3D Spatial Trajectories — P0/P1/Target paths showing Striker-Interceptor roles
  2. Self-Attention Weight Timeline — Mate vs Target token attention + pincer angle

Usage:
    conda activate marl_env
    python scripts/viz_paper_figures.py --data data/viz/exp2_best_trajectory.npz --episode 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Academic styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Color palette (colorblind-friendly, publication-safe) ─────────────────────
COLORS = {
    "p0": "#377eb8",       # Blue — Striker
    "p1": "#ff7f00",       # Orange — Interceptor
    "target": "#e41a1c",   # Red — Target
    "p0_light": "#a6cee3",
    "p1_light": "#fdbf6f",
    "grid": "#e0e0e0",
    "text": "#333333",
}
TOKEN_NAMES = ["Self", "Target", "Mate"]
TOKEN_COLORS = ["#4daf4a", "#e41a1c", "#377eb8"]  # Green, Red, Blue


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1: 3D Spatial Trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3d_trajectories(ep_data: dict, save_path: str):
    """3D spatial plot showing P0 (Striker), P1 (Interceptor), and Target paths.

    Key visual elements:
      - Trajectory ribbons in 3D
      - Start/end markers for each entity
      - Pincer angle annotations at peak moments
      - Distance ring around target showing the 200m OR-gate success zone
    """
    pos_p0 = ep_data["p0_positions"]      # [T, 3] NED
    pos_p1 = ep_data["p1_positions"]      # [T, 3] NED
    pos_t = ep_data["target_positions"]    # [T, 3] NED
    pincer = ep_data.get("pincer_angles", np.array([]))

    # ── Center on target initial position ─────────────────────────────────
    origin = pos_t[0].copy()
    p0_rel = pos_p0 - origin
    p1_rel = pos_p1 - origin
    t_rel = pos_t - origin

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ── Plot trajectories ──────────────────────────────────────────────────
    T = len(p0_rel)
    alpha_curve = np.linspace(0.4, 1.0, T)

    for i in range(T - 1):
        ax.plot(
            p0_rel[i:i+2, 1], p0_rel[i:i+2, 0], p0_rel[i:i+2, 2],
            color=COLORS["p0"], linewidth=1.5, alpha=alpha_curve[i],
        )
        ax.plot(
            p1_rel[i:i+2, 1], p1_rel[i:i+2, 0], p1_rel[i:i+2, 2],
            color=COLORS["p1"], linewidth=1.5, alpha=alpha_curve[i],
        )
        ax.plot(
            t_rel[i:i+2, 1], t_rel[i:i+2, 0], t_rel[i:i+2, 2],
            color=COLORS["target"], linewidth=1.2, alpha=alpha_curve[i],
            linestyle="--",
        )

    # ── Start/End markers ──────────────────────────────────────────────────
    for label, pos, color, marker, offset in [
        ("P0 Start", p0_rel[0], COLORS["p0"], "o", np.array([0, 0, 50])),
        ("P1 Start", p1_rel[0], COLORS["p1"], "o", np.array([0, 0, 50])),
        ("P0 End", p0_rel[-1], COLORS["p0"], "s", np.array([0, 0, 50])),
        ("P1 End", p1_rel[-1], COLORS["p1"], "s", np.array([0, 0, 50])),
        ("Target", t_rel[-1], COLORS["target"], "X", np.array([0, 0, 50])),
    ]:
        ax.scatter(pos[1], pos[0], pos[2], c=color, marker=marker,
                   s=40, edgecolors="white", linewidth=0.5, zorder=5)

    # ── Pincer angle annotation at peak ────────────────────────────────────
    if len(pincer) > 0:
        peak_idx = np.argmax(pincer)
        peak_val = pincer[peak_idx]
        apex = (p0_rel[peak_idx] + p1_rel[peak_idx]) / 2
        ax.text(
            apex[1], apex[0], apex[2] - 100,
            f"Pincer: {peak_val:.0f}°",
            fontsize=7, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # ── Styling ────────────────────────────────────────────────────────────
    ax.set_xlabel("East (m)", labelpad=8)
    ax.set_ylabel("North (m)", labelpad=8)
    ax.set_zlabel("Altitude (m)", labelpad=8)

    # Equal aspect ratio on horizontal plane
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    xy_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
    x_mid = (x_lim[0] + x_lim[1]) / 2
    y_mid = (y_lim[0] + y_lim[1]) / 2
    ax.set_xlim(x_mid - xy_range/2, x_mid + xy_range/2)
    ax.set_ylim(y_mid - xy_range/2, y_mid + xy_range/2)

    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=COLORS["p0"], linewidth=2, label="P0 (Striker)"),
        Line2D([0], [0], color=COLORS["p1"], linewidth=2, label="P1 (Interceptor)"),
        Line2D([0], [0], color=COLORS["target"], linewidth=1.5,
               linestyle="--", label="Target"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              framealpha=0.9, edgecolor="none")

    ax.set_title(
        f"3D Formation Pursuit Trajectory — Parameter-Shared MAPPO (CTDE)\n"
        f"d0_min={ep_data.get('d0_min', 0):.0f}m, "
        f"d1_min={ep_data.get('d1_min', 0):.0f}m, "
        f"Pincer max={pincer.max() if len(pincer) > 0 else 0:.0f}°",
        fontsize=9, pad=15,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] 3D Trajectory: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2: Self-Attention Weight Timeline
# ═══════════════════════════════════════════════════════════════════════════════

def plot_attention_timeline(ep_data: dict, save_path: str):
    """Attention weight evolution over a pursuit episode — P0 & P1 side-by-side.

    The hook captures attention for both agents interleaved (p0 step 0, p1 step 0,
    p0 step 1, ...). We separate them and show P0 (upper row) and P1 (lower row).

    For each agent — three subplot rows:
      Pool weights [Self, Target, Mate] over time
      MHA Self→Target vs Self→Mate attention
      Pincer angle (shared tactical context)
    """
    pool_all = ep_data.get("pool_weights", None)     # [2T, 1, 3]
    attn_all = ep_data.get("attn_weights", None)     # [2T, 1, 3, 3]
    pincer = ep_data.get("pincer_angles", np.array([]))

    if pool_all is None or pool_all.size == 0:
        print("[WARN] No attention weights — skipping figure 2")
        return

    # ── Reshape: interleaved [p0,p1,p0,p1,...] → [T, 2, ...] ────────────
    pool_all = np.squeeze(pool_all, axis=1)  # [2T, 3]
    if attn_all is not None:
        attn_all = np.squeeze(attn_all, axis=1)  # [2T, 3, 3]

    total_captures = len(pool_all)
    T_agents = total_captures // 2  # steps per agent

    pool_p0 = pool_all[0::2]  # [T, 3]
    pool_p1 = pool_all[1::2]  # [T, 3]

    if attn_all is not None:
        attn_p0 = attn_all[0::2]  # [T, 3, 3]
        attn_p1 = attn_all[1::2]  # [T, 3, 3]

    T = min(T_agents, len(pincer) if len(pincer) > 0 else T_agents)

    # ── Create figure: 2 columns (P0, P1) × 3 rows ────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex="col",
                              gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})

    t_axis = np.arange(T) * 0.2  # 5 Hz → seconds

    for col, (pool_w, attn_w, label) in enumerate([
        (pool_p0[:T], attn_p0[:T] if attn_all is not None else None, "P0 (Striker)"),
        (pool_p1[:T], attn_p1[:T] if attn_all is not None else None, "P1 (Interceptor)"),
    ]):
        color_agent = COLORS["p0"] if col == 0 else COLORS["p1"]

        # ── Row 1: Pool weights ──────────────────────────────────────────
        ax = axes[0, col]
        for idx, (name, tc) in enumerate(zip(TOKEN_NAMES, TOKEN_COLORS)):
            ax.plot(t_axis, pool_w[:T, idx], color=tc, linewidth=1.2,
                    label=name, alpha=0.85)
        ax.axhline(y=1/3, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Pool Weight", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", ncol=4, framealpha=0.8, fontsize=6)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{label} — Learned Attention Pooling", fontsize=9,
                     color=color_agent)

        # ── Row 2: MHA Self→Target vs Self→Mate ──────────────────────────
        ax = axes[1, col]
        if attn_w is not None:
            s2t = attn_w[:T, 0, 1]  # Self attends to Target
            s2m = attn_w[:T, 0, 2]  # Self attends to Mate

            ax.plot(t_axis, s2t, color=COLORS["target"], linewidth=1.5,
                    label="Self→Target", alpha=0.85)
            ax.plot(t_axis, s2m, color=COLORS["p1"], linewidth=1.5,
                    label="Self→Mate", alpha=0.85, linestyle="--")

            # Highlight coordination spikes
            if len(s2m) > 1:
                s2m_med = float(np.median(s2m))
                for i in range(T):
                    if s2m[i] > s2m_med + 0.05:
                        ax.axvspan(t_axis[max(0, i-1)], t_axis[min(T-1, i+1)],
                                   color=COLORS["p1"], alpha=0.06)

        ax.axhline(y=1/3, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Attention Weight", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", ncol=2, framealpha=0.8, fontsize=6)
        ax.grid(True, alpha=0.25)

        # ── Row 3: Pincer angle ──────────────────────────────────────────
        ax = axes[2, col]
        if len(pincer) > 0:
            ax.fill_between(t_axis, 0, pincer[:T], color=COLORS["p0_light"],
                            alpha=0.4)
            ax.plot(t_axis, pincer[:T], color=COLORS["p0"], linewidth=1.2)
            ax.axhline(y=30, color="green", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.axhline(y=60, color="orange", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Pincer (°)", fontsize=8)
        ax.grid(True, alpha=0.25)

    # Suppress redundant y-labels on right column
    for row in range(3):
        axes[row, 1].set_ylabel("")

    axes[2, 0].text(0.98, 0.95, "30° = AND threshold\n60° = ideal pincer",
                    transform=axes[2, 0].transAxes, fontsize=5,
                    va="top", ha="right", color="gray", alpha=0.7)

    fig.suptitle("Self-Attention Weight Dynamics During Cooperative Pursuit\n"
                 "Parameter-Shared MAPPO (CTDE) — OR-gate Phase",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Attention Timeline: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper-ready visualizations from trajectory + attention data")
    parser.add_argument("--data", type=str,
                        default="data/viz/exp2_best_trajectory.npz",
                        help="Path to collected .npz file")
    parser.add_argument("--episode", type=int, default=0,
                        help="Which episode to visualize (0-indexed)")
    parser.add_argument("--output-dir", type=str, default="results/viz",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    data = np.load(args.data, allow_pickle=True)
    n_eps = int(data["n_episodes"])

    if args.episode >= n_eps:
        print(f"Error: episode {args.episode} >= {n_eps} available episodes")
        sys.exit(1)

    prefix = f"ep{args.episode}_"
    ep = {}
    for aid in ["p0", "p1"]:
        for key in ["positions", "actions", "rewards", "distances"]:
            ep[f"{aid}_{key}"] = data[f"{prefix}{aid}_{key}"]

    ep["target_positions"] = data[f"{prefix}target_positions"]
    ep["pincer_angles"] = data[f"{prefix}pincer_angles"]
    ep["n_steps"] = int(data[f"{prefix}n_steps"])
    ep["total_reward"] = float(data[f"{prefix}total_reward"])

    # Compute min distances for display
    ep["d0_min"] = float(ep["p0_distances"].min())
    ep["d1_min"] = float(ep["p1_distances"].min())

    # Attention weights (may not exist for all episodes)
    attn_key = f"{prefix}attn_weights"
    pool_key = f"{prefix}pool_weights"
    if attn_key in data:
        ep["attn_weights"] = data[attn_key]
    if pool_key in data:
        ep["pool_weights"] = data[pool_key]

    print(f"Episode {args.episode}: {ep['n_steps']} steps, "
          f"rew={ep['total_reward']:.0f}, "
          f"d0_min={ep['d0_min']:.0f}m, d1_min={ep['d1_min']:.0f}m, "
          f"pincer_max={ep['pincer_angles'].max():.0f}°")

    # ── Generate figures ───────────────────────────────────────────────────
    traj_path = os.path.join(args.output_dir,
                             f"fig1_3d_trajectory_ep{args.episode}.pdf")
    plot_3d_trajectories(ep, traj_path)

    attn_path = os.path.join(args.output_dir,
                             f"fig2_attention_timeline_ep{args.episode}.pdf")
    plot_attention_timeline(ep, attn_path)

    print(f"\nDone! Figures saved to {args.output_dir}/")
