"""Fig 3: Role-grouped averaged Self-Attention matrices (large-sample paper figure).

Core innovation: instead of plotting attention timelines per-episode (Fig 2),
we aggregate across N episodes, group agents by ROLE (Striker vs Interceptor)
rather than agent ID (P0 vs P1), and visualize the average attention matrix.

This directly answers: "Does the Self-Attention architecture learn a
position-invariant coordination strategy, or is it just memorizing P0/P1 IDs?"

Key panels:
  Left:  Averaged MHA attention matrix for Striker role [3×3]
  Right: Averaged MHA attention matrix for Interceptor role [3×3]
  Below: Pool weight distribution (violin/box) per token per role

Usage:
    conda activate marl_env
    python scripts/viz_fig3_role_attention.py --data data/viz/exp2_eval_50ep.npz
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

# ── Academic styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

TOKEN_NAMES = ["Self", "Target", "Mate"]
ROLE_COLORS = {"Striker": "#377eb8", "Interceptor": "#ff7f00"}
TOKEN_COLORS = ["#4daf4a", "#e41a1c", "#377eb8"]  # Green, Red, Blue


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading & role grouping
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_group(data_path: str, min_steps: int = 20):
    """Load all episodes, classify each agent-step by role.

    Returns:
        striker_mha:   list of [3, 3] MHA matrices (all steps where agent is Striker)
        interceptor_mha: list of [3, 3] MHA matrices
        striker_pool:  list of [3] pool weights
        interceptor_pool: list of [3] pool weights
        role_stats:    dict with episode-level stats
    """
    data = np.load(data_path, allow_pickle=True)
    n_eps = int(data["n_episodes"])

    striker_mha = []
    interceptor_mha = []
    striker_pool = []
    interceptor_pool = []

    role_stats = {
        "n_striker_p0": 0, "n_striker_p1": 0,
        "n_episodes_used": 0,
        "striker_d_min": [], "interceptor_d_min": [],
        "pincer_means": [],
    }

    for ep_idx in range(n_eps):
        prefix = f"ep{ep_idx}_"
        steps = int(data[f"{prefix}n_steps"])

        if steps < min_steps:
            continue  # skip crashed/truncated episodes

        role_stats["n_episodes_used"] += 1

        d0 = data[f"{prefix}p0_distances"]
        d1 = data[f"{prefix}p1_distances"]

        # Determine overall Striker (min distance to target)
        d0_min = float(d0.min())
        d1_min = float(d1.min())
        overall_striker = "P0" if d0_min < d1_min else "P1"
        if overall_striker == "P0":
            role_stats["n_striker_p0"] += 1
        else:
            role_stats["n_striker_p1"] += 1
        role_stats["striker_d_min"].append(min(d0_min, d1_min))
        role_stats["interceptor_d_min"].append(max(d0_min, d1_min))

        pincer = data.get(f"{prefix}pincer_angles", np.array([]))
        if len(pincer) > 0:
            role_stats["pincer_means"].append(float(pincer.mean()))

        # Load attention data (interleaved: p0 step0, p1 step0, ...)
        if f"{prefix}attn_weights" not in data:
            continue

        attn_all = np.squeeze(data[f"{prefix}attn_weights"], axis=1)  # [2T, 3, 3]
        pool_all = np.squeeze(data[f"{prefix}pool_weights"], axis=1)  # [2T, 3]

        T = min(steps, len(attn_all) // 2)

        for t in range(T):
            # Per-step role classification
            d0_t = float(d0[t])
            d1_t = float(d1[t])

            # P0 data at even indices, P1 at odd
            attn_p0 = attn_all[2 * t]      # [3, 3]
            attn_p1 = attn_all[2 * t + 1]  # [3, 3]
            pool_p0 = pool_all[2 * t]      # [3]
            pool_p1 = pool_all[2 * t + 1]  # [3]

            if d0_t <= d1_t:
                # P0 is Striker at this step
                striker_mha.append(attn_p0)
                striker_pool.append(pool_p0)
                interceptor_mha.append(attn_p1)
                interceptor_pool.append(pool_p1)
            else:
                # P1 is Striker at this step
                striker_mha.append(attn_p1)
                striker_pool.append(pool_p1)
                interceptor_mha.append(attn_p0)
                interceptor_pool.append(pool_p0)

    result = {
        "striker_mha": np.array(striker_mha) if striker_mha else np.array([]),
        "interceptor_mha": np.array(interceptor_mha) if interceptor_mha else np.array([]),
        "striker_pool": np.array(striker_pool) if striker_pool else np.array([]),
        "interceptor_pool": np.array(interceptor_pool) if interceptor_pool else np.array([]),
        "role_stats": role_stats,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3: Role-grouped averaged attention
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3(grouped: dict, save_path: str):
    """Generate the paper's Fig 3: Averaged Attention Matrix by Role.

    Layout (2×2 grid):
      Top-left:  Striker MHA [3×3] heatmap
      Top-right: Interceptor MHA [3×3] heatmap
      Bottom-left:  Pool weight distribution — Striker
      Bottom-right: Pool weight distribution — Interceptor

    With annotations:
      - Mean ± SEM per cell
      - Statistical comparison arrows between Striker/Interceptor
      - Sample size (N steps, N episodes)
    """
    s_mha = grouped["striker_mha"]       # [N_s, 3, 3]
    i_mha = grouped["interceptor_mha"]   # [N_i, 3, 3]
    s_pool = grouped["striker_pool"]     # [N_s, 3]
    i_pool = grouped["interceptor_pool"] # [N_i, 3]
    stats = grouped["role_stats"]

    if len(s_mha) == 0 or len(i_mha) == 0:
        print("[ERROR] No attention data after filtering")
        return

    fig = plt.figure(figsize=(9, 8))

    # ── Color normalization range for heatmaps ────────────────────────────
    vmin = min(s_mha.mean(axis=0).min(), i_mha.mean(axis=0).min())
    vmax = max(s_mha.mean(axis=0).max(), i_mha.mean(axis=0).max())
    # Ensure 1/3 baseline is at center of colormap
    vcenter = 1 / 3
    vrange = max(vmax - vcenter, vcenter - vmin) * 1.1
    vmin_c = vcenter - vrange
    vmax_c = vcenter + vrange

    # ═══════════════════════════════════════════════════════════════════════
    #  Top-left: Striker MHA Matrix
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(2, 2, 1)
    s_mean = s_mha.mean(axis=0)  # [3, 3]
    s_sem = s_mha.std(axis=0) / np.sqrt(len(s_mha))  # [3, 3]

    im1 = ax1.imshow(s_mean, cmap="RdBu_r", vmin=vmin_c, vmax=vmax_c,
                      aspect="equal", interpolation="nearest")

    # Annotate each cell with mean±SEM
    for i in range(3):
        for j in range(3):
            val = s_mean[i, j]
            err = s_sem[i, j]
            color = "white" if abs(val - vcenter) > vrange * 0.5 else "black"
            ax1.text(j, i, f"{val:.3f}\n±{err:.3f}",
                     ha="center", va="center", fontsize=7,
                     color=color, linespacing=1.2)

    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(TOKEN_NAMES, fontsize=8)
    ax1.set_yticklabels(TOKEN_NAMES, fontsize=8)
    ax1.set_xlabel("Key (attended TO)", fontsize=8)
    ax1.set_ylabel("Query (attend FROM)", fontsize=8)
    ax1.set_title(f"Striker MHA Attention\n"
                  f"N={len(s_mha)} steps, {stats['n_episodes_used']} eps",
                  fontsize=9, color=ROLE_COLORS["Striker"])

    # ═══════════════════════════════════════════════════════════════════════
    #  Top-right: Interceptor MHA Matrix
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(2, 2, 2)
    i_mean = i_mha.mean(axis=0)
    i_sem = i_mha.std(axis=0) / np.sqrt(len(i_mha))

    im2 = ax2.imshow(i_mean, cmap="RdBu_r", vmin=vmin_c, vmax=vmax_c,
                      aspect="equal", interpolation="nearest")

    for i in range(3):
        for j in range(3):
            val = i_mean[i, j]
            err = i_sem[i, j]
            color = "white" if abs(val - vcenter) > vrange * 0.5 else "black"
            ax2.text(j, i, f"{val:.3f}\n±{err:.3f}",
                     ha="center", va="center", fontsize=7,
                     color=color, linespacing=1.2)

    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(TOKEN_NAMES, fontsize=8)
    ax2.set_yticklabels(TOKEN_NAMES, fontsize=8)
    ax2.set_xlabel("Key (attended TO)", fontsize=8)
    ax2.set_ylabel("Query (attend FROM)", fontsize=8)
    ax2.set_title(f"Interceptor MHA Attention\n"
                  f"N={len(i_mha)} steps",
                  fontsize=9, color=ROLE_COLORS["Interceptor"])

    # ── Shared colorbar for MHA ───────────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.55, 0.015, 0.33])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Attention Weight", fontsize=8)
    cbar.ax.axhline(y=vcenter, color="black", linestyle="--", linewidth=1)
    cbar.ax.text(0.5, vcenter, "1/3", fontsize=6, ha="left", va="center",
                 transform=cbar.ax.get_yaxis_transform())

    # ═══════════════════════════════════════════════════════════════════════
    #  Bottom: Pool weight comparison (grouped bar chart)
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(2, 2, (3, 4))

    x = np.arange(3)
    width = 0.35

    s_pool_mean = s_pool.mean(axis=0)  # [3]
    i_pool_mean = i_pool.mean(axis=0)  # [3]
    s_pool_sem = s_pool.std(axis=0) / np.sqrt(len(s_pool))
    i_pool_sem = i_pool.std(axis=0) / np.sqrt(len(i_pool))

    bars1 = ax3.bar(x - width/2, s_pool_mean, width,
                    yerr=s_pool_sem, capsize=3,
                    color=ROLE_COLORS["Striker"], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    label=f"Striker (n={len(s_pool)})")
    bars2 = ax3.bar(x + width/2, i_pool_mean, width,
                    yerr=i_pool_sem, capsize=3,
                    color=ROLE_COLORS["Interceptor"], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    label=f"Interceptor (n={len(i_pool)})")

    # Annotate significance
    for idx in range(3):
        diff = s_pool_mean[idx] - i_pool_mean[idx]
        se_diff = np.sqrt(s_pool_sem[idx]**2 + i_pool_sem[idx]**2)
        z = abs(diff) / max(se_diff, 1e-9)
        sig = "***" if z > 3 else "**" if z > 2 else "*" if z > 1.5 else "ns"
        y_max = max(s_pool_mean[idx] + s_pool_sem[idx],
                    i_pool_mean[idx] + i_pool_sem[idx])
        ax3.text(x[idx], y_max + 0.02, sig, ha="center", fontsize=8,
                 fontweight="bold" if sig != "ns" else "normal")

    ax3.axhline(y=1/3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.text(2.5, 1/3, "uniform (1/3)", fontsize=6, va="center", color="gray")

    ax3.set_xticks(x)
    ax3.set_xticklabels(TOKEN_NAMES, fontsize=9)
    ax3.set_ylabel("Pool Weight", fontsize=9)
    ax3.set_ylim(0, 0.7)
    ax3.set_title("Learned Attention Pooling by Role\n"
                   "(mean ± SEM, *** p<0.001 by z-test)",
                   fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.8)
    ax3.grid(True, alpha=0.2, axis="y")

    # ── Super-title with key statistics ───────────────────────────────────
    n_striker_p0 = stats["n_striker_p0"]
    n_striker_p1 = stats["n_striker_p1"]
    avg_pincer = np.mean(stats["pincer_means"]) if stats["pincer_means"] else 0
    avg_s_d = np.mean(stats["striker_d_min"]) if stats["striker_d_min"] else 0
    avg_i_d = np.mean(stats["interceptor_d_min"]) if stats["interceptor_d_min"] else 0

    fig.suptitle(
        f"Role-Grouped Self-Attention: Striker vs Interceptor\n"
        f"Parameter-Shared MAPPO (CTDE) — {stats['n_episodes_used']} episodes, "
        f"{len(s_mha) + len(i_mha)} total steps | "
        f"Striker=P0 in {n_striker_p0} eps, P1 in {n_striker_p1} eps | "
        f"Mean pincer={avg_pincer:.0f}°, "
        f"d̄(Striker)={avg_s_d:.0f}m, d̄(Interceptor)={avg_i_d:.0f}m",
        fontsize=8, y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Fig 3 saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fig 3: Role-grouped averaged Self-Attention matrices")
    parser.add_argument("--data", type=str,
                        default="data/viz/exp2_eval_50ep.npz",
                        help="Path to multi-episode .npz file")
    parser.add_argument("--min-steps", type=int, default=20,
                        help="Minimum episode steps to include")
    parser.add_argument("--output", type=str,
                        default="results/viz/fig3_role_attention_matrix.pdf",
                        help="Output path for figure")
    args = parser.parse_args()

    grouped = load_and_group(args.data, min_steps=args.min_steps)

    if len(grouped["striker_mha"]) == 0:
        print("ERROR: No attention data collected. Check min_steps threshold.")
        sys.exit(1)

    stats = grouped["role_stats"]
    print(f"Loaded {stats['n_episodes_used']} episodes "
          f"(≥{args.min_steps} steps each)")
    print(f"  Striker = P0: {stats['n_striker_p0']}, "
          f"Striker = P1: {stats['n_striker_p1']}")
    print(f"  Striker steps: {len(grouped['striker_mha'])}, "
          f"Interceptor steps: {len(grouped['interceptor_mha'])}")
    print(f"  Mean Striker d_min: {np.mean(stats['striker_d_min']):.0f}m, "
          f"Interceptor d_min: {np.mean(stats['interceptor_d_min']):.0f}m")
    print(f"  Mean pincer angle: {np.mean(stats['pincer_means']):.1f}°")

    plot_fig3(grouped, args.output)
