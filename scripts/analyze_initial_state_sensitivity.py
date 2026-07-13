"""Initial State Sensitivity Analysis for OR-gate MAPPO Checkpoints.

Runs N episodes with fixed checkpoint (= no training), recording initial
conditions and final rewards to quantify the "eval variance" phenomenon.

Key questions:
  1. Does initial distance asymmetry predict episode outcome?
  2. Does initial bearing offset to target predict episode outcome?
  3. Is there a "sweet spot" region in initial-condition space where the
     policy reliably succeeds, and a "death zone" where it always fails?

Output:
  - data/viz/initial_state_sensitivity.npz  (raw data)
  - results/viz/fig_init_state_sensitivity.pdf/png  (scatter + marginal histograms)

Usage:
    conda activate marl_env
    python scripts/analyze_initial_state_sensitivity.py \
        --ckpt PATH_TO_CHECKPOINT --episodes 100 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Academic styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Skip RLlib/ray overhead for stateless eval
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")


def run_sensitivity_analysis(ckpt_path: str, n_episodes: int = 100, seed: int = 42):
    """Run N eval episodes with different seeds, recording initial state + reward."""
    import sys
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from src.environment.formation_rllib_env import FormationRLlibEnv
    from src.models.formation_rllib_model import RLlibAttentionActor

    register_env("formation_2v1_rllib", lambda c: FormationRLlibEnv(c))
    ModelCatalog.register_custom_model("attention_formation", RLlibAttentionActor)

    ckpt_abs = os.path.abspath(ckpt_path)
    print(f"Loading checkpoint: {ckpt_abs}")
    algo = PPO.from_checkpoint(ckpt_abs)

    # Create env manually (same approach as collect_viz_data.py)
    env = FormationRLlibEnv({
        "difficulty_level": 0.0,
        "lock_altitude": True,
        "record_tacview": False,
        "cooperative_mode": True,
    })

    results = {
        "n_episodes": n_episodes,
        "init_d0": [],           # initial p0 distance to target
        "init_d1": [],           # initial p1 distance to target
        "init_dist_diff": [],    # |d0 - d1| initial
        "init_bearing_err": [],  # mean bearing error to target
        "init_yaw_diff": [],     # |yaw_p0 - yaw_p1|
        "init_target_dist": [],  # target distance from cluster center
        "init_asymmetric": [],   # was asymmetric reset?
        "init_disadvantaged": [], # which pursuer was disadvantaged
        "total_reward": [],
        "ep_length": [],
        "min_d0": [],
        "min_d1": [],
        "max_pincer": [],
        "p0_reward": [],
        "p1_reward": [],
    }

    rng = np.random.default_rng(seed)
    base_seeds = rng.integers(0, 2**31 - 1, size=n_episodes)

    for ep_idx in range(n_episodes):
        ep_seed = int(base_seeds[ep_idx])

        # Reset env with specific seed
        obs_dict, _ = env.reset(seed=ep_seed)

        # ── Record initial conditions ────────────────────────────────────
        p0_pos = env.pursuers[0].aircraft.position_ned.copy()
        p1_pos = env.pursuers[1].aircraft.position_ned.copy()
        target_pos = env.targets[0].aircraft.position_ned.copy()

        d0_init = float(np.linalg.norm(p0_pos - target_pos))
        d1_init = float(np.linalg.norm(p1_pos - target_pos))

        p0_yaw = float(env.pursuers[0].aircraft.state["yaw_deg"])
        p1_yaw = float(env.pursuers[1].aircraft.state["yaw_deg"])

        # Bearing error: angle between pursuer heading and target direction
        def bearing_error(p_pos, p_yaw, t_pos):
            vec_to_target = t_pos - p_pos
            bearing = float(np.degrees(np.arctan2(vec_to_target[1], vec_to_target[0]))) % 360
            err = abs(bearing - p_yaw) % 360
            if err > 180:
                err = 360 - err
            return err

        be0 = bearing_error(p0_pos, p0_yaw, target_pos)
        be1 = bearing_error(p1_pos, p1_yaw, target_pos)

        results["init_d0"].append(d0_init)
        results["init_d1"].append(d1_init)
        results["init_dist_diff"].append(abs(d0_init - d1_init))
        results["init_bearing_err"].append((be0 + be1) / 2)
        results["init_yaw_diff"].append(abs(p0_yaw - p1_yaw) % 360)
        results["init_target_dist"].append(float(np.linalg.norm(target_pos - np.array([0, 0, 3000]))))
        results["init_asymmetric"].append(int(getattr(env, "_last_asymmetric", False)))
        results["init_disadvantaged"].append(int(getattr(env, "_last_disadvantaged", -1)))

        # ── Run episode (using algo.compute_single_action for correct inference) ─
        ep_rewards = []
        min_d0_ep = d0_init
        min_d1_ep = d1_init
        pincer_angles = []
        step = 0

        done = False
        while not done:
            actions = {}
            for aid in env._agent_ids:
                if aid in obs_dict:
                    raw_act = algo.compute_single_action(
                        obs_dict[aid], policy_id="shared_policy", explore=False)
                    # MultiDiscrete action is returned as array([turn, speed])
                    actions[aid] = np.array(raw_act, dtype=np.int64)

            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

            # Track per-step metrics
            for aid in ["p0", "p1"]:
                if aid in rewards:
                    ep_rewards.append(rewards[aid])

            # Update min distances
            d0 = float(np.linalg.norm(
                env.pursuers[0].aircraft.position_ned - env.targets[0].aircraft.position_ned))
            d1 = float(np.linalg.norm(
                env.pursuers[1].aircraft.position_ned - env.targets[0].aircraft.position_ned))
            min_d0_ep = min(min_d0_ep, d0)
            min_d1_ep = min(min_d1_ep, d1)

            # Compute pincer angle
            p0_pos_e = env.pursuers[0].aircraft.position_ned
            p1_pos_e = env.pursuers[1].aircraft.position_ned
            t_pos_e = env.targets[0].aircraft.position_ned
            los0 = t_pos_e - p0_pos_e
            los1 = t_pos_e - p1_pos_e
            norm0 = np.linalg.norm(los0)
            norm1 = np.linalg.norm(los1)
            if norm0 > 1 and norm1 > 1:
                cos_pincer = np.dot(los0, los1) / (norm0 * norm1)
                cos_pincer = np.clip(cos_pincer, -1.0, 1.0)
                pincer_angles.append(float(np.degrees(np.arccos(cos_pincer))))

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
            step += 1
            if step >= 500:
                break

        results["total_reward"].append(sum(ep_rewards) if ep_rewards else 0.0)
        results["ep_length"].append(step)
        results["min_d0"].append(min_d0_ep)
        results["min_d1"].append(min_d1_ep)
        results["max_pincer"].append(max(pincer_angles) if pincer_angles else 0.0)
        results["p0_reward"].append(sum(r for r in ep_rewards[::2]) if ep_rewards else 0.0)
        results["p1_reward"].append(sum(r for r in ep_rewards[1::2]) if ep_rewards else 0.0)

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1:3d}/{n_episodes}] "
                  f"rew={results['total_reward'][-1]:.0f}  "
                  f"d0_min={min_d0_ep:.0f}m  d1_min={min_d1_ep:.0f}m  "
                  f"d_init_diff={abs(d0_init-d1_init):.0f}m  "
                  f"asym={results['init_asymmetric'][-1]}")

    algo.stop()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(results: dict, save_path: str):
    """Generate scatter + marginal histogram figure."""
    dist_diff = np.array(results["init_dist_diff"])
    bearing_err = np.array(results["init_bearing_err"])
    rewards = np.array(results["total_reward"])
    asymmetric = np.array(results["init_asymmetric"], dtype=bool)
    min_d0 = np.array(results["min_d0"])
    min_d1 = np.array(results["min_d1"])

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Scatter: Initial Distance Difference vs Reward ──────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sc1 = ax1.scatter(dist_diff[~asymmetric], rewards[~asymmetric],
                      c=rewards[~asymmetric], cmap="RdYlGn", s=40,
                      alpha=0.7, edgecolors="black", linewidth=0.3,
                      marker="o", label="Symmetric start")
    sc2 = ax1.scatter(dist_diff[asymmetric], rewards[asymmetric],
                      c=rewards[asymmetric], cmap="RdYlGn", s=50,
                      alpha=0.8, edgecolors="black", linewidth=0.5,
                      marker="s", label="Asymmetric start")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Initial |d0 − d1| (m)")
    ax1.set_ylabel("Episode Total Reward")
    ax1.set_title(f"Initial State Sensitivity: Distance Asymmetry vs Reward\n"
                  f"(N={len(rewards)} episodes)")
    ax1.legend(fontsize=8, framealpha=0.8)
    cbar1 = fig.colorbar(sc2, ax=ax1, label="Reward")
    ax1.grid(True, alpha=0.2)

    # ── Scatter: Initial Bearing Error vs Reward ────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    sc3 = ax2.scatter(bearing_err[~asymmetric], rewards[~asymmetric],
                      c=rewards[~asymmetric], cmap="RdYlGn", s=40,
                      alpha=0.7, edgecolors="black", linewidth=0.3,
                      marker="o", label="Symmetric start")
    sc4 = ax2.scatter(bearing_err[asymmetric], rewards[asymmetric],
                      c=rewards[asymmetric], cmap="RdYlGn", s=50,
                      alpha=0.8, edgecolors="black", linewidth=0.5,
                      marker="s", label="Asymmetric start")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Initial Mean Bearing Error (°)")
    ax2.set_ylabel("Episode Total Reward")
    ax2.set_title(f"Initial State Sensitivity: Bearing Error vs Reward")
    ax2.legend(fontsize=8, framealpha=0.8)
    cbar2 = fig.colorbar(sc4, ax=ax2, label="Reward")
    ax2.grid(True, alpha=0.2)

    # ── Heatmap: Distance Diff × Bearing Error → Mean Reward ────────────
    ax3 = fig.add_subplot(gs[2, :2])
    # Bin the data
    x_bins = np.linspace(0, max(dist_diff.max(), 3000), 12)
    y_bins = np.linspace(0, max(bearing_err.max(), 180), 10)
    heatmap = np.full((len(y_bins)-1, len(x_bins)-1), np.nan)
    counts = np.zeros_like(heatmap)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = ((dist_diff >= x_bins[i]) & (dist_diff < x_bins[i+1]) &
                    (bearing_err >= y_bins[j]) & (bearing_err < y_bins[j+1]))
            if mask.sum() >= 2:  # need at least 2 episodes per bin
                heatmap[j, i] = rewards[mask].mean()
                counts[j, i] = mask.sum()

    im = ax3.imshow(heatmap, origin="lower", aspect="auto", cmap="RdYlGn",
                    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
    ax3.set_xlabel("Initial |d0 − d1| (m)")
    ax3.set_ylabel("Initial Mean Bearing Error (°)")
    ax3.set_title(f"Reward Heatmap: Distance Asymmetry × Bearing Error\n"
                  f"(binned mean, min {int(counts[counts>0].min())} eps/cell)")

    # Annotate bins with values
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            if not np.isnan(heatmap[j, i]):
                ax3.text((x_bins[i]+x_bins[i+1])/2, (y_bins[j]+y_bins[j+1])/2,
                         f"{heatmap[j,i]:.0f}", ha="center", va="center",
                         fontsize=7, fontweight="bold",
                         color="white" if abs(heatmap[j,i]) > 5000 else "black")

    cbar3 = fig.colorbar(im, ax=ax3, label="Mean Reward")

    # ── Histogram: Reward distribution ──────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.hist(rewards, bins=20, color="#377eb8", alpha=0.7, edgecolor="white")
    ax4.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax4.axvline(x=rewards.mean(), color="darkblue", linestyle="-", linewidth=1,
                label=f"Mean={rewards.mean():.0f}")
    ax4.set_xlabel("Total Reward")
    ax4.set_ylabel("Count")
    ax4.set_title(f"Reward Distribution\n"
                  f"Pos={int((rewards > 0).sum())}/{len(rewards)} "
                  f"({100*(rewards>0).mean():.0f}%)")
    ax4.legend(fontsize=7)

    # ── Bar: Asymmetric vs Symmetric mean reward ────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    sym_mean = rewards[~asymmetric].mean() if (~asymmetric).any() else 0
    asym_mean = rewards[asymmetric].mean() if asymmetric.any() else 0
    sym_std = rewards[~asymmetric].std() if (~asymmetric).any() else 0
    asym_std = rewards[asymmetric].std() if asymmetric.any() else 0
    bars = ax5.bar(["Symmetric", "Asymmetric"], [sym_mean, asym_mean],
                   yerr=[sym_std, asym_std], capsize=5,
                   color=["#4daf4a", "#ff7f00"], alpha=0.8, edgecolor="white")
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax5.set_ylabel("Mean Episode Reward")
    ax5.set_title(f"Reward by Start Type\n"
                  f"Sym n={(~asymmetric).sum()}, Asym n={asymmetric.sum()}")
    # Annotate
    for bar, val in zip(bars, [sym_mean, asym_mean]):
        ax5.text(bar.get_x() + bar.get_width()/2, val + (50 if val > 0 else -50),
                 f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

    # ── Scatter: Final min distance vs reward ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    min_both = np.minimum(min_d0, min_d1)
    ax6.scatter(min_both, rewards, c=rewards, cmap="RdYlGn", s=30,
               alpha=0.6, edgecolors="black", linewidth=0.3)
    ax6.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax6.axvline(x=200, color="blue", linestyle=":", linewidth=0.8, alpha=0.5,
                label="OR success threshold (200m)")
    ax6.set_xlabel("Min(min_d0, min_d1) per episode (m)")
    ax6.set_ylabel("Total Reward")
    ax6.set_title("Final Outcome: Closest Approach vs Reward")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.2)

    # ── Super-title ─────────────────────────────────────────────────────
    pos_rate = 100 * (rewards > 0).mean()
    fig.suptitle(
        f"Initial State Sensitivity Analysis — OR-gate MAPPO\n"
        f"N={len(rewards)} episodes | {pos_rate:.0f}% positive | "
        f"Mean reward={rewards.mean():.0f} ± {rewards.std():.0f} | "
        f"Range=[{rewards.min():.0f}, {rewards.max():.0f}]",
        fontsize=10, y=1.01,
    )

    fig.savefig(save_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n[OK] Sensitivity analysis figure saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initial State Sensitivity Analysis for OR-gate MAPPO")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to RLlib checkpoint directory")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--output-npz", type=str,
                       default="data/viz/initial_state_sensitivity.npz",
                       help="Output path for raw data")
    parser.add_argument("--output-fig", type=str,
                       default="results/viz/fig_init_state_sensitivity.pdf",
                       help="Output path for figure")
    args = parser.parse_args()

    print("=" * 60)
    print("Initial State Sensitivity Analysis")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes:   {args.episodes}")
    print(f"Seed:       {args.seed}")
    print("=" * 60)

    results = run_sensitivity_analysis(args.ckpt, args.episodes, args.seed)

    # Save raw data
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, **{k: np.array(v) for k, v in results.items()})
    print(f"[OK] Raw data saved: {args.output_npz}")

    # Generate figure
    os.makedirs(os.path.dirname(args.output_fig) or ".", exist_ok=True)
    plot_sensitivity(results, args.output_fig)

    # Summary stats
    rewards = np.array(results["total_reward"])
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"  Episodes:          {len(rewards)}")
    print(f"  Mean reward:       {rewards.mean():.0f} ± {rewards.std():.0f}")
    print(f"  Range:             [{rewards.min():.0f}, {rewards.max():.0f}]")
    print(f"  Positive rate:     {100*(rewards>0).mean():.0f}%")
    print(f"  Asymmetric starts: {results['init_asymmetric'].count(1)}/{len(rewards)}")
    print(f"  Mean init dist diff: {np.mean(results['init_dist_diff']):.0f}m")
    print(f"  Mean init bearing err: {np.mean(results['init_bearing_err']):.1f}°")
    print(f"  Correlation(diff, rew): {np.corrcoef(results['init_dist_diff'], rewards)[0,1]:.3f}")
    print(f"  Correlation(bearing, rew): {np.corrcoef(results['init_bearing_err'], rewards)[0,1]:.3f}")
