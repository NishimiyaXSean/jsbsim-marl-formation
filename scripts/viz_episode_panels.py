"""Per-episode time-series panels for the distilled ASAP policy.

Panels: range (+WEZ band / lost boundary), ATA (+15/10 deg thresholds),
closure (+0 line), altitude (pursuer vs target), speed, fire-mask-open +
launch/hit markers. Optionally --reward-dist N renders the P2-reward
distribution over N episodes (damage +250/HP, kill +1000, bad -100).

Usage:
  python scripts/viz_episode_panels.py --classic \
      --outdir results/ctrl_viz/panels_distilled_asap
  python scripts/viz_episode_panels.py --reward-dist 100
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch

MAX_STEPS = 1500
MAX_DIST = 15000.0


def run_trace(model, device, seed, cell_cfg):
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(cell_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    rows = {"range": [], "ata": [], "closure": [], "p_alt": [], "t_alt": [],
            "p_spd": [], "t_spd": [], "hdg_err": [], "mask_open": [],
            "launch": [], "hit": [], "reward": []}
    launches = hits = 0
    reason = "timeout"
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        g = launch_geometry(env)
        ps, ts = env.pursuers[0], env.targets[0]
        rows["range"].append(g["range_m"])
        rows["ata"].append(g["ata_deg"])
        rows["closure"].append(g["closure_mps"])
        rows["p_alt"].append(ps.aircraft.state["alt_m"])
        rows["t_alt"].append(ts.aircraft.state["alt_m"])
        rows["p_spd"].append(np.linalg.norm(ps.aircraft.velocity_ned))
        rows["t_spd"].append(np.linalg.norm(ts.aircraft.velocity_ned))
        rows["hdg_err"].append(float(obs["p0"][21]) * 180.0)
        rows["mask_open"].append(1.0 if mask[10] == 1.0 else 0.0)
        rows["launch"].append(0.0)
        rows["hit"].append(0.0)
        act = policy_action(model, obs["p0"], device)
        hits_before = ts.hits_taken
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            rows["launch"][-1] = 1.0
        if ts.hits_taken > hits_before:
            hits += 1
            rows["hit"][-1] = 1.0
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    out = {k: np.array(v) for k, v in rows.items()}
    out["reason"] = reason
    out["launches"] = launches
    out["hits"] = hits
    out["steps"] = len(rows["range"])
    return out


def plot_episode(t, seed, label, outdir):
    n = t["steps"]
    x = np.arange(n) * 0.2
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    launch_idx = np.flatnonzero(t["launch"] == 1.0)
    hit_idx = np.flatnonzero(t["hit"] == 1.0)
    wez_idx = np.flatnonzero(t["mask_open"] == 1.0)
    wez_first = wez_idx[0] if len(wez_idx) else None

    ax = axes[0, 0]
    ax.plot(x, t["range"] / 1000.0, color="#1f77b4", lw=1.4)
    ax.axhspan(1.5, 8.0, color="green", alpha=0.08)
    ax.axhline(15.0, color="red", ls="--", lw=1)
    ax.plot(x[launch_idx], t["range"][launch_idx] / 1000.0, "^", color="black",
            ms=9, label="launch")
    ax.plot(x[hit_idx], t["range"][hit_idx] / 1000.0, "*", color="gold",
            ms=14, markeredgecolor="black", label="hit")
    ax.set_title(f"range (km) | WEZ band 1.5-8, lost 15 | seed {seed} {label}")
    ax.set_ylim(0, 18)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, t["ata"], color="#d62728", lw=1.4)
    ax.axhline(15.0, color="green", ls=":", lw=1)
    ax.axhline(10.0, color="blue", ls=":", lw=1)
    ax.plot(x[launch_idx], t["ata"][launch_idx], "^", color="black", ms=9)
    ax.set_title("ATA (deg), thresholds 15/10")

    ax = axes[1, 0]
    ax.plot(x, t["closure"], color="#9467bd", lw=1.4)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.plot(x[launch_idx], t["closure"][launch_idx], "^", color="black", ms=9)
    ax.set_title("closure (m/s), <0 = closing")

    ax = axes[1, 1]
    ax.plot(x, t["p_alt"], color="#1f77b4", lw=1.2, label="pursuer")
    ax.plot(x, t["t_alt"], color="#d62728", lw=1.2, ls="--", label="target")
    ax.set_title("altitude (m)")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(x, t["p_spd"], color="#1f77b4", lw=1.2, label="pursuer")
    ax.plot(x, t["t_spd"], color="#d62728", lw=1.2, ls="--", label="target")
    ax.set_title("speed (m/s)")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(x, t["mask_open"], color="green", lw=2.0, label="fire mask open")
    ax.plot(x, t["launch"], color="black", lw=1.0, alpha=0.7, label="launch")
    ax.plot(x, t["hit"], color="gold", lw=1.0, alpha=0.9, label="hit")
    ax.set_ylim(-0.1, 1.4)
    ax.set_title(f"fire window / launch / hit | reason={t['reason']} "
                 f"launches={t['launches']} hits={t['hits']} "
                 f"WEZ_first={wez_first}")
    ax.legend(fontsize=8)

    fig.suptitle(f"seed {seed} {label} — {t['reason']} — {n} steps "
                 f"({n*0.2:.0f}s)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(outdir, exist_ok=True)
    pth = os.path.join(outdir, f"episode_seed{seed}_{label}.png")
    fig.savefig(pth, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  saved {pth}")
    return pth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds")
    parser.add_argument("--classic", action="store_true",
                        help="auto-pick kill / dist2_3k-fire / timeout episodes")
    parser.add_argument("--reward-dist", type=int, default=0,
                        help="render P2-reward distribution over N episodes")
    parser.add_argument("--outdir", default="results/ctrl_viz/panels_distilled_asap")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    if args.reward_dist > 0:
        rewards = []
        for s in range(args.reward_dist):
            t = run_trace(model, device, s, {})
            r = 0.0
            for i in range(t["steps"]):
                if t["hit"][i]:
                    r += 250.0

            if t["reason"] == "target_killed":
                r += 1000.0
            rewards.append(r)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(rewards, bins=24, color="#1f77b4", alpha=0.85)
        ax.axvline(np.mean(rewards), color="red", ls="--",
                   label=f"mean {np.mean(rewards):.0f}")
        ax.set_title(f"P2-reward distribution (damage+kill), n={args.reward_dist}")
        ax.set_xlabel("episode reward")
        ax.legend()
        fig.tight_layout()
        pth = os.path.join(args.outdir, "eval_reward_dist_distilled_asap.png")
        os.makedirs(args.outdir, exist_ok=True)
        fig.savefig(pth, dpi=140, facecolor="white")
        plt.close(fig)
        print(f"reward dist mean={np.mean(rewards):.0f} kill_rate="
              f"{sum(1 for r in rewards if r >= 1000) / len(rewards) * 100:.1f}%")
        print(f"  saved {pth}")
        return

    if args.classic:
        id_cfg = {}
        d2_cfg = {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0}
        kill_seed = fire_seed = timeout_seed = None
        for s in range(80):
            t = run_trace(model, device, s, id_cfg)
            if kill_seed is None and t["reason"] == "target_killed" \
                    and t["launches"] >= 4:
                kill_seed = s
            if timeout_seed is None and t["reason"] == "timeout" \
                    and t["launches"] >= 2:
                timeout_seed = s
            if kill_seed is not None and timeout_seed is not None:
                break
        for s in range(80):
            t = run_trace(model, device, s, d2_cfg)
            if t["reason"] == "target_killed" and t["launches"] >= 4:
                fire_seed = s
                break
        jobs = []
        if kill_seed is not None:
            jobs.append((kill_seed, id_cfg, "kill_4hit"))
        if fire_seed is not None:
            jobs.append((fire_seed, d2_cfg, "dist2_3k_4launch"))
        if timeout_seed is not None:
            jobs.append((timeout_seed, id_cfg, "timeout"))
        print(f"classic seeds: kill={kill_seed} dist2_3k={fire_seed} "
              f"timeout={timeout_seed}")
        for s, cfg, label in jobs:
            t = run_trace(model, device, s, cfg)
            plot_episode(t, s, label, args.outdir)
        return

    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        t = run_trace(model, device, s, {})
        plot_episode(t, s, f"seed{s}", args.outdir)


if __name__ == "__main__":
    main()


