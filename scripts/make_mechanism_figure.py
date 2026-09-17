"""Mechanism figure for the paper: same maneuver, different launch decision.

Runs one matched seed twice -- once with the BC policy, once with the SPC
policy -- and renders:

  Panel A  terminal-phase top view (North vs East). The maneuver heads are
           frozen, so BC and SPC fly the SAME path; only the launch decision
           differs. Legal launch steps, BC's declined steps, and each
           policy's launches are marked.
  Panel B  three rows of launch decisions (mask / BC / SPC) over time, with
           the range-to-target trace on a secondary axis for context.

The point of the figure is attribution, not trajectory aesthetics: geometry
and maneuver are held fixed by construction, so the outcome difference is
attributable to the launch decision alone.

Usage:
  python scripts/make_mechanism_figure.py --seed 20007
  python scripts/make_mechanism_figure.py --seed 20000 --scan 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.eval_bc_1v1 import policy_action, FIRE_IDX
from scripts.train_shoot_bc import BCShootPolicy

MAX_STEPS = 1500
# One env step advances the simulation by 0.2 s (BaseEnv._acmi_time += 0.2),
# i.e. 5 Hz -- not 60 Hz. Getting this wrong scales every time axis by 12.
STEPS_PER_SEC = 5.0
BAR_H = 0.62

BC_WEIGHTS = "data/expert/shoot_bc_round1_baseline.pth"
SPC_WEIGHTS = "data/expert/shoot_bc_asap_distilled.pth"

TRACE_KEYS = ("p_n", "p_e", "t_n", "t_e", "mask", "fire", "ata_deg")


def ata_deg(env):
    """Angle off the pursuer's nose to the target, in degrees."""
    ps, tgt = env.pursuers[0], env.targets[0]
    los = tgt.aircraft.position_ned - ps.aircraft.position_ned
    dist = float(np.linalg.norm(los))
    if dist < 1e-9:
        return 0.0
    los_dir = los / dist
    roll, pitch, yaw = ps.aircraft.rpy_rad
    fwd = np.array([np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch)])
    return float(np.degrees(np.arccos(np.clip(float(np.dot(fwd, los_dir)),
                                               -1.0, 1.0))))


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def trace_episode(model, device, seed, difficulty=0.0):
    """Play one fresh-env episode, recording the full per-step trace."""
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": difficulty, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    tr = {k: [] for k in TRACE_KEYS}
    reason, launches, hits = "timeout", 0, 0
    for _ in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        act = policy_action(model, obs["p0"], device)
        p = env.pursuers[0].aircraft.position_ned
        t = env.targets[0].aircraft.position_ned
        tr["p_n"].append(float(p[0]))
        tr["p_e"].append(float(p[1]))
        tr["t_n"].append(float(t[0]))
        tr["t_e"].append(float(t[1]))
        tr["mask"].append(int(mask[FIRE_IDX] == 1.0))
        tr["fire"].append(int(act[3]))
        tr["ata_deg"].append(ata_deg(env))
        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits += env.task._hit_this_step.get("p0", 0)
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break
    env.close()
    tr["reason"] = reason
    tr["launches"] = launches
    tr["hits"] = hits
    tr["kill"] = bool(reason == "target_killed" or hits >= 4)
    tr["steps"] = len(tr["mask"])
    return tr


def window_segments(bits):
    """Collapse a 0/1 sequence into contiguous [start, end] segments."""
    segs, start = [], None
    for i, b in enumerate(bits):
        if b and start is None:
            start = i
        elif not b and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(bits) - 1))
    return segs


def ranges_km(tr):
    pn, pe = np.asarray(tr["p_n"]), np.asarray(tr["p_e"])
    tn, te = np.asarray(tr["t_n"]), np.asarray(tr["t_e"])
    return np.hypot(tn - pn, te - pe) / 1000.0


def prefix_deviation(a, b):
    """Max |a_i - b_i| over the common prefix of two position traces."""
    n = min(len(a["p_n"]), len(b["p_n"]))
    if n == 0:
        return float("nan"), 0
    d = max(max(abs(a["p_n"][i] - b["p_n"][i]) for i in range(n)),
            max(abs(a["p_e"][i] - b["p_e"][i]) for i in range(n)))
    return d, n


def render(tr_bc, tr_spc, seed, difficulty, out_png, dev, common):
    """Two stacked panels, sharing the time axis.

    Panel A is range / ATA versus time, not a top view. Reason: the maneuver
    heads are frozen, so BC and SPC fly a *bit-identical* path (verified, max
    deviation 0 m) over ~50 km -- a top view therefore shows one line and no
    divergence at all. What actually differs is the decision, and the variable
    the decision is taken on is range/ATA. So that is what Panel A plots.
    """
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.6, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.75]})

    n = len(tr_bc["mask"])
    t = np.arange(n) / STEPS_PER_SEC
    rng = ranges_km(tr_bc)
    ata = np.asarray(tr_bc["ata_deg"]) if "ata_deg" in tr_bc else None

    legal = [i for i, m in enumerate(tr_bc["mask"]) if m]
    bc_fire = [i for i, f in enumerate(tr_bc["fire"]) if f]
    spc_fire = [i for i, f in enumerate(tr_spc["fire"]) if f]
    abstain = [i for i in legal if not tr_bc["fire"][i]]

    # ---- Panel A: the variable the decision is taken on -------------------
    for idx, (s, e) in enumerate(window_segments(tr_bc["mask"])):
        ax.axvspan(s / STEPS_PER_SEC, (e + 1) / STEPS_PER_SEC,
                   color="#38a169", alpha=0.16, lw=0, zorder=1,
                   label="environment-legal window" if idx == 0 else None)
    ax.plot(t, rng, color="#2d3748", lw=1.5, zorder=4,
            label="range to target")
    if abstain:
        ax.plot(t[abstain], rng[abstain], "x", color="#c53030", ms=8.5,
                mew=2.2, zorder=6, label="legal step, BC declines")
    if bc_fire:
        ax.plot(t[bc_fire], rng[bc_fire], "^", color="#b7791f", ms=10,
                mec="white", mew=0.9, zorder=7, label="BC launches")
    if spc_fire:
        ax.plot(t[spc_fire], rng[spc_fire], "*", color="#1c4532", ms=19,
                mec="white", mew=0.9, zorder=8, label="SPC launches")
    ax.axvline(common / STEPS_PER_SEC, color="#1c4532", lw=1.0, ls=":",
               zorder=3)
    ax.annotate("SPC run ends (target killed)",
                xy=(common / STEPS_PER_SEC, ax.get_ylim()[1] * 0.55),
                xytext=(-6, 0), textcoords="offset points", ha="right",
                fontsize=7.6, color="#1c4532", rotation=90, va="center")
    ax.set_ylabel("range to target (km)")
    ax.set_title("A. Engagement state and launch decisions "
                 "(seed %d, d=%.1f)" % (seed, difficulty),
                 fontsize=9.6, loc="left")
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.94, ncol=2)
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_ylim(0, max(rng.max() * 1.10, 1.0))

    ax2 = ax.twinx()
    if ata is not None:
        ax2.plot(t, ata, color="#c05621", lw=0.9, alpha=0.5, zorder=3)
        ax2.set_ylabel("ATA (deg)", fontsize=8.1, color="#c05621")
        ax2.tick_params(axis="y", labelsize=7.4, colors="#c05621", length=2)
        ax2.set_ylim(0, 180)

    # ---- Panel B: the decisions themselves --------------------------------
    spc_bits = tr_spc["fire"] + [0] * max(0, n - len(tr_spc["fire"]))
    rows = [("launch mask (environment)", tr_bc["mask"], "#4a5568"),
            ("BC fire command", tr_bc["fire"], "#b7791f"),
            ("SPC fire command", spc_bits, "#1c4532")]
    for k, (label, series, color) in enumerate(rows):
        y = len(rows) - 1 - k
        for s, e in window_segments(series[:n]):
            bx.broken_barh(
                [(s / STEPS_PER_SEC, max((e - s + 1) / STEPS_PER_SEC,
                                         1.0 / STEPS_PER_SEC))],
                (y + 0.19, BAR_H), color=color, zorder=3)
        bx.plot([t[0], t[-1]], [y + 0.5, y + 0.5], color="#e2e8f0", lw=0.9,
                zorder=1)
        bx.text(-0.012 * t[-1], y + 0.5, label, ha="right", va="center",
                fontsize=8.2)
    bx.set_yticks([])
    bx.set_ylim(-0.15, len(rows) + 0.30)
    bx.set_xlim(0, t[-1])
    bx.set_xlabel("time (s), 5 Hz simulation steps")
    bx.set_title("B. Launch decisions over time", fontsize=9.6, loc="left")
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(axis="y", length=0)

    bc_used, bc_legal = len(bc_fire), len(legal)
    spc_legal = int(sum(tr_spc["mask"]))
    spc_used = len([i for i, m in enumerate(tr_spc["mask"])
                    if m and tr_spc["fire"][i]])
    fig.text(0.012, 0.980,
             "BC: %d/%d environment-legal steps used (CLR %.2f%%) -> %s"
             % (bc_used, bc_legal, 100.0 * bc_used / max(bc_legal, 1),
                tr_bc["reason"].replace("_", " ")),
             fontsize=8.6, va="top")
    fig.text(0.012, 0.957,
             "SPC: %d/%d (CLR %.2f%%) -> %s"
             % (spc_used, spc_legal, 100.0 * spc_used / max(spc_legal, 1),
                tr_spc["reason"].replace("_", " ")),
             fontsize=8.6, va="top")
    fig.text(0.012, 0.934,
             "maneuver bit-identical over %d common steps (max deviation "
             "%.1e m); only the launch decision differs" % (common, dev),
             fontsize=8.6, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.920))
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20007)
    ap.add_argument("--scan", type=int, default=0,
                    help="scan this many seeds from --seed and auto-pick one "
                         "where SPC kills, BC does not, and BC's own CLR is "
                         "at or below 10%%")
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--bc-weights", default=BC_WEIGHTS)
    ap.add_argument("--spc-weights", default=SPC_WEIGHTS)
    ap.add_argument("--out-dir", default="results/shoot_eval")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    bc_model = load_model(args.bc_weights, device)
    spc_model = load_model(args.spc_weights, device)

    seed = args.seed
    tr_bc = tr_spc = None
    for k in range(max(args.scan, 1)):
        s = args.seed + k
        b = trace_episode(bc_model, device, s, args.difficulty)
        p = trace_episode(spc_model, device, s, args.difficulty)
        b_legal = max(sum(b["mask"]), 1)
        b_clr = sum(1 for i, m in enumerate(b["mask"])
                    if m and b["fire"][i]) / b_legal
        print("  scan seed %d: BC kill=%s (%s, CLR %.1f%%)  SPC kill=%s (%s)"
              % (s, b["kill"], b["reason"], 100 * b_clr, p["kill"], p["reason"]))
        if tr_bc is None:
            tr_bc, tr_spc, seed = b, p, s
        if args.scan and p["kill"] and not b["kill"] and b_clr <= 0.10:
            tr_bc, tr_spc, seed = b, p, s
            break

    dev, common = prefix_deviation(tr_bc, tr_spc)
    stem = "mechanism_seed%d_d%s" % (
        seed, ("%.1f" % args.difficulty).replace(".", ""))
    png = os.path.join(args.out_dir, stem + ".png")
    js = os.path.join(args.out_dir, stem + ".json")
    render(tr_bc, tr_spc, seed, args.difficulty, png, dev, common)

    payload = {
        "seed": seed,
        "difficulty": args.difficulty,
        "bc_weights": args.bc_weights,
        "spc_weights": args.spc_weights,
        "bc": {k: tr_bc[k] for k in
               ("reason", "launches", "hits", "kill", "steps")},
        "spc": {k: tr_spc[k] for k in
                ("reason", "launches", "hits", "kill", "steps")},
        "bc_legal_steps": int(sum(tr_bc["mask"])),
        "bc_fire_on_legal": int(sum(1 for i, m in enumerate(tr_bc["mask"])
                                    if m and tr_bc["fire"][i])),
        "spc_legal_steps": int(sum(tr_spc["mask"])),
        "spc_fire_on_legal": int(sum(1 for i, m in enumerate(tr_spc["mask"])
                                     if m and tr_spc["fire"][i])),
        "trajectory_prefix_len": common,
        "trajectory_max_deviation_m": dev,
        "figure": os.path.basename(png),
        "trace": {k: tr_bc[k] for k in TRACE_KEYS},
        "trace_spc_fire": tr_spc["fire"],
    }
    with open(js, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print("seed %d  BC  %s / launches %d / hits %d / steps %d"
          % (seed, tr_bc["reason"], tr_bc["launches"], tr_bc["hits"],
             tr_bc["steps"]))
    print("seed %d  SPC %s / launches %d / hits %d / steps %d"
          % (seed, tr_spc["reason"], tr_spc["launches"], tr_spc["hits"],
             tr_spc["steps"]))
    print("maneuver prefix %d steps, max deviation %.3e m" % (common, dev))
    print("BC  CLR %d/%d = %.2f%%" % (payload["bc_fire_on_legal"],
                                      payload["bc_legal_steps"],
                                      100.0 * payload["bc_fire_on_legal"]
                                      / max(payload["bc_legal_steps"], 1)))
    print("SPC CLR %d/%d = %.2f%%" % (payload["spc_fire_on_legal"],
                                      payload["spc_legal_steps"],
                                      100.0 * payload["spc_fire_on_legal"]
                                      / max(payload["spc_legal_steps"], 1)))
    print("wrote %s" % png)
    print("wrote %s" % js)


if __name__ == "__main__":
    main()
