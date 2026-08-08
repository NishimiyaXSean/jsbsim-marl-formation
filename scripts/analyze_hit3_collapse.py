"""Per-case geometry-collapse analysis for 3-hit timeout episodes.

For each 3-hit timeout seed:
  * full trace: range / ATA / closure / speeds / cmd_hdg_err /
    heading & speed actions / fire-mask sub-conditions
  * align on t_hit3 (3rd hit) and find the FIRST persistent failure among
    {ATA>15, closure>=0, range out of DLZ} (first cause, not final mode)
  * auto-type:
      Type 1 near-pass overshoot : range dips near-min then ATA flips >90
      Type 2 turn-late divergence : ATA first persistent bad
      Type 3 speed-separation     : closure first persistent bad
      Type 4 range-drift          : range first persistent bad (no near-pass)
  * aligned panels [t_hit3-100, t_hit3+500] for representative seeds

Usage:
  python scripts/analyze_hit3_collapse.py --seeds 20 --outdir results/ctrl_viz/hit3_collapse
"""

from __future__ import annotations

import argparse
import json
import math
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
from src.environment.singlecombat_shoot_task import (
    MIN_ATTACK_DISTANCE, MAX_ATTACK_ANGLE, MIN_ATTACK_INTERVAL)
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action

MAX_STEPS = 1500
PERSIST = 20


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def geometry(env):
    ps = env.pursuers[0]
    tgt = env.targets[0]
    p_pos = ps.aircraft.position_ned
    t_pos = tgt.aircraft.position_ned
    los = t_pos - p_pos
    dist = float(np.linalg.norm(los))
    los_dir = los / max(dist, 1e-6)
    p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
    t_fwd = compute_forward_vector(tgt.aircraft.rpy_rad)
    ata = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
    aa = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    dyn_max = 3000.0 + 5000.0 * (aa / 180.0)
    return {"dist": dist, "ata": ata, "closure": closure, "dyn_max": dyn_max}


def run_trace(model, device, seed):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    t = {"range": [], "ata": [], "closure": [], "dyn_max": [],
         "p_spd": [], "t_spd": [], "hdg_err": [], "hdg_act": [],
         "spd_act": [], "mask": [], "launch": [], "hit": [],
         "ata_ok": [], "range_ok": [], "closure_ok": [], "cooldown_ok": []}
    last_fire = None
    reason = "timeout"
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        g = geometry(env)
        ps, ts = env.pursuers[0], env.targets[0]
        t["range"].append(g["dist"])
        t["ata"].append(g["ata"])
        t["closure"].append(g["closure"])
        t["dyn_max"].append(g["dyn_max"])
        t["p_spd"].append(float(np.linalg.norm(ps.aircraft.velocity_ned)))
        t["t_spd"].append(float(np.linalg.norm(ts.aircraft.velocity_ned)))
        t["hdg_err"].append(float(obs["p0"][21]) * 180.0)
        t["mask"].append(float(mask[10]))
        t["ata_ok"].append(1.0 if g["ata"] < MAX_ATTACK_ANGLE else 0.0)
        t["range_ok"].append(
            1.0 if MIN_ATTACK_DISTANCE < g["dist"] < g["dyn_max"] else 0.0)
        t["closure_ok"].append(1.0 if g["closure"] < 0.0 else 0.0)
        cd = (last_fire is None
              or env._step_counter - last_fire >= MIN_ATTACK_INTERVAL)
        t["cooldown_ok"].append(1.0 if cd else 0.0)
        t["launch"].append(0.0)
        t["hit"].append(0.0)
        act = policy_action(model, obs["p0"], device)
        t["hdg_act"].append(int(act[1]))
        t["spd_act"].append(int(act[0]))
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            t["launch"][-1] = 1.0
            last_fire = env._step_counter
        if env.task._hit_this_step.get("p0", 0) > 0:
            t["hit"][-1] = 1.0
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    t["reason"] = reason
    t["steps"] = len(t["range"])
    return {k: (np.array(v) if isinstance(v, list) else v)
            for k, v in t.items()}


def first_persistent_bad(ok, start, persist=PERSIST):
    """First index >= start where ok==0 for `persist` consecutive steps."""
    n = len(ok)
    for s in range(start, n - persist + 1):
        if int(ok[s:s + persist].sum()) == 0:
            return s
    return None


def classify(t):
    hit_idx = np.flatnonzero(t["hit"] == 1.0)
    if len(hit_idx) < 3:
        return None
    t_hit3 = int(hit_idx[2])
    rng = t["range"][t_hit3 + 1:t_hit3 + 82]
    near_min = float(rng.min()) if len(rng) else None
    ata_win = t["ata"][t_hit3:t_hit3 + 121]
    ata_flip = None
    for i, a in enumerate(ata_win):
        if a > 90.0:
            ata_flip = t_hit3 + i
            break
    ata_first = first_persistent_bad(t["ata_ok"], t_hit3 + 1)
    closure_first = first_persistent_bad(t["closure_ok"], t_hit3 + 1)
    range_first = first_persistent_bad(t["range_ok"], t_hit3 + 1)

    near_pass = near_min is not None and near_min <= 1800.0
    if near_pass and ata_flip is not None:
        ftype = "T1_near_pass_overshoot"
    else:
        order = sorted([x for x in
                        (("ata", ata_first), ("closure", closure_first),
                         ("range", range_first)) if x[1] is not None],
                       key=lambda x: x[1])
        if not order:
            ftype = "T0_no_persistent_bad"
        elif order[0][0] == "ata":
            ftype = "T2_turn_late_divergence"
        elif order[0][0] == "closure":
            ftype = "T3_speed_separation"
        else:
            ftype = "T4_range_drift"
    return {
        "t_hit3": t_hit3,
        "near_min": near_min,
        "ata_flip": ata_flip,
        "ata_first": ata_first,
        "closure_first": closure_first,
        "range_first": range_first,
        "delta_ata": (ata_first - t_hit3) if ata_first is not None else None,
        "delta_closure": (closure_first - t_hit3)
        if closure_first is not None else None,
        "delta_range": (range_first - t_hit3)
        if range_first is not None else None,
        "type": ftype,
    }


def plot_aligned(t, seed, meta, outdir):
    h3 = meta["t_hit3"]
    s0, s1 = max(0, h3 - 100), min(t["steps"], h3 + 500)
    x = np.arange(s0, s1) * 0.2
    sl = slice(s0, s1)
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    launch_idx = np.flatnonzero(t["launch"][sl] == 1.0) + s0
    hit_idx = np.flatnonzero(t["hit"][sl] == 1.0) + s0
    h3_x = h3 * 0.2

    def mark(ax):
        ax.axvline(h3_x, color="red", ls="--", lw=1.2, label="t_hit3")
        ax.plot(x[launch_idx - s0], t["range"][launch_idx] / 1000.0, "^",
                color="black", ms=8) if launch_idx.size else None
        ax.plot(x[hit_idx - s0], t["range"][hit_idx] / 1000.0, "*",
                color="gold", ms=12, markeredgecolor="black") if hit_idx.size else None

    ax = axes[0, 0]
    ax.plot(x, t["range"][sl] / 1000.0, color="#1f77b4", lw=1.3)
    ax.axhspan(1.5, 8.0, color="green", alpha=0.08)
    ax.axhline(15.0, color="red", ls=":", lw=1)
    mark(ax)
    ax.set_title("range (km)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, t["ata"][sl], color="#d62728", lw=1.3)
    ax.axhline(15.0, color="green", ls=":", lw=1)
    ax.axhline(90.0, color="red", ls=":", lw=1)
    mark(ax)
    ax.set_title("ATA (deg)")

    ax = axes[1, 0]
    ax.plot(x, t["closure"][sl], color="#9467bd", lw=1.3)
    ax.axhline(0.0, color="black", lw=0.8)
    mark(ax)
    ax.set_title("closure (m/s)")

    ax = axes[1, 1]
    ax.plot(x, t["p_spd"][sl], color="#1f77b4", lw=1.2, label="pursuer")
    ax.plot(x, t["t_spd"][sl], color="#d62728", lw=1.2, ls="--", label="target")
    mark(ax)
    ax.set_title("speed (m/s)")
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(x, t["hdg_err"][sl], color="#2ca02c", lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8)
    mark(ax)
    ax.set_title("cmd_hdg_err (deg)")

    ax = axes[2, 1]
    ax.plot(x, t["hdg_act"][sl], color="#8c564b", lw=1.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title("heading action (0..4 = -10..+10 deg)")
    mark(ax)

    ax = axes[3, 0]
    ax.plot(x, t["spd_act"][sl], color="#7f7f7f", lw=1.2)
    ax.set_yticks([0, 1, 2])
    ax.set_title("speed action (0..2 = -20/0/+20)")
    mark(ax)

    ax = axes[3, 1]
    ax.plot(x, t["mask"][sl], color="green", lw=1.8, label="mask open")
    ax.plot(x, t["ata_ok"][sl] + 0.6, color="#d62728", lw=1.0, label="ATA ok")
    ax.plot(x, t["range_ok"][sl] + 1.2, color="#1f77b4", lw=1.0, label="range ok")
    ax.plot(x, t["closure_ok"][sl] + 1.8, color="#9467bd", lw=1.0, label="closure ok")
    ax.plot(x, t["cooldown_ok"][sl] + 2.4, color="#ff7f0e", lw=1.0,
            label="cooldown ok")
    ax.set_ylim(-0.2, 3.8)
    ax.set_title("fire sub-conditions (offset stacked)")
    ax.legend(fontsize=7)
    mark(ax)

    fig.suptitle(f"seed {seed} — {meta['type']} — hit3={h3} "
                 f"({h3*0.2:.0f}s) Δata={meta['delta_ata']} "
                 f"Δclosure={meta['delta_closure']} Δrange={meta['delta_range']}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(outdir, exist_ok=True)
    pth = os.path.join(outdir, f"hit3_seed{seed}_{meta['type']}.png")
    fig.savefig(pth, dpi=130, facecolor="white")
    plt.close(fig)
    return pth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20,
                        help="number of 3-hit timeout episodes to collect")
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--outdir", default="results/ctrl_viz/hit3_collapse")
    parser.add_argument("--out-json", default="results/shoot_eval/hit3_collapse.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    collected = []
    seed = 0
    while len(collected) < args.seeds and seed < 1200:
        t = run_trace(model, device, seed)
        if t["reason"] == "timeout" and int(t["hit"].sum()) == 3:
            meta = classify(t)
            collected.append({"seed": seed, "trace": t, "meta": meta})
        seed += 1
    print(f"[hit3] collected {len(collected)} 3-hit timeout episodes "
          f"(scanned {seed} seeds)")

    by_type = {}
    for c in collected:
        by_type.setdefault(c["meta"]["type"], []).append(c["seed"])
    print("type distribution:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<28s} n={len(v):3d}  seeds={v[:10]}")

    summary = {
        "n": len(collected),
        "by_type": {k: v for k, v in by_type.items()},
        "seeds": [c["seed"] for c in collected],
        "details": [{"seed": c["seed"], **{k: c["meta"][k] for k in
                     ("type", "t_hit3", "near_min", "ata_flip",
                      "ata_first", "closure_first", "range_first",
                      "delta_ata", "delta_closure", "delta_range")}}
                    for c in collected],
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # one aligned panel per type (first seed of each type)
    plotted = set()
    for c in collected:
        tp = c["meta"]["type"]
        if tp in plotted:
            continue
        plotted.add(tp)
        plot_aligned(c["trace"], c["seed"], c["meta"], args.outdir)
    print(f"[hit3] saved aligned panels to {args.outdir}/")
    print(f"[hit3] saved json to {args.out_json}")


if __name__ == "__main__":
    main()

