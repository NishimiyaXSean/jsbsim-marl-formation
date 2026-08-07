"""Failure-episode attribution for the distilled ASAP policy.

For every non-kill episode, records:
  * hits (0/1/2/3) and reason (mostly timeout)
  * last-hit -> timeout gap
  * total fire_allowed steps and DISTINCT legal windows
  * whether a 4th legal window ever appeared
  * if not: which condition blocked it (range close / range far / ATA /
    closure / cooldown-timing) by scanning the post-last-window phase

Goal: check the hypothesis that most failures are "3-hit timeout + missing
4th window", and whether the missing window is a geometry problem (range/ATA/
closure) or a timing problem (cooldown / too-brief window).

Usage:
  python scripts/analyze_fail_episodes.py --id-seeds 500 --d2-seeds 200
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
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action
from src.environment.singlecombat_shoot_task import (
    MIN_ATTACK_DISTANCE, MAX_ATTACK_ANGLE, MIN_ATTACK_INTERVAL)

MAX_STEPS = 1500


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
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


def run_episode(model, device, seed, scene_cfg):
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(scene_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    launches, hits, hit_steps, launch_steps = [], [], [], []
    windows = []
    win_start = None
    mask_open_total = 0
    post_window = []
    reason = "timeout"
    last_fire = None
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        open_now = mask[10] == 1.0
        if open_now:
            mask_open_total += 1
            if win_start is None:
                win_start = step
        else:
            if win_start is not None:
                windows.append((win_start, step - 1))
                win_start = None
        if win_start is None and len(windows) > 0:
            post_window.append((step, geometry(env)))
        act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches.append(step)
            last_fire = step
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits.append(step)
            hit_steps.append(step)
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    if win_start is not None:
        windows.append((win_start, step))
    env.close()
    n_windows = len(windows)
    last_window_end = windows[-1][1] if windows else -1
    rec = {
        "seed": seed, "reason": reason, "kill": reason == "target_killed",
        "hits": len(hits), "launches": len(launches),
        "steps": step + 1,
        "hit_steps": hit_steps, "launch_steps": launches,
        "mask_open_total": mask_open_total, "n_windows": n_windows,
        "windows": windows, "last_window_end": last_window_end,
        "last_fire": last_fire,
        "gap_last_hit": (step - hit_steps[-1]) if hit_steps else None,
        "post_window": post_window,
    }
    return rec


def attribute_missing_window(rec):
    """Classify why the 4th (or Nth) window never appeared, for non-kill eps."""
    if rec["kill"] or rec["n_windows"] >= 4:
        return None
    if rec["launches"] >= 4:
        return "launched4_no_4th_hit"
    pw = rec["post_window"]
    if not pw:
        return "episode_ended_in_window" if rec["windows"] else "no_window_at_all"
    counts = {"range_close": 0, "range_far": 0, "ata": 0, "closure": 0,
              "static_ok": 0}
    # dominant failing condition AFTER the cooldown has cleared
    post_cd = {"range_close": 0, "range_far": 0, "ata": 0, "closure": 0}
    last_fire = rec["last_fire"]
    static_ok_outside_cooldown = 0
    for (st, g) in pw:
        range_ok = MIN_ATTACK_DISTANCE < g["dist"] < g["dyn_max"]
        ata_ok = g["ata"] < MAX_ATTACK_ANGLE
        closure_ok = g["closure"] < 0.0
        cd_clear = last_fire is None or st - last_fire >= MIN_ATTACK_INTERVAL
        if range_ok and ata_ok and closure_ok:
            counts["static_ok"] += 1
            if cd_clear:
                static_ok_outside_cooldown += 1
            continue
        if not range_ok:
            if g["dist"] <= MIN_ATTACK_DISTANCE:
                counts["range_close"] += 1
                if cd_clear:
                    post_cd["range_close"] += 1
            else:
                counts["range_far"] += 1
                if cd_clear:
                    post_cd["range_far"] += 1
        if not ata_ok:
            counts["ata"] += 1
            if cd_clear:
                post_cd["ata"] += 1
        if not closure_ok:
            counts["closure"] += 1
            if cd_clear:
                post_cd["closure"] += 1
    if static_ok_outside_cooldown > 0:
        return "static_ok_but_window_missing"
    if counts["static_ok"] > 0:
        return "cooldown_timing"
    total = sum(v for k, v in counts.items() if k != "static_ok")
    if total == 0:
        return "unknown"
    best = max(("range_close", "range_far", "ata", "closure"),
               key=lambda k: counts[k])
    best_cd = max(("range_close", "range_far", "ata", "closure"),
                  key=lambda k: post_cd[k]) if sum(post_cd.values()) else None
    return best + "|post_cd_" + (best_cd if best_cd else "none")


def summarize(records):
    nonkill = [r for r in records if not r["kill"]]
    n = len(records)
    nk = len(nonkill)
    hits_hist = {h: sum(1 for r in nonkill if r["hits"] == h) for h in range(4)}
    attrs = [attribute_missing_window(r) for r in nonkill]
    attr_hist = {k: attrs.count(k) for k in sorted(set(attrs))}
    gaps = [r["gap_last_hit"] for r in nonkill if r["gap_last_hit"] is not None]
    nw = [r["n_windows"] for r in nonkill]
    four = sum(1 for r in nonkill if r["n_windows"] >= 4)
    return {
        "episodes": n, "kill_rate": (n - nk) / max(n, 1),
        "nonkill": nk,
        "nonkill_hits_hist": hits_hist,
        "nonkill_reasons": {k: sum(1 for r in nonkill if r["reason"] == k)
                            for k in sorted({r["reason"] for r in nonkill})},
        "missing_4th_window_rate": (nk - four) / max(nk, 1),
        "4th_window_present_rate": four / max(nk, 1),
        "windows_dist": {str(k): nw.count(k) for k in sorted(set(nw))},
        "attr_hist": attr_hist,
        "gap_last_hit_median": float(np.median(gaps)) if gaps else None,
        "gap_last_hit_mean": float(np.mean(gaps)) if gaps else None,
        "gap_last_hit_p75": float(np.percentile(gaps, 75)) if gaps else None,
    }


def plot(agg_id, agg_d2, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, agg) in zip(axes, (("ID", agg_id), ("dist2_3k", agg_d2))):
        hist = agg["nonkill_hits_hist"]
        keys = [0, 1, 2, 3]
        ax.bar([str(k) for k in keys], [hist[k] for k in keys],
               color="#1f77b4", alpha=0.85)
        ax.set_title(f"{name}: non-kill hits hist (n={agg['nonkill']})")
        ax.set_xlabel("hits in non-kill episodes")
        ax.set_ylabel("episodes")
        for k in keys:
            ax.text(str(k), hist[k] + max(1, agg["nonkill"] * 0.01),
                    str(hist[k]), ha="center", fontsize=9)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    pth = os.path.join(outdir, "fail_attribution_hits.png")
    fig.savefig(pth, dpi=140, facecolor="white")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, (name, agg) in zip(axes2, (("ID", agg_id), ("dist2_3k", agg_d2))):
        attr = agg["attr_hist"]
        keys = sorted(attr)
        ax.bar(keys, [attr[k] for k in keys], color="#d62728", alpha=0.85)
        ax.set_title(f"{name}: missing-window blocker attribution")
        ax.set_ylabel("episodes")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        for k in keys:
            ax.text(k, attr[k] + 0.5, str(attr[k]), ha="center", fontsize=9)
    fig2.tight_layout()
    pth2 = os.path.join(outdir, "fail_attribution_blockers.png")
    fig2.savefig(pth2, dpi=140, facecolor="white")
    plt.close(fig2)
    print(f"  saved {pth}")
    print(f"  saved {pth2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-seeds", type=int, default=500)
    parser.add_argument("--d2-seeds", type=int, default=200)
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--outdir", default="results/shoot_eval")
    parser.add_argument("--vizdir", default="results/ctrl_viz")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    id_cfg = {}
    d2_cfg = {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0}
    id_recs = [run_episode(model, device, s, id_cfg)
               for s in range(args.id_seeds)]
    print(f"[fail-attr] ID {len(id_recs)} episodes done")
    d2_recs = [run_episode(model, device, s, d2_cfg)
               for s in range(args.d2_seeds)]
    print(f"[fail-attr] dist2_3k {len(d2_recs)} episodes done")

    agg_id = summarize(id_recs)
    agg_d2 = summarize(d2_recs)
    out = {"id": agg_id, "dist2_3k": agg_d2}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "fail_attribution.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    plot(agg_id, agg_d2, args.vizdir)

    for name, agg in (("ID", agg_id), ("dist2_3k", agg_d2)):
        print("=" * 60)
        print(f"[{name}] kill={agg['kill_rate']*100:.1f}%  "
              f"non-kill={agg['nonkill']}")
        print(f"  non-kill hits hist: {agg['nonkill_hits_hist']}")
        print(f"  4th-window present={agg['4th_window_present_rate']*100:.1f}%  "
              f"missing={agg['missing_4th_window_rate']*100:.1f}%  "
              f"windows dist={agg['windows_dist']}")
        print(f"  gap last-hit->end: median={agg['gap_last_hit_median']} "
              f"mean={agg['gap_last_hit_mean']:.0f} p75={agg['gap_last_hit_p75']} steps")
        print(f"  blockers: {agg['attr_hist']}")


if __name__ == "__main__":
    main()



