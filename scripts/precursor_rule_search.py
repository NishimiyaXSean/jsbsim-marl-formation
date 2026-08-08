"""Offline precursor extraction + geometry-rule grid search.

Step 1 of the anti-overshoot rule pipeline. Extracts, from 3-hit timeout
(T1) failure episodes and matched 4-hit kill success episodes, the state at
fire3-120/100/80/60/40 steps (the precursor band), then grid-searches a
simple rule:

    R < R_thr AND delta_speed > DV_thr AND closure < -C_thr AND ATA < A_thr

Thresholds are drawn from data percentiles (no hand-picked values). A
candidate passes the first filter if:
  * failure-seed trigger recall >= 90% (>=1 trigger in the band per seed)
  * success-seed false-trigger <= 10% (same relative band + early window)

The final ranking happens on CLOSED-LOOP override (rule_override_eval.py).

Usage:
  python scripts/precursor_rule_search.py --fail-seeds 20 --success-seeds 80
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action

MAX_STEPS = 1500
OFFSETS = [120, 100, 80, 60, 40]


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])


def state_features(env):
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
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned,
                           los_dir))
    rel = tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned
    cross2d = los[0] * rel[1] - los[1] * rel[0]
    los_rate = abs(cross2d) / max(los[0] ** 2 + los[1] ** 2, 1e-6)
    p_spd = float(np.linalg.norm(ps.aircraft.velocity_ned))
    t_spd = float(np.linalg.norm(tgt.aircraft.velocity_ned))
    return {
        "range": dist, "delta_speed": p_spd - t_spd, "ata": ata,
        "closure": closure, "los_rate": los_rate,
        "p_spd": p_spd, "t_spd": t_spd,
    }


def run_trace(model, device, seed):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True}))
    obs, _ = env.reset(seed=seed)
    feats, events = [], {}
    launches = hits = 0
    reason = "timeout"
    for step in range(MAX_STEPS):
        feats.append(state_features(env))
        act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches <= 3:
                events[f"fire{launches}"] = step
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
            if hits <= 3:
                events[f"hit{hits}"] = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {"feats": feats, "events": events, "reason": reason,
            "hits": hits, "launches": launches}


def precursor_states(rec, offsets):
    """Features at fire3-offset for each offset; None if out of range."""
    f3 = rec["events"].get("fire3")
    if f3 is None:
        return {}
    out = {}
    for off in offsets:
        i = f3 - off
        if 0 <= i < len(rec["feats"]):
            out[off] = rec["feats"][i]
    return out


def early_states(rec, offsets):
    """Much-earlier states (before fire3-150) that should NOT trigger."""
    f3 = rec["events"].get("fire3")
    if f3 is None:
        return []
    out = []
    for i in range(30, max(31, f3 - 150), 20):
        if i < len(rec["feats"]):
            out.append(rec["feats"][i])
    return out


def triggers(rule, states):
    n = 0
    for s in states:
        if (s["range"] < rule["R"] and s["delta_speed"] > rule["DV"]
                and s["closure"] < -rule["C"] and s["ata"] < rule["A"]):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-seeds", type=int, default=20)
    parser.add_argument("--success-seeds", type=int, default=80)
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="results/shoot_eval/precursor_rule_search.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    # positive: T1 failure episodes with all offsets present
    pos = []
    seed = 0
    while len(pos) < args.fail_seeds and seed < 1200:
        r = run_trace(model, device, seed)
        if r["reason"] == "timeout" and r["hits"] == 3 \
                and len(precursor_states(r, OFFSETS)) == len(OFFSETS):
            pos.append({"seed": seed, "rec": r})
        seed += 1
    # negatives A: success episodes (same relative band)
    neg_a = []
    seed = 0
    while len(neg_a) < args.success_seeds and seed < 1200:
        r = run_trace(model, device, seed)
        if r["reason"] == "target_killed" and r["hits"] == 4 \
                and len(precursor_states(r, OFFSETS)) == len(OFFSETS):
            neg_a.append({"seed": seed, "rec": r})
        seed += 1
    print(f"[precursor] pos(fail)={len(pos)} negA(success)={len(neg_a)}")

    # aggregate feature stats for percentile thresholds
    all_pos = [s for p in pos for s in precursor_states(p["rec"], OFFSETS).values()]
    feat_keys = ["range", "delta_speed", "ata"]
    pc = {k: [float(np.percentile([s[k] for s in all_pos], q))
              for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
          for k in feat_keys}
    closing = [-s["closure"] for s in all_pos if s["closure"] < 0.0]
    pc["closing_mag"] = [float(np.percentile(closing, q))
                         for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    print("[precursor] positive feature percentiles (10..90):")
    for k in ["range", "delta_speed", "closing_mag", "ata"]:
        print(f"  {k:<12s} {[round(x, 1) for x in pc[k]]}")

    # grid: R_thr / DV_thr / C_thr / A_thr from percentile candidates
    def cand(key, idx):
        return pc[key][idx]

    grid = []
    for ri, ci, di, ai in itertools.product(range(2, 8), range(2, 8),
                                            range(2, 8), range(2, 8)):
        rule = {"R": cand("range", ri), "DV": cand("delta_speed", di),
                "C": cand("closing_mag", ci), "A": cand("ata", ai)}
        # per-episode trigger rates
        pos_trig = sum(1 for p in pos
                       if triggers(rule, precursor_states(p["rec"], OFFSETS).values()) > 0)
        neg_trig = sum(1 for n in neg_a
                       if triggers(rule, precursor_states(n["rec"], OFFSETS).values()) > 0)
        pos_early = sum(1 for p in pos
                        if triggers(rule, early_states(p["rec"], OFFSETS)) > 0)
        recall = pos_trig / max(len(pos), 1)
        fpr_succ = neg_trig / max(len(neg_a), 1)
        fpr_early = pos_early / max(len(pos), 1)
        passed = recall >= 0.90 and fpr_succ <= 0.10 and fpr_early <= 0.15
        grid.append({"rule": rule, "recall": recall, "fpr_succ": fpr_succ,
                     "fpr_early": fpr_early, "passed": passed})
    grid.sort(key=lambda x: (-x["recall"], x["fpr_succ"], x["fpr_early"]))
    n_pass = sum(1 for g in grid if g["passed"])
    print(f"[precursor] candidates: total={len(grid)} passing filter={n_pass}")
    for g in grid[:12]:
        print(f"  R={g['rule']['R']:.0f} DV={g['rule']['DV']:.1f} "
              f"C={g['rule']['C']:.1f} A={g['rule']['A']:.1f} "
              f"recall={g['recall']*100:.0f}% fpr_succ={g['fpr_succ']*100:.0f}% "
              f"fpr_early={g['fpr_early']*100:.0f}% "
              f"{'PASS' if g['passed'] else ''}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "percentiles": pc,
            "pos_seeds": [p["seed"] for p in pos],
            "negA_seeds": [n["seed"] for n in neg_a],
            "candidates": grid[:20],
        }, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

