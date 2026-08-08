"""Acceptance: speed-distilled student vs B2 rule override (paired seeds).

Fresh seeds only (never used in search/dev). Per seed, runs:
  * student  : the trained speed-distilled policy (policy_action)
  * teacher  : B2 memoryless_decel rule override (frozen)
and reports paired kill/lost/bad/3-hit/launches/TTK/min-range + speed-action
agreement in the B2 region and outside it. Also a fresh 13-cell sweep for the
student.

Usage:
  python scripts/accept_speed_distilled.py --id-seeds 400 --d2-seeds 200
"""

from __future__ import annotations

import argparse
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
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch
from scripts.rule_override_eval import (
    run_episode as teacher_run, state_features)
from scripts.eval_scenario_matrix import CELLS

MAX_STEPS = 1500
RULE = (4000.0, 90.0, 100.0, 10.0)


def student_run(model, device, seed, cfg):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True, **cfg}))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    launches = hits = 0
    bad = 0
    fire3 = None
    t4win = None
    min_range3 = float("inf")
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        if fire3 is not None and step > fire3:
            ps, ts = env.pursuers[0], env.targets[0]
            min_range3 = min(min_range3,
                             float(np.linalg.norm(ps.aircraft.position_ned
                                                  - ts.aircraft.position_ned)))
            if mask[10] == 1.0 and t4win is None:
                t4win = step
        act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches == 3:
                fire3 = step
            if classify_launch(launch_geometry(env)) == "bad":
                bad += 1
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {"kill": reason == "target_killed", "lost": reason == "lost_target",
            "bad": bad, "3hit": reason == "timeout" and hits == 3,
            "launches": launches, "hits": hits, "steps": step + 1,
            "t4win_gap": (t4win - fire3) if t4win is not None and fire3 is not None
            else None, "min_range3": min_range3}


def speed_agreement(model, device, cfg, seeds, n_steps=20000):
    """Sample speed actions from student vs B2 rule on fresh rollouts."""
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True, **cfg}))
    agree_b2 = agree_non = n_b2 = n_non = 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        for step in range(MAX_STEPS):
            f = state_features(env)
            act = policy_action(model, obs["p0"], device)
            strict = (f["range"] < RULE[0] and f["delta_speed"] > RULE[1]
                      and f["closure"] < -RULE[2] and f["ata"] < RULE[3])
            wide = (f["range"] < 1.2 * RULE[0] and f["delta_speed"] > 0.7 * RULE[1]
                    and f["closure"] < -0.7 * RULE[2] and f["ata"] < 1.5 * RULE[3])
            trig = strict or (wide and f["p_spd"] > 245.0)
            teacher_spd = 0 if (trig and f["p_spd"] > 240.0) else 1
            if trig:
                n_b2 += 1
                agree_b2 += int(act[0] == teacher_spd)
            else:
                n_non += 1
                agree_non += int(act[0] == teacher_spd)
            obs, rews, terms, truncs, info = env.step({"p0": act})
            if terms.get("__all__") or truncs.get("__all__"):
                break
        if n_b2 + n_non >= n_steps:
            break
    env.close()
    return {"b2_agreement": agree_b2 / max(n_b2, 1),
            "non_b2_agreement": agree_non / max(n_non, 1),
            "n_b2": n_b2, "n_non": n_non}


def agg(recs, key):
    return sum(1 for r in recs if r[key]) / max(len(recs), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/expert/shoot_bc_speed_distilled.pth")
    parser.add_argument("--id-seeds", type=int, default=400)
    parser.add_argument("--d2-seeds", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20000)
    parser.add_argument("--out", default="results/shoot_eval/accept_speed_distilled.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    out = {"student": {}, "teacher": {}, "agreement": {}, "cells": {}}
    for name, cfg, seeds in (
            ("id", {}, list(range(args.base_seed, args.base_seed + args.id_seeds))),
            ("d2", {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0},
             list(range(args.base_seed + 10000,
                        args.base_seed + 10000 + args.d2_seeds)))):
        s_recs = [student_run(model, device, s, cfg) for s in seeds]
        t_recs = [teacher_run(model, device, s, cfg, RULE, "memoryless_decel")
                  for s in seeds]
        out["student"][name] = {
            "kill": agg(s_recs, "kill"), "lost": agg(s_recs, "lost"),
            "bad": sum(r["bad"] for r in s_recs),
            "3hit": sum(r["3hit"] for r in s_recs),
            "ttk_p50": float(np.median([r["steps"] * 0.2
                                        for r in s_recs if r["kill"]]))
            if any(r["kill"] for r in s_recs) else None,
            "4th": sum(1 for r in s_recs if r["t4win_gap"] is not None)
            / max(len(s_recs), 1),
            "min_range3_median": float(np.median(
                [r["min_range3"] for r in s_recs]))}
        out["teacher"][name] = {
            "kill": agg(t_recs, "kill"), "lost": agg(t_recs, "lost"),
            "bad": sum(r["bad"] for r in t_recs),
            "3hit": sum(r["3hit_timeout"] for r in t_recs)}
        out["agreement"][name] = speed_agreement(model, device, cfg, seeds[:60])
        print(f"[accept] {name}: student kill={out['student'][name]['kill']*100:.1f}% "
              f"lost={out['student'][name]['lost']:.2f} bad={out['student'][name]['bad']} "
              f"3hit={out['student'][name]['3hit']} | "
              f"teacher kill={out['teacher'][name]['kill']*100:.1f}% "
              f"agr(b2/non)={out['agreement'][name]['b2_agreement']*100:.0f}/"
              f"{out['agreement'][name]['non_b2_agreement']*100:.0f}%")

    # fresh 13-cell sweep for the student (15 seeds per cell)
    for label, cfg in CELLS:
        recs = [student_run(model, device, args.base_seed + 20000 + s, cfg)
                for s in range(15)]
        out["cells"][label] = {
            "kill": agg(recs, "kill"), "lost": agg(recs, "lost"),
            "bad": sum(r["bad"] for r in recs)}
    print("[accept] cells:", {k: f"{v['kill']*100:.0f}%" for k, v in
                              out["cells"].items()})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

