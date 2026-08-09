"""Final version-freeze holdout: student vs B2 teacher, paired seeds.

Pre-registered acceptance (2026-08-09, commit fb1ac46, weights sha
a9417686...) -- NO tuning after this run:

Hard gates (student, every group):
  * lost = 0
  * bad launch = 0
  * no systematic T1 near-pass overshoot (T1 rate <= 1% in ID/dist2_3k,
    no re-emergence in stress)
Performance gates:
  * ID kill >= 98%
  * dist2_3k kill >= 96%
  * student vs teacher: ID not worse by >1pp, dist2_3k not worse by >2pp
  * stress: no lost

Scale: ID 1000, dist2_3k 400, 13 cells (key 100 / other 50), L0-L4 stress
100 per disturbance condition. All paired student-vs-teacher on the same
seeds. Fresh seed ranges: ID 40000+, d2 41000+, cells 42000+, stress 44000+.

Usage:
  python scripts/final_holdout.py
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
from src.environment.singlecombat_shoot_task import MIN_ATTACK_INTERVAL
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch
from scripts.rule_override_eval import state_features
from scripts.stress_eval import DISTURBANCES, LEVELS
from scripts.eval_scenario_matrix import CELLS

MAX_STEPS = 1500
RULE = (4000.0, 90.0, 100.0, 10.0)
KEY_CELLS = {"dist2_3k", "alt_diff300", "target_evasive"}
STRESS_LIST = (["none"] + LEVELS["L1"] + LEVELS["L2"]
               + LEVELS["L3"] + LEVELS["L4"])


def teacher_speed(f):
    strict = (f["range"] < RULE[0] and f["delta_speed"] > RULE[1]
              and f["closure"] < -RULE[2] and f["ata"] < RULE[3])
    wide = (f["range"] < 1.2 * RULE[0] and f["delta_speed"] > 0.7 * RULE[1]
            and f["closure"] < -0.7 * RULE[2] and f["ata"] < 1.5 * RULE[3])
    trig = strict or (wide and f["p_spd"] > 245.0)
    return trig, (0 if f["p_spd"] > 240.0 else 1)


def run_episode(model, device, seed, cfg, mode, dist=None):
    """mode: 'student' or 'teacher'; dist: disturbance dict or None."""
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True, **cfg}))
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    base_hdg = float(env.task._target_base_hdg)
    exec_hist = []
    if dist is not None:
        t0 = int(rng.integers(int(MAX_STEPS * 0.10), int(MAX_STEPS * 0.35)))
        dist = dict(dist)
        dist["_t0"], dist["_t1"] = t0, t0 + dist.get("k", 1)
    reason = "timeout"
    launches = hits = 0
    bad = 0
    fire3 = fire4 = None
    t4win = None
    min_r3 = float("inf")
    ata_gt90 = 0
    win_steps = 0
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        f = state_features(env)
        obs_in = obs["p0"]
        if dist is not None and dist["type"] in ("obs_noise", "combo") \
                and dist["_t0"] <= step < dist["_t1"]:
            obs_in = obs_in.copy()
            obs_in[:30] += rng.normal(0.0, dist["std"], 30)
        act = policy_action(model, obs_in, device)
        if mode == "teacher":
            trig, spd = teacher_speed(f)
            act[0] = spd if trig else act[0]
        act[3] = 1 if mask[10] == 1.0 else 0
        if dist is not None and dist["_t0"] <= step < dist["_t1"]:
            typ = dist["type"]
            if typ in ("hdg", "combo"):
                act[1] = 4 - act[1]
            if typ == "delay" or typ == "combo":
                k = dist["k"]
                if len(exec_hist) >= k:
                    act = exec_hist[-k].copy()
        exec_hist.append(act.copy())
        if dist is not None and dist["type"] == "target_turn" \
                and dist["_t0"] <= step < dist["_t1"]:
            env.task._target_base_hdg = (base_hdg + dist["deg"]) % 360.0
        elif dist is not None and dist["type"] == "combo" \
                and dist["_t0"] <= step < dist["_t0"] + dist.get("turn_k", 0):
            env.task._target_base_hdg = (base_hdg + dist["turn_deg"]) % 360.0
        else:
            env.task._target_base_hdg = base_hdg
        if fire3 is not None and step > fire3:
            win_steps += 1
            min_r3 = min(min_r3, f["range"])
            if f["ata"] > 90.0:
                ata_gt90 += 1
            if mask[10] == 1.0 and t4win is None:
                t4win = step
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches == 3:
                fire3 = step
            if launches == 4:
                fire4 = step
            if classify_launch(launch_geometry(env)) == "bad":
                bad += 1
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {
        "kill": reason == "target_killed",
        "lost": reason == "lost_target",
        "bad": bad,
        "3hit": reason == "timeout" and hits == 3,
        "launches": launches,
        "4launch": launches >= 4,
        "4th_window": t4win is not None,
        "fire3_fire4": (fire4 - fire3) if fire4 is not None and fire3 is not None
        else None,
        "min_range3": min_r3 if fire3 is not None else None,
        "T1": (min_r3 < 1500.0) if fire3 is not None else False,
        "ata_gt90_ratio": ata_gt90 / max(win_steps, 1),
        "steps": step + 1,
    }


def wilson_upper(n_fail, n, z=1.96):
    if n == 0:
        return 1.0
    p = n_fail / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return min(1.0, c + h)


def bootstrap_ci(diffs, n_boot=10000, seed=123):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean()
                  for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg_pair(s_recs, t_recs):
    n = len(s_recs)
    d_kill = np.array([s["kill"] - t["kill"] for s, t in zip(s_recs, t_recs)])
    d_3hit = np.array([s["3hit"] - t["3hit"] for s, t in zip(s_recs, t_recs)])
    d_ttk = np.array([(s["steps"] - t["steps"]) * 0.2
                      for s, t in zip(s_recs, t_recs)
                      if s["kill"] and t["kill"]])
    s_ttk = [r["steps"] * 0.2 for r in s_recs if r["kill"]]
    t_ttk = [r["steps"] * 0.2 for r in t_recs if r["kill"]]
    g = {
        "n": n,
        "kill": sum(r["kill"] for r in s_recs) / n,
        "t_kill": sum(r["kill"] for r in t_recs) / n,
        "lost": sum(r["lost"] for r in s_recs),
        "t_lost": sum(r["lost"] for r in t_recs),
        "lost_wilson_95_upper": wilson_upper(
            sum(r["lost"] for r in s_recs), n),
        "bad": sum(r["bad"] for r in s_recs),
        "3hit": sum(r["3hit"] for r in s_recs),
        "t_3hit": sum(r["3hit"] for r in t_recs),
        "T1": sum(r["T1"] for r in s_recs) / n,
        "t_T1": sum(r["T1"] for r in t_recs) / n,
        "4th_window": sum(r["4th_window"] for r in s_recs) / n,
        "4launch": sum(r["4launch"] for r in s_recs) / n,
        "fire3_fire4_median": float(np.median([r["fire3_fire4"]
                                               for r in s_recs
                                               if r["fire3_fire4"] is not None]))
        if any(r["fire3_fire4"] is not None for r in s_recs) else None,
        "min_range3_median": float(np.median([r["min_range3"]
                                              for r in s_recs
                                              if r["min_range3"] is not None])),
        "ttk_p50": float(np.median(s_ttk)) if s_ttk else None,
        "ttk_p90": float(np.percentile(s_ttk, 90)) if s_ttk else None,
        "diff_kill_ci": bootstrap_ci(d_kill),
        "diff_3hit_ci": bootstrap_ci(d_3hit),
        "diff_ttk_ci": bootstrap_ci(d_ttk) if len(d_ttk) else None,
    }
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="data/expert/shoot_bc_speed_distilled_v2.pth")
    parser.add_argument("--id-seeds", type=int, default=1000)
    parser.add_argument("--d2-seeds", type=int, default=400)
    parser.add_argument("--cell-seeds-key", type=int, default=100)
    parser.add_argument("--cell-seeds-other", type=int, default=50)
    parser.add_argument("--stress-seeds", type=int, default=100)
    parser.add_argument("--out", default="results/shoot_eval/final_holdout.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    student = BCShootPolicy().to(device)
    student.load_state_dict(torch.load(args.student, map_location="cpu")["state_dict"])
    student.eval()

    groups = {}
    id_seeds = list(range(40000, 40000 + args.id_seeds))
    d2_seeds = list(range(41000, 41000 + args.d2_seeds))
    groups["id"] = (id_seeds, {})
    groups["dist2_3k"] = (d2_seeds,
                          {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0})
    for i, (label, cfg) in enumerate(CELLS):
        n = args.cell_seeds_key if label in KEY_CELLS else args.cell_seeds_other
        groups[f"cell_{label}"] = (
            list(range(42000 + i * 100, 42000 + i * 100 + n)), cfg)
    for j, name in enumerate(STRESS_LIST):
        dist = None if name == "none" else dict(DISTURBANCES[name])
        if dist is not None:
            dist["name"] = name
        groups[f"stress_{name}"] = (
            list(range(44000 + j * 100, 44000 + j * 100 + args.stress_seeds)),
            {}, dist)

    out = {"manifest": {
        "student_sha256": "a94176861f71e521c21183463c8edc34c8bffff4664c76a9acea4871dd8960d3",
        "commit": "fb1ac4659a12a91da7f742fc1f68fcfe2e96b4a2",
        "rule": {"R": 4000, "DV": 90, "C": 100, "A": 10},
    }, "groups": {}}
    for name, (seeds, cfg, *rest) in groups.items():
        dist = rest[0] if rest else None
        s_recs = [run_episode(student, device, s, cfg, "student", dist)
                  for s in seeds]
        t_recs = [run_episode(student, device, s, cfg, "teacher", dist)
                  for s in seeds]
        g = agg_pair(s_recs, t_recs)
        out["groups"][name] = g
        print(f"[{name:<20s}] n={g['n']:4d} kill={g['kill']*100:5.1f}% "
              f"(t={g['t_kill']*100:5.1f}%) lost={g['lost']} bad={g['bad']} "
              f"3hit={g['3hit']} T1={g['T1']*100:4.1f}% "
              f"minR={g['min_range3_median']:.0f}m "
              f"d_kill_ci=[{g['diff_kill_ci'][0]*100:+.1f},{g['diff_kill_ci'][1]*100:+.1f}]pp")

    # pre-registered verdicts
    v = {}
    idg, d2g = out["groups"]["id"], out["groups"]["dist2_3k"]
    lost_total = sum(g["lost"] for g in out["groups"].values())
    bad_total = sum(g["bad"] for g in out["groups"].values())
    v["lost_zero"] = lost_total == 0
    v["bad_zero"] = bad_total == 0
    v["id_kill_ge98"] = idg["kill"] >= 0.98
    v["d2_kill_ge96"] = d2g["kill"] >= 0.96
    v["id_not_worse_1pp"] = (idg["kill"] - idg["t_kill"]) >= -0.01
    v["d2_not_worse_2pp"] = (d2g["kill"] - d2g["t_kill"]) >= -0.02
    v["T1_low"] = idg["T1"] <= 0.01 and d2g["T1"] <= 0.01
    stress_lost = sum(g["lost"] for k, g in out["groups"].items()
                      if k.startswith("stress_"))
    v["stress_no_lost"] = stress_lost == 0
    v["overall"] = all(v.values())
    out["verdicts"] = v
    print("=" * 60)
    for k, val in v.items():
        print(f"  {k:<20s} {'PASS' if val else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if v['overall'] else 'FAIL'}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()


