"""Pure-BC stress evaluation (NO expert takeover) — stress-DAgger triage.

Injects ONE disturbance type per run (L1/L2), or a combination (L3 holdout),
into BOTH the discrete expert and the frozen BC policy (paired seeds, same
initial geometry). No takeover: this measures each policy's TRUE failure
rate under perturbation, which decides where DAgger data is needed.

Disturbance types (mask is never polluted; obs noise excludes mask fields):
  hdg_k1/k3     : wrong (opposite) heading action for k steps
  spd_k1/k3     : wrong (opposite) speed action for k steps
  delay_k1/k3/k6: execute the policy action from k steps ago for k steps
  noise_small/med: Gaussian noise on obs[:30] for a window (mask excluded)
  target_turn30/45: target flies base+deg heading for k steps
  combo_l3      : hdg_k3 + delay_k3 + target_turn40 + noise0.04

Metrics: standard + max ATA, closure<0 streak, danger entries, self-recovery,
expert-BC heading divergence, launch quality / ammo.

Usage:
  python scripts/stress_eval.py --level L1 --seeds 100
  python scripts/stress_eval.py --disturbance hdg_k3 --seeds 80
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
from scripts.generate_shoot_rule_expert import (
    hdg_label, spd_label, fire_desired, FIRE_IDX)
from scripts.eval_bc_1v1 import policy_action
from scripts.train_shoot_bc import BCShootPolicy

MAX_STEPS = 1500
CMD_SPEED = 280.0

DISTURBANCES = {
    "hdg_k1": {"type": "hdg", "k": 1},
    "hdg_k3": {"type": "hdg", "k": 3},
    "spd_k1": {"type": "spd", "k": 1},
    "spd_k3": {"type": "spd", "k": 3},
    "delay_k1": {"type": "delay", "k": 1},
    "delay_k3": {"type": "delay", "k": 3},
    "delay_k6": {"type": "delay", "k": 6},
    "noise_small": {"type": "obs_noise", "std": 0.015, "k": 30},
    "noise_med": {"type": "obs_noise", "std": 0.030, "k": 60},
    "target_turn30": {"type": "target_turn", "k": 15, "deg": 30.0},
    "target_turn45": {"type": "target_turn", "k": 25, "deg": 45.0},
    "combo_l3": {"type": "combo", "k": 3, "std": 0.040, "turn_deg": 40.0,
                 "turn_k": 20, "noise_k": 40},
    "hdg_k10": {"type": "hdg", "k": 10},
    "hdg_k20": {"type": "hdg", "k": 20},
    "delay_k10": {"type": "delay", "k": 10},
    "target_turn90": {"type": "target_turn", "k": 40, "deg": 45.0},
    "combo_l4": {"type": "combo", "k": 10, "std": 0.050,
                  "turn_deg": 45.0, "turn_k": 30, "noise_k": 50},
}
LEVELS = {
    "L0": ["none"],
    "L1": ["hdg_k1", "delay_k1", "noise_small", "spd_k1"],
    "L2": ["hdg_k3", "delay_k3", "noise_med", "target_turn30"],
    "L3": ["combo_l3", "delay_k6", "target_turn45"],  # holdout
    "L4": ["hdg_k10", "hdg_k20", "delay_k10", "target_turn90", "combo_l4"],
}


def compute_forward_vector(rpy_rad):
    roll, pitch, yaw = rpy_rad
    return np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])


def launch_geometry(env):
    ps = env.pursuers[0]
    tgt = env.targets[0]
    p_pos = ps.aircraft.position_ned
    t_pos = tgt.aircraft.position_ned
    los = t_pos - p_pos
    dist = float(np.linalg.norm(los))
    los_dir = los / max(dist, 1e-6)
    p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
    t_fwd = compute_forward_vector(tgt.aircraft.rpy_rad)
    ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    return {"range_m": dist, "ata_deg": ata_deg, "closure_mps": closure}


def classify_launch(g):
    if g["closure_mps"] < 0 and g["ata_deg"] < 10.0 and 2000.0 < g["range_m"] < 4000.0:
        return "premium"
    if g["closure_mps"] < 0:
        return "good"
    return "bad"


def expert_action(env, obs):
    mask = env.task.get_action_mask(env, "p0")
    allowed = mask[FIRE_IDX] == 1.0
    desired = fire_desired(obs)
    fire = 1 if (allowed and desired) else 0
    return np.array([spd_label(obs, CMD_SPEED), hdg_label(obs), 0, fire],
                    dtype=np.int64)


def in_win(dc, step):
    return dc is not None and dc.get("_t0") is not None \
        and dc["_t0"] <= step < dc["_t1"]


def run_one_stressed(policy, model, device, seed, dist, difficulty=0.0,
                     inject_lo=0.10, inject_hi=0.35, max_steps=MAX_STEPS,
                     task_cfg=None, inject_mode="frac", inject_offset=10,
                     fire_mode="policy"):
    cfg = {"difficulty_level": difficulty, "obs_include_closure": True}
    if task_cfg:
        cfg.update(task_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    base_hdg = float(env.task._target_base_hdg)

    dist_cfg = None
    if dist is not None:
        dist_cfg = dict(dist)
        dist_cfg["_exec_hist"] = []
        if inject_mode == "wez":
            dist_cfg["_t0"] = None
            dist_cfg["_t1"] = None
        else:
            t_inject = int(rng.integers(int(max_steps * inject_lo),
                                        int(max_steps * inject_hi)))
            dist_cfg["_t0"] = t_inject
            dist_cfg["_t1"] = t_inject + dist["k"]

    reason = "timeout"
    wez_first = fire_first = None
    launches = hits = 0
    quality = {"bad": 0, "good": 0, "premium": 0}
    ata_hist, closure_hist, dist_hist = [], [], []
    hdg_seq = []
    delay_mismatch = 0
    danger = False
    ever_aligned = False
    danger_streak = 0
    danger_streak_max = 0
    danger_entered = False
    safe_streak = 0
    recovered = False

    for step in range(max_steps):
        raw_obs = obs["p0"]
        obs_in = raw_obs
        if dist_cfg is not None and dist_cfg["type"] in ("obs_noise", "combo") \
                and in_win(dist_cfg, step):
            obs_in = raw_obs.copy()
            obs_in[:30] += rng.normal(0.0, dist_cfg["std"], 30)

        if policy == "expert":
            new_act = expert_action(env, obs_in)
        else:
            new_act = policy_action(model, obs_in, device)
        act = new_act.copy()
        if fire_mode == "asap":
            mask = env.task.get_action_mask(env, "p0")
            act[3] = 1 if mask[FIRE_IDX] == 1.0 else 0
        perturbed = False
        if in_win(dist_cfg, step):
            typ = dist_cfg["type"]
            if typ in ("hdg", "combo"):
                act[1] = 4 - act[1]
            if typ == "spd":
                act[0] = 2 - act[0]
            if typ in ("delay", "combo"):
                k = dist_cfg["k"]
                if len(dist_cfg["_exec_hist"]) >= k:
                    act = dist_cfg["_exec_hist"][-k].copy()
            perturbed = True
        if perturbed and not np.array_equal(act, new_act):
            delay_mismatch += 1
        if dist_cfg is not None:
            dist_cfg["_exec_hist"].append(act)
        hdg_seq.append(int(act[1]))

        if dist_cfg is not None and dist_cfg["type"] == "target_turn" \
                and in_win(dist_cfg, step):
            env.task._target_base_hdg = (base_hdg + dist_cfg["deg"]) % 360.0
        elif dist_cfg is not None and dist_cfg["type"] == "combo" \
                and dist_cfg["_t0"] is not None \
                and dist_cfg["_t0"] <= step < dist_cfg["_t0"] + dist_cfg["turn_k"]:
            env.task._target_base_hdg = (base_hdg + dist_cfg["turn_deg"]) % 360.0
        else:
            env.task._target_base_hdg = base_hdg

        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits += env.task._hit_this_step.get("p0", 0)
        g = launch_geometry(env)
        ata_hist.append(g["ata_deg"])
        closure_hist.append(g["closure_mps"])
        dist_hist.append(g["range_m"])
        if wez_first is None and env.task._is_valid_launch_envelope(
                env.pursuers[0], env.targets[0]):
            wez_first = step
            if dist_cfg is not None and inject_mode == "wez" \
                    and dist_cfg["_t0"] is None:
                dist_cfg["_t0"] = step + inject_offset
                dist_cfg["_t1"] = dist_cfg["_t0"] + dist["k"]
        if env.task._has_launched_this_step.get("p0", False):
            if fire_first is None:
                fire_first = step
            quality[classify_launch(g)] += 1
            launches += 1

        # danger / recovery bookkeeping: only AFTER the injection window.
        # "danger" = approaching the task's lost boundary (MAX_DIST=15km):
        # dist > 12km sustained 3 steps. Recovery = back below 10km for 10
        # consecutive steps. Normal approach/overshoot cycles stay well below
        # 12km, so an unstressed baseline has danger ~0.
        post_inject = dist_cfg is None or (dist_cfg["_t0"] is not None
                                           and step >= dist_cfg["_t0"])
        in_danger = post_inject and g["range_m"] > 12000.0
        danger_streak = danger_streak + 1 if in_danger else 0
        if danger_streak >= 3:
            danger_entered = True
        danger_streak_max = max(danger_streak_max, danger_streak)
        if in_danger:
            danger = True
            safe_streak = 0
        else:
            if g["range_m"] < 10000.0:
                safe_streak += 1
                if danger and safe_streak >= 10:
                    recovered = True
                    danger = False
            else:
                safe_streak = 0

        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break

    env.close()
    closure_arr = np.asarray(closure_hist)
    neg_streak = 0
    neg_streak_max = 0
    for c in closure_arr:
        neg_streak = neg_streak + 1 if c < 0.0 else 0
        neg_streak_max = max(neg_streak_max, neg_streak)
    return {
        "seed": int(seed),
        "policy": policy,
        "disturbance": dist["name"] if dist is not None else "none",
        "reason": reason,
        "steps": step + 1,
        "kill": reason == "target_killed",
        "lost": reason == "lost_target",
        "launches": launches,
        "hits": hits,
        "first_fire": fire_first,
        "wez_first": wez_first,
        "quality": quality,
        "max_ata": float(np.max(ata_hist)) if ata_hist else 0.0,
        "closure_neg_streak_max": int(neg_streak_max),
        "closure_neg_frac": float((closure_arr < 0.0).mean()) if len(closure_arr) else 0.0,
        "danger_entered": danger_entered,
        "danger_streak_max": int(danger_streak_max),
        "recovered": recovered,
        "hdg_seq": hdg_seq,
        "delay_mismatch": delay_mismatch,
    }


def wilson_upper(n_fail, n, z=1.96):
    if n == 0:
        return 1.0
    p = n_fail / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, center + half)


def aggregate(records):
    n = len(records)
    lost = sum(r["lost"] for r in records)
    kills = sum(r["kill"] for r in records)
    launches = sum(r["launches"] for r in records)
    hits = sum(r["hits"] for r in records)
    wez = [r["wez_first"] for r in records if r["wez_first"] is not None]
    ff = [r["first_fire"] for r in records if r["first_fire"] is not None]
    danger_n = sum(r["danger_entered"] for r in records)
    return {
        "n": n,
        "lost": lost,
        "lost_rate": lost / max(n, 1),
        "lost_wilson_95_upper": wilson_upper(lost, n),
        "kill_rate": kills / max(n, 1),
        "launches_per_episode": launches / max(n, 1),
        "hit_rate": hits / max(launches, 1),
        "wez_reach_rate": len(wez) / max(n, 1),
        "time_to_wez_median": float(np.median(wez)) if wez else None,
        "first_fire_median": float(np.median(ff)) if ff else None,
        "max_ata_mean": float(np.mean([r["max_ata"] for r in records])),
        "closure_neg_frac_mean": float(np.mean([r["closure_neg_frac"]
                                                for r in records])),
        "closure_neg_streak_max_mean": float(np.mean([r["closure_neg_streak_max"]
                                                      for r in records])),
        "danger_entered_rate": danger_n / max(n, 1),
        "danger_streak_max_mean": float(np.mean([r["danger_streak_max"]
                                                 for r in records])),
        "recovery_rate_given_danger": (
            sum(r["recovered"] for r in records) / max(danger_n, 1)
            if danger_n else None),
        "delay_mismatch_mean": float(np.mean([r["delay_mismatch"]
                                              for r in records])),
        "quality": {
            "bad": sum(r["quality"]["bad"] for r in records),
            "good": sum(r["quality"]["good"] for r in records),
            "premium": sum(r["quality"]["premium"] for r in records),
        },
        "reasons": {k: sum(1 for r in records if r["reason"] == k)
                    for k in sorted({r["reason"] for r in records})},
    }


def divergence(expert_records, bc_records):
    by_seed = {r["seed"]: r for r in expert_records}
    out = []
    for br in bc_records:
        er = by_seed.get(br["seed"])
        if er is None:
            continue
        m = min(len(br["hdg_seq"]), len(er["hdg_seq"]))
        if m == 0:
            continue
        diff = sum(1 for i in range(m) if br["hdg_seq"][i] != er["hdg_seq"][i])
        out.append(diff / m)
    return float(np.mean(out)) if out else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=list(LEVELS), default=None,
                        help="run all disturbances in this level")
    parser.add_argument("--disturbance", choices=list(DISTURBANCES) + ["none"], default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--inject-mode", choices=["frac", "wez"], default="frac",
                        help="frac: random step in [10-35%] of episode; "
                             "wez: inject_offset steps after first WEZ entry")
    parser.add_argument("--inject-offset", type=int, default=10)
    parser.add_argument("--cell", default=None,
                        help="scenario-matrix cell config to overlay (e.g. dist2_3k)")
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out", default="results/shoot_eval/stress_l1.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    assert (args.level is None) != (args.disturbance is None), \
        "specify exactly one of --level / --disturbance"
    names = LEVELS[args.level] if args.level else [args.disturbance]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    task_cfg = None
    if args.cell:
        from scripts.eval_scenario_matrix import CELLS
        matches = [cfg for lbl, cfg in CELLS if lbl == args.cell]
        assert matches, f"unknown cell {args.cell}"
        task_cfg = matches[0]

    results = {}
    for name in names:
        dist = None if name == "none" else dict(DISTURBANCES[name])
        if dist is not None:
            dist["name"] = name
        exp_recs, bc_recs = [], []
        for s in range(args.start_seed, args.start_seed + args.seeds):
            exp_recs.append(run_one_stressed(
                "expert", None, device, s, dist, difficulty=args.difficulty,
                task_cfg=task_cfg, inject_mode=args.inject_mode,
                inject_offset=args.inject_offset))
            bc_recs.append(run_one_stressed(
                "bc", model, device, s, dist, difficulty=args.difficulty,
                task_cfg=task_cfg, inject_mode=args.inject_mode,
                inject_offset=args.inject_offset))
        exp_a = aggregate(exp_recs)
        bc_a = aggregate(bc_recs)
        results[name] = {
            "disturbance": dist,
            "expert": exp_a,
            "bc": bc_a,
            "hdg_divergence_frac": divergence(exp_recs, bc_recs),
        }
        print(f"[{name:<14s}] expert lost={exp_a['lost_rate']*100:4.1f}% "
              f"(u95 {exp_a['lost_wilson_95_upper']*100:4.1f}%) kill={exp_a['kill_rate']*100:4.1f}%  "
              f"bc lost={bc_a['lost_rate']*100:4.1f}% (u95 {bc_a['lost_wilson_95_upper']*100:4.1f}%) "
              f"kill={bc_a['kill_rate']*100:4.1f}%  "
              f"danger={bc_a['danger_entered_rate']*100:3.0f}% recov={bc_a['recovery_rate_given_danger']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()








