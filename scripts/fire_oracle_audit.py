"""dist2_3k reachability audit + fire oracle (PPO step 0).

Hypothesis: dist2_3k's low kill rate is a reachability problem (cooldown,
engagement-window length, episode length) rather than a fire-timing problem.
Missiles do not affect the pursuer's flight path (difficulty 0 target flies
straight), so we fix the BC heading/speed trajectory per seed and enumerate
fire policies on top of it:

  asap        : fire at the first legal step (cooldown already in mask)
  delay_30/60 : fire at the first legal step >= first_allowed + N
  dlz_mid     : fire only when DLZ depth in [0.4, 0.6]
  dlz_deep    : fire only when DLZ depth >= 0.6
  interval_100: fire ASAP, then only every 100 steps when legal
  bc          : the frozen BC's own fire decision (reference)

Also reports the reachability audit for the `asap` run:
  first fire_allowed step, cumulative allowed-window length, 1st-4th launch
  times, 4th-launch completion rate, blocker reasons, target HP progression,
  max legal launches per seed.

Usage:
  python scripts/fire_oracle_audit.py --cell dist2_3k --seeds 60 --oracles all
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
from scripts.eval_bc_1v1 import policy_action
from scripts.train_shoot_bc import BCShootPolicy
from scripts.generate_shoot_rule_expert import dlz_depth, FIRE_IDX
from scripts.eval_scenario_matrix import CELLS
from src.environment.singlecombat_shoot_task import MIN_ATTACK_INTERVAL

MAX_STEPS = 1500
ORACLES = ["asap", "delay_30", "delay_60", "dlz_mid", "dlz_deep",
           "interval_100", "bc"]


def fire_decision(oracle, mask, obs, step, first_allowed, last_fire):
    legal = mask[FIRE_IDX] == 1.0
    if not legal:
        return 0
    if oracle == "asap":
        return 1
    if oracle == "bc":
        return 1  # overwritten by the policy below (not used through here)
    if oracle == "delay_30":
        return 1 if step >= first_allowed + 30 else 0
    if oracle == "delay_60":
        return 1 if step >= first_allowed + 60 else 0
    if oracle == "dlz_mid":
        d = dlz_depth(obs)
        return 1 if 0.4 <= d <= 0.6 else 0
    if oracle == "dlz_deep":
        return 1 if dlz_depth(obs) >= 0.6 else 0
    if oracle == "interval_100":
        return 1 if last_fire is None or step - last_fire >= 100 else 0
    raise ValueError(oracle)


def run_seed(policy_model, device, seed, cell_cfg, oracle):
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(cell_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    wez_first = fire_first = first_allowed = None
    allowed_total = 0
    allowed_after_last_fire = 0
    launches = hits = 0
    launch_steps = []
    last_fire = None
    hit_after_launch = []  # hits attributed to each launch index
    hp_after_hit = []
    quality = {"bad": 0, "good": 0, "premium": 0}
    hit_flags = set()
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        if mask[FIRE_IDX] == 1.0:
            if first_allowed is None:
                first_allowed = step
            allowed_total += 1
            if last_fire is not None:
                allowed_after_last_fire += 1
        act = policy_action(policy_model, obs["p0"], device)
        if oracle == "bc":
            fire = act[3]
        else:
            fire = fire_decision(oracle, mask, obs["p0"], step,
                                 first_allowed if first_allowed is not None else 0,
                                 last_fire)
        act[3] = fire
        hits_before = hits
        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits += env.task._hit_this_step.get("p0", 0)
        if env.task._has_launched_this_step.get("p0", False):
            if fire_first is None:
                fire_first = step
            launch_steps.append(step)
            last_fire = step
            launches += 1
            hit_after_launch.append(0)
        if hits > hits_before:
            for i in range(len(hit_after_launch) - 1, -1, -1):
                if hit_after_launch[i] == 0 and i not in hit_flags:
                    hit_after_launch[i] = hits - hits_before
                    hit_flags.add(i)
                    break
            hp_after_hit.append(env.targets[0].hits_taken)
        if wez_first is None and env.task._is_valid_launch_envelope(
                env.pursuers[0], env.targets[0]):
            wez_first = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break
    # blocker classification for not reaching 4 launches
    blocker = None
    if launches < 4:
        if reason == "target_killed":
            blocker = "target_killed_early"
        elif reason == "lost_target":
            blocker = "lost_target"
        elif last_fire is not None and step - last_fire < MIN_ATTACK_INTERVAL:
            blocker = "cooldown_at_end"
        elif allowed_after_last_fire == 0:
            blocker = "window_never_reopened"
        else:
            blocker = "episode_end_in_window"
    env.close()
    return {
        "seed": int(seed),
        "oracle": oracle,
        "reason": reason,
        "steps": step + 1,
        "kill": reason == "target_killed",
        "lost": reason == "lost_target",
        "launches": launches,
        "hits": hits,
        "first_fire": fire_first,
        "wez_first": wez_first,
        "first_allowed": first_allowed,
        "allowed_total": allowed_total,
        "launch_steps": launch_steps,
        "hit_after_launch": hit_after_launch,
        "hp_after_hit": hp_after_hit,
        "blocker": blocker,
    }


def aggregate(records):
    n = len(records)
    launches = sum(r["launches"] for r in records)
    hits = sum(r["hits"] for r in records)
    kills = sum(r["kill"] for r in records)
    reach4 = sum(1 for r in records if r["launches"] >= 4)
    la = [r["launch_steps"] for r in records if r["launch_steps"]]
    first2fourth = []
    for r in records:
        if len(r["launch_steps"]) >= 4:
            first2fourth.append(r["launch_steps"][3] - r["launch_steps"][0])
    return {
        "n": n,
        "lost_rate": sum(r["lost"] for r in records) / max(n, 1),
        "kill_rate": kills / max(n, 1),
        "launches_per_episode": launches / max(n, 1),
        "hit_rate": hits / max(launches, 1),
        "reach_4_launches_rate": reach4 / max(n, 1),
        "first_fire_median": float(np.median([r["first_fire"] for r in records
                                              if r["first_fire"] is not None]))
        if any(r["first_fire"] is not None for r in records) else None,
        "first_allowed_median": float(np.median(
            [r["first_allowed"] for r in records
             if r["first_allowed"] is not None]))
        if any(r["first_allowed"] is not None for r in records) else None,
        "allowed_total_median": float(np.median(
            [r["allowed_total"] for r in records])),
        "first_to_fourth_median": float(np.median(first2fourth)) \
        if first2fourth else None,
        "blockers": {k: sum(1 for r in records if r["blocker"] == k)
                     for k in sorted({r["blocker"] for r in records
                                      if r["blocker"]})},
        "reasons": {k: sum(1 for r in records if r["reason"] == k)
                    for k in sorted({r["reason"] for r in records})},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="dist2_3k")
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--oracles", default="all")
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out", default="results/shoot_eval/fire_oracle_dist2_3k.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cell_cfg = dict([cfg for lbl, cfg in CELLS if lbl == args.cell][0])
    oracles = ORACLES if args.oracles == "all" else args.oracles.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    print(f"[fire-oracle] cell={args.cell} config={cell_cfg} "
          f"seeds={args.seeds} oracles={oracles}")
    print(f"[constants] NUM_MISSILES=4, MIN_ATTACK_INTERVAL={MIN_ATTACK_INTERVAL} "
          f"steps ({MIN_ATTACK_INTERVAL*0.2:.0f}s), min 1st->4th = "
          f"{3*MIN_ATTACK_INTERVAL} steps ({3*MIN_ATTACK_INTERVAL*0.2:.0f}s)")

    results = {}
    for oracle in oracles:
        records = [run_seed(model, device, s, cell_cfg, oracle)
                   for s in range(args.start_seed, args.start_seed + args.seeds)]
        agg = aggregate(records)
        results[oracle] = {"aggregate": agg, "per_seed": records}
        print(f"[{oracle:<12s}] kills={agg['kill_rate']*100:5.1f}%  "
              f"launches={agg['launches_per_episode']:.2f}/ep  "
              f"hits={agg['hit_rate']*100:4.1f}%  reach4={agg['reach_4_launches_rate']*100:4.1f}%  "
              f"lost={agg['lost_rate']*100:4.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({o: {"aggregate": results[o]["aggregate"]}
                   for o in results}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

