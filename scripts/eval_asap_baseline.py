"""ASAP fire control group for the PPO stage (non-blocking reference).

heading/speed = frozen BC (deterministic argmax)
fire          = fire whenever the environment mask allows it (ASAP)

Covers:
  * id     : default ID geometry, 500 seeds
  * cells  : all 13 scenario-matrix cells, --seeds-per-cell each
  * stress : representative L1/L2/L3/L4 disturbances, --stress-seeds each

This is the P2 ceiling: if ASAP beats BC everywhere with bad=0, PPO's job is
basically to approach ASAP; if ASAP only helps at close range, PPO must learn
the conditional boundary.

Usage:
  python scripts/eval_asap_baseline.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from scripts.stress_eval import (
    run_one_stressed, aggregate, DISTURBANCES, LEVELS)
from scripts.eval_scenario_matrix import CELLS
from scripts.train_shoot_bc import BCShootPolicy

STRESS_REPS = [
    ("L1", "hdg_k1"), ("L2", "hdg_k3"),
    ("L3", "combo_l3"), ("L4", "hdg_k20"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["id", "cells", "stress", "all"],
                        default="all")
    parser.add_argument("--id-seeds", type=int, default=500)
    parser.add_argument("--seeds-per-cell", type=int, default=50)
    parser.add_argument("--stress-seeds", type=int, default=60)
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out", default="results/shoot_eval/asap_baseline.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    results = {}

    def run_block(key, task_cfg, dist, n_seeds):
        recs = [run_one_stressed("bc", model, device, s, dist,
                                 task_cfg=task_cfg, fire_mode="asap")
                for s in range(n_seeds)]
        agg = aggregate(recs)
        results[key] = {"aggregate": agg, "n": n_seeds}
        print(f"[{key:<18s}] lost={agg['lost_rate']*100:4.1f}%  "
              f"kill={agg['kill_rate']*100:5.1f}%  "
              f"launches={agg['launches_per_episode']:.2f}/ep  "
              f"hits={agg['hit_rate']*100:4.1f}%  quality={agg['quality']}")

    if args.mode in ("id", "all"):
        run_block("id_500", {}, None, args.id_seeds)
    if args.mode in ("cells", "all"):
        for label, cfg in CELLS:
            run_block(f"cell_{label}", cfg, None, args.seeds_per_cell)
    if args.mode in ("stress", "all"):
        for lv, name in STRESS_REPS:
            dist = dict(DISTURBANCES[name])
            dist["name"] = name
            run_block(f"stress_{lv}_{name}", {}, dist, args.stress_seeds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: v["aggregate"] for k, v in results.items()}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
