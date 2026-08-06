"""ID/OOD scenario matrix: discrete-expert acceptance + BC zero-shot.

Each cell varies one geometry/behavior dimension from the ID baseline
(bias 30-60 deg, dist 2-5 km, random lateral, alt 0, closing, straight
target). Fixed seeds per cell; the same seed is played by the expert and
by the BC policy (paired within a cell).

Cell verdicts:
  * expert-fail  -> do NOT use as BC training data (per plan)
  * bc-fail      -> candidate stress-DAgger cell

Usage:
  python scripts/eval_scenario_matrix.py --eps-per-cell 12
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

from scripts.eval_paired_bc_vs_expert import run_one_episode, aggregate
from scripts.train_shoot_bc import BCShootPolicy

# label -> task config override (one dimension from the ID baseline)
CELLS = [
    ("id_bias30_60", {}),
    ("bias0_30", {"max_heading_bias_deg": 30.0}),
    ("bias60_90", {"max_heading_bias_deg": 90.0}),
    ("bias90_120", {"max_heading_bias_deg": 120.0}),
    ("dist2_3k", {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0}),
    ("dist5_8k", {"chase_dist_min": 5000.0, "chase_dist_max": 8000.0}),
    ("alt_diff300", {"alt_diff_m": 300.0}),
    ("closure_neutral", {"p_speed_range": (215.0, 235.0),
                         "t_speed_range": (205.0, 235.0)}),
    ("closure_separating", {"p_speed_range": (150.0, 180.0),
                            "t_speed_range": (210.0, 240.0)}),
    ("target_evasive", {"difficulty_level": 0.3}),
    ("lateral_left", {"lateral_sign": -1.0}),
    ("lateral_right", {"lateral_sign": 1.0}),
    ("lateral_center", {"lateral_max_m": 50.0}),
]


KEY_CELLS = {"dist2_3k", "alt_diff300", "target_evasive"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps-per-cell", type=int, default=12)
    parser.add_argument("--eps-key", type=int, default=100,
                        help="episodes per key cell (dist2_3k/alt_diff300/target_evasive)")
    parser.add_argument("--eps-other", type=int, default=50,
                        help="episodes per other cell")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out", default="results/shoot_eval/scenario_matrix_round1.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location=device)
    model = BCShootPolicy().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    results = {}
    for label, cfg in CELLS:
        records = []
        eps_n = args.eps_key if label in KEY_CELLS else args.eps_other
        for s in range(args.start_seed, args.start_seed + eps_n):
            records.append(run_one_episode("expert", None, device, s, **cfg))
            records.append(run_one_episode("bc", model, device, s, **cfg))
        exp = aggregate([r for r in records if r["policy"] == "expert"])
        bc = aggregate([r for r in records if r["policy"] == "bc"])
        expert_ok = exp["lost_rate"] <= 0.15 and exp["wez_reach_rate"] >= 0.8
        bc_ok = bc["lost_rate"] <= 0.15 and bc["wez_reach_rate"] >= 0.8
        verdict = ("ok" if (expert_ok and bc_ok)
                   else "bc-fail" if expert_ok
                   else "expert-fail")
        results[label] = {
            "config": cfg,
            "expert": exp,
            "bc": bc,
            "expert_ok": expert_ok,
            "bc_ok": bc_ok,
            "verdict": verdict,
        }
        print(f"[{label:<20s}] verdict={verdict:<12s} "
              f"expert(lost {exp['lost_rate']*100:4.1f}% kill {exp['kill_rate']*100:4.1f}% "
              f"WEZ {exp['wez_reach_rate']*100:3.0f}%)  "
              f"bc(lost {bc['lost_rate']*100:4.1f}% kill {bc['kill_rate']*100:4.1f}% "
              f"WEZ {bc['wez_reach_rate']*100:3.0f}% launches {bc['launches_per_episode']:.2f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")
    print("verdict counts:",
          {v: sum(1 for r in results.values() if r["verdict"] == v)
           for v in ("ok", "bc-fail", "expert-fail")})


if __name__ == "__main__":
    main()

