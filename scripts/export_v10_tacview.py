"""Export Tacview ACMI for 2v1 cooperative formation pursuit.

Runs the FormationRLlibEnv directly (no model loading needed)
and exports ACMI files showing the full 3-aircraft engagement.

Usage:
    conda activate marl_env
    python scripts/export_v10_tacview.py --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import logging
import numpy as np

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
logging.getLogger("gymnasium").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment.formation_rllib_env import FormationRLlibEnv
from src.logging.tacview_exporter import TacviewExporter
from src.utils.units import rad_to_deg


def _get_state(aircraft):
    """Extract position + attitude from Aircraft wrapper."""
    s = aircraft.aircraft.state
    return {
        "lon_deg": float(s["lon_deg"]),
        "lat_deg": float(s["lat_deg"]),
        "alt_m": float(s["alt_m"]),
        "roll_deg": float(s["roll_deg"]),
        "pitch_deg": float(s["pitch_deg"]),
        "yaw_deg": float(s["yaw_deg"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/v10_export_tacview")
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Target maneuver difficulty (0.0 = straight, 1.0 = max evasion)")
    parser.add_argument("--and-dist", type=int, default=800,
                        help="AND-gate distance threshold (meters)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Output dir: {args.output}")
    print(f"Difficulty: {args.difficulty}, AND-dist: {args.and_dist}m")
    print(f"Running {args.episodes} episodes...\n")

    success_count = 0

    for ep_idx in range(args.episodes):
        env = FormationRLlibEnv({
            "difficulty_level": args.difficulty,
            "cooperative_mode": True,
            "and_dist": args.and_dist,
        })

        obs_dict, _ = env.reset(seed=args.seed + ep_idx)
        done = False
        frames = []
        ep_reward = 0.0

        while not done and len(frames) < 1000:
            # Record frame
            frame = {"time": len(frames) * 0.2}  # 0.2s per decision step at 5Hz
            for i, ps in enumerate(env.pursuers):
                frame[f"p{i}"] = _get_state(ps)
            for i, ts in enumerate(env.targets):
                frame[f"t{i}"] = _get_state(ts)
            frames.append(frame)

            # Sample actions from action space (discrete tactical primitives)
            actions = {}
            for aid in env._agent_ids:
                actions[aid] = env.action_space[aid].sample()

            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            for aid in env._agent_ids:
                ep_reward += rewards.get(aid, 0.0)
            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        # Check termination reason
        termination = infos.get("p0", {}).get("termination_reason",
                     infos.get("__common__", {}).get("termination_reason", "timeout"))
        is_success = "success" in str(termination).lower()
        if is_success:
            success_count += 1
            tag = "SUCCESS"
        else:
            tag = str(termination).replace(" ", "_")[:30]

        print(f"  Ep {ep_idx+1:2d}: steps={len(frames):3d}  rew={ep_reward:+.0f}  [{tag}]")

        # Export ACMI
        acmi_frames = []
        for f in frames:
            acmi_frames.append({
                "time": f["time"],
                "attacker": f.get("p0", f.get("t0", {"lon_deg": 120, "lat_deg": 60, "alt_m": 3000,
                                                      "roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0})),
                "evader": f.get("p1", {"lon_deg": 120, "lat_deg": 60.01, "alt_m": 3000,
                                       "roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0}),
            })

        exporter = TacviewExporter(
            filepath=os.path.join(args.output, f"v10_ep{ep_idx+1}_{tag}.txt.acmi"),
            base_lat=30.0, base_lon=120.0)
        exporter.write(acmi_frames)

    print(f"\n  Success: {success_count}/{args.episodes}  ({100*success_count/max(1,args.episodes):.0f}%)")
    print(f"  ACMI files saved to: {args.output}/")


if __name__ == "__main__":
    main()
