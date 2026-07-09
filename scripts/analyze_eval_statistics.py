"""Statistical autopsy of Experiment 3 (AND-gate) evaluation episodes.

Prints quantitative metrics for paper:
  Metric 1: Timeout rate (% episodes reaching 600 steps at 5 Hz)
  Metric 2: Single-entry rate (% steps where at least 1 pursuer < 800m)
  Metric 3: Synchronized-entry rate (% steps where BOTH pursuers < 800m)
  Metric 4: Cooperative success rate (% episodes with AND-gate triggered)
  Metric 5: Termination cause distribution

Usage:
    conda activate marl_env
    python scripts/analyze_eval_statistics.py --data data/viz/exp3_andgate_trajectory.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Constants (must match formation_rllib_env.py)
MAX_EPISODE_TIME = 120.0
DECISION_DT = 0.2      # 5 Hz
MAX_STEPS = int(MAX_EPISODE_TIME / DECISION_DT)  # 600
AND_DIST = 800.0        # AND-gate distance threshold
AND_ANGLE = 30.0        # AND-gate angle threshold
OR_DIST = 200.0         # OR-gate distance


def analyze(data_path: str, min_steps: int = 10):
    data = np.load(data_path, allow_pickle=True)
    n_eps = int(data["n_episodes"])

    print(f"\n{'='*65}")
    print(f"Experiment 3 (AND-gate) — Eval Episode Autopsy")
    print(f"Loaded {n_eps} episodes from {data_path}")
    print(f"{'='*65}")

    # Accumulators
    timeout_count = 0
    total_steps = 0
    successes = 0
    term_reasons: dict = {}

    single_entry_steps = 0   # at least 1 pursuer < 800m
    sync_entry_steps = 0     # both < 800m
    or_entry_steps = 0       # at least 1 pursuer < 200m
    valid_steps = 0

    pincer_above_30_steps = 0
    sync_and_pincer_steps = 0  # both < 800m AND pincer > 30°
    total_pincer_steps = 0

    ep_single_entry_ratios = []
    ep_sync_entry_ratios = []
    ep_timeout_flags = []
    ep_d0_min_list = []
    ep_d1_min_list = []
    ep_pincer_means = []

    for ep_idx in range(n_eps):
        prefix = f"ep{ep_idx}_"
        steps = int(data[f"{prefix}n_steps"])
        total_r = float(data[f"{prefix}total_reward"])

        if steps < min_steps:
            continue

        d0 = data[f"{prefix}p0_distances"]
        d1 = data[f"{prefix}p1_distances"]
        pincer = data.get(f"{prefix}pincer_angles", np.array([]))

        # Timeout detection
        is_timeout = (steps >= MAX_STEPS)
        if is_timeout:
            timeout_count += 1
        ep_timeout_flags.append(is_timeout)

        # Success detection (AND-gate: both < 800m + pincer > 30° for 6 steps)
        sustained = 0
        ep_success = False
        for t in range(min(steps, len(pincer))):
            d0_t = float(d0[t])
            d1_t = float(d1[t])
            pincer_t = float(pincer[t]) if t < len(pincer) else 0

            # Single entry (at least 1 pursuer < 800m)
            if d0_t < AND_DIST or d1_t < AND_DIST:
                single_entry_steps += 1

            # Sync entry (both < 800m)
            if d0_t < AND_DIST and d1_t < AND_DIST:
                sync_entry_steps += 1

            # OR entry (at least 1 < 200m)
            if d0_t < OR_DIST or d1_t < OR_DIST:
                or_entry_steps += 1

            # Pincer above AND threshold
            if pincer_t > AND_ANGLE:
                pincer_above_30_steps += 1

            # Both in AND range + pincer > 30 (AND-gate condition)
            if d0_t < AND_DIST and d1_t < AND_DIST and pincer_t > AND_ANGLE:
                sync_and_pincer_steps += 1
                sustained += 1
                if sustained >= 6:
                    ep_success = True
            else:
                sustained = 0

            total_pincer_steps += 1
            valid_steps += 1

        if ep_success:
            successes += 1

        # Per-episode ratios
        if steps > 0:
            n_single = int((d0 < AND_DIST).sum() + (d1 < AND_DIST).sum())
            n_sync = int(((d0 < AND_DIST) & (d1 < AND_DIST)).sum())
            # Note: single_entry_ratio = fraction of steps where EITHER agent < 800m
            single_ratio = float(
                ((d0 < AND_DIST) | (d1 < AND_DIST)).sum()) / steps
            sync_ratio = float(n_sync) / steps
            ep_single_entry_ratios.append(single_ratio)
            ep_sync_entry_ratios.append(sync_ratio)

        ep_d0_min_list.append(float(d0.min()))
        ep_d1_min_list.append(float(d1.min()))
        if len(pincer) > 0:
            ep_pincer_means.append(float(pincer.mean()))

        total_steps += steps

    # ── Print report ──────────────────────────────────────────────────────
    n_valid = len(ep_d0_min_list)

    print(f"\n── Metric 1: Timeout Rate ──")
    print(f"  Episodes reaching {MAX_STEPS} steps (timeout): "
          f"{timeout_count}/{n_valid} = {100*timeout_count/n_valid:.0f}%")
    print(f"  Mean episode length: {total_steps/n_valid:.0f} steps "
          f"({total_steps/n_valid*DECISION_DT:.1f}s)")

    print(f"\n── Metric 2: Single-Entry Rate (< {AND_DIST:.0f}m) ──")
    print(f"  Steps with ≥1 pursuer < {AND_DIST:.0f}m: "
          f"{single_entry_steps}/{valid_steps} = "
          f"{100*single_entry_steps/valid_steps:.1f}%")
    print(f"  Steps with ≥1 pursuer < {OR_DIST:.0f}m (OR-gate): "
          f"{or_entry_steps}/{valid_steps} = "
          f"{100*or_entry_steps/valid_steps:.1f}%")
    print(f"  Per-episode single-entry ratio: "
          f"mean={np.mean(ep_single_entry_ratios)*100:.1f}%, "
          f"median={np.median(ep_single_entry_ratios)*100:.1f}%")

    print(f"\n── Metric 3: Synchronized-Entry Rate (BOTH < {AND_DIST:.0f}m) ──")
    print(f"  Steps with BOTH < {AND_DIST:.0f}m: "
          f"{sync_entry_steps}/{valid_steps} = "
          f"{100*sync_entry_steps/valid_steps:.1f}%")
    print(f"  Per-episode sync-entry ratio: "
          f"mean={np.mean(ep_sync_entry_ratios)*100:.1f}%, "
          f"median={np.median(ep_sync_entry_ratios)*100:.1f}%")

    print(f"\n── Metric 4: AND-gate Condition (BOTH < {AND_DIST:.0f}m + Pincer > {AND_ANGLE:.0f}°) ──")
    print(f"  Steps satisfying AND-gate condition: "
          f"{sync_and_pincer_steps}/{total_pincer_steps} = "
          f"{100*sync_and_pincer_steps/max(total_pincer_steps,1):.1f}%")
    print(f"  Steps with pincer > {AND_ANGLE:.0f}°: "
          f"{pincer_above_30_steps}/{total_pincer_steps} = "
          f"{100*pincer_above_30_steps/max(total_pincer_steps,1):.1f}%")
    print(f"  AND-gate cooperative successes: {successes}/{n_valid} = "
          f"{100*successes/n_valid:.1f}%")

    print(f"\n── Metric 5: Distance & Pincer Summary ──")
    print(f"  d_min(any pursuer): "
          f"mean={np.mean([min(a,b) for a,b in zip(ep_d0_min_list, ep_d1_min_list)]):.0f}m")
    print(f"  d0_min: mean={np.mean(ep_d0_min_list):.0f}m, "
          f"median={np.median(ep_d0_min_list):.0f}m")
    print(f"  d1_min: mean={np.mean(ep_d1_min_list):.0f}m, "
          f"median={np.median(ep_d1_min_list):.0f}m")
    if ep_pincer_means:
        print(f"  Pincer angle: mean={np.mean(ep_pincer_means):.1f}°, "
              f"median={np.median(ep_pincer_means):.1f}°")

    # ── Diagnosis ─────────────────────────────────────────────────────────
    print(f"\n── DIAGNOSIS ──")
    sync_pct = 100 * sync_entry_steps / max(valid_steps, 1)
    single_pct = 100 * single_entry_steps / max(valid_steps, 1)
    and_pct = 100 * sync_and_pincer_steps / max(total_pincer_steps, 1)
    timeout_pct = 100 * timeout_count / max(n_valid, 1)

    print(f"  Bottleneck analysis:")
    if single_pct > 50 and sync_pct < 20:
        print(f"  → PRIMARY BOTTLENECK: Synchronization. "
              f"Agents CAN approach target ({single_pct:.0f}% single-entry), "
              f"but rarely BOTH within {AND_DIST:.0f}m ({sync_pct:.0f}% sync-entry).")
    if sync_pct > 10 and and_pct < 2:
        print(f"  → BOTTLENECK: Pincer angle. "
              f"Agents achieve dual-entry ({sync_pct:.0f}%) but lack flanking geometry.")
    if timeout_pct > 60:
        print(f"  → BOTTLENECK: Time efficiency. "
              f"Agents survive full episodes ({timeout_pct:.0f}% timeout) "
              f"but can't close distance fast enough.")
    if single_pct < 30:
        print(f"  → BOTTLENECK: Approach. "
              f"Agents rarely approach within {AND_DIST:.0f}m ({single_pct:.0f}%).")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical autopsy of AND-gate eval episodes")
    parser.add_argument("--data", type=str,
                        default="data/viz/exp3_andgate_trajectory.npz",
                        help="Path to collected .npz file")
    parser.add_argument("--min-steps", type=int, default=10,
                        help="Minimum episode steps to include")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found. Run collect_viz_data.py first.")
        sys.exit(1)

    analyze(args.data, min_steps=args.min_steps)
