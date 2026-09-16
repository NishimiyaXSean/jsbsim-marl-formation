"""L1.2 v2: 海豚跳专项测试，使用 SinglePursuitTask（无 target 干扰）。

v1 错用 SingleCombatShootTask，pursuer hold 不动会让 target 飞走触发 lost_target，
不是真正的海豚跳。SinglePursuitTask 是 heading-only 任务，target 不参与胜负判断。

测试方法：保持 heading 不变 30s，记录俯仰/高度振荡。
通过条件：俯仰变化 < ±2°、高度变化 < ±20m

Usage:
  python scripts/diag_porpoise_v2.py --seconds 30 --seed 42
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.single_pursuit_task import SinglePursuitTask


# SinglePursuitTask: Discrete(5) heading-only. DELTA_HEADINGS = [-15, -5, 0, 5, 15]
# Index 2 = 0° delta = "hold heading"
HOLD_HDG_IDX = 2
# Other test actions
HDG_SLIGHT_LEFT = 1   # -5° delta
HDG_LEFT = 0         # -15° delta
HDG_HOLD = 2         # 0° delta (hold)
HDG_SLIGHT_RIGHT = 3 # +5° delta
HDG_RIGHT = 4        # +15° delta


def run_pattern(action_idx, seconds, seed, sim_freq_hz=60):
    """Run SinglePursuitTask with action_idx for `seconds`. Return trace + stats."""
    max_steps = int(seconds * sim_freq_hz)
    cfg = {"difficulty_level": 0.0}  # difficulty 0 = no evasion
    env = BaseEnv(task=SinglePursuitTask(cfg))
    obs, _ = env.reset(seed=seed)

    log = []
    initial = None
    termination_reason = None

    for step in range(max_steps):
        obs, rews, terms, truncs, info = env.step({"p0": action_idx})

        try:
            state = env.pursuers[0].aircraft.state
            pitch = float(state["pitch_deg"])
            roll = float(state["roll_deg"])
            alt = float(state["alt_m"])
            spd = float(state["airspeed_mps"])
            hdg = float(state.get("heading_deg", 0.0))
        except Exception as e:
            break

        if initial is None:
            initial = {"pitch": pitch, "roll": roll, "alt": alt, "spd": spd, "hdg": hdg}

        log.append({
            "step": step,
            "t": step / sim_freq_hz,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "alt_m": alt,
            "airspeed_mps": spd,
            "heading_deg": hdg,
            "pitch_dev": pitch - initial["pitch"],
            "alt_dev": alt - initial["alt"],
            "spd_dev": spd - initial["spd"],
            "hdg_dev": hdg - initial["hdg"],
        })

        if terms.get("__all__") or truncs.get("__all__"):
            termination_reason = info.get("p0", {}).get("termination_reason", "unknown")
            break

    env.close()
    return log, initial, termination_reason


def detect_oscillation(trace, axis_key):
    vals = np.array([r[axis_key] for r in trace])
    peak_to_peak = float(vals.max() - vals.min())
    std = float(vals.std())
    return peak_to_peak, std


def osc_freq(trace, axis_key):
    """Detect oscillation frequency via zero crossings."""
    vals = np.array([r[axis_key] for r in trace])
    crossings = ((vals[:-1] * vals[1:]) < 0).sum()
    n_sec = len(trace) / 60.0
    return crossings / max(n_sec, 1) / 2  # full cycles per second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/health_check/porpoise_v2.json")
    args = parser.parse_args()

    patterns = {
        "hold_hdg": HOLD_HDG_IDX,
        "slight_left": HDG_SLIGHT_LEFT,
        "slight_right": HDG_SLIGHT_RIGHT,
        "left": HDG_LEFT,
        "right": HDG_RIGHT,
    }

    print(f"[porpoise-v2] seconds={args.seconds}, seed={args.seed}")
    print(f"[porpoise-v2] running {len(patterns)} heading patterns...")

    results = {}
    for name, idx in patterns.items():
        print(f"\n[porpoise-v2] === pattern: {name} (action={idx}) ===")
        trace, initial, term_reason = run_pattern(idx, args.seconds, args.seed)

        if not trace:
            print(f"  NO TRACE — failed immediately")
            results[name] = {"verdict": "FAIL: no trace"}
            continue

        pitch_pp, pitch_std = detect_oscillation(trace, "pitch_dev")
        alt_pp, alt_std = detect_oscillation(trace, "alt_dev")
        spd_pp, spd_std = detect_oscillation(trace, "spd_dev")
        hdg_pp, hdg_std = detect_oscillation(trace, "hdg_dev")
        pitch_f = osc_freq(trace, "pitch_dev")
        alt_f = osc_freq(trace, "alt_dev")

        verdict = "PASS"
        if term_reason:
            verdict = f"PARTIAL: terminated at step {len(trace)} with reason '{term_reason}'"
        elif pitch_pp > 4.0 or alt_pp > 40.0:
            verdict = f"FAIL: pitch_pp={pitch_pp:.2f}° alt_pp={alt_pp:.1f}m"
        elif pitch_pp > 2.0 or alt_pp > 20.0:
            verdict = f"MARGINAL: pitch_pp={pitch_pp:.2f}° alt_pp={alt_pp:.1f}m"

        results[name] = {
            "n_steps": len(trace),
            "ran_full_seconds": term_reason is None,
            "termination_reason": term_reason,
            "initial_state": initial,
            "pitch_pp_deg": pitch_pp,
            "pitch_std_deg": pitch_std,
            "pitch_osc_hz": pitch_f,
            "alt_pp_m": alt_pp,
            "alt_std_m": alt_std,
            "alt_osc_hz": alt_f,
            "spd_pp_mps": spd_pp,
            "hdg_pp_deg": hdg_pp,
            "verdict": verdict,
        }

        print(f"  ran {len(trace)} steps ({'full' if term_reason is None else f'term={term_reason}'})")
        print(f"  initial: pitch={initial['pitch']:.2f}° alt={initial['alt']:.0f}m spd={initial['spd']:.1f}mps")
        print(f"  pitch pp={pitch_pp:.2f}° (std={pitch_std:.3f}°), osc {pitch_f:.2f}Hz")
        print(f"  alt   pp={alt_pp:.1f}m (std={alt_std:.2f}m), osc {alt_f:.2f}Hz")
        print(f"  spd   pp={spd_pp:.2f}mps, hdg pp={hdg_pp:.2f}°")
        print(f"  VERDICT: {verdict}")

    # Overall
    overall = "PASS"
    fail_patterns = [n for n, r in results.items() if "FAIL" in r.get("verdict", "")]
    marginal_patterns = [n for n, r in results.items() if "MARGINAL" in r.get("verdict", "")]
    term_patterns = [n for n, r in results.items() if "PARTIAL" in r.get("verdict", "")]
    if fail_patterns:
        overall = f"FAIL ({len(fail_patterns)} patterns)"
    elif marginal_patterns:
        overall = f"MARGINAL ({len(marginal_patterns)} patterns)"
    elif term_patterns and not fail_patterns:
        overall = f"ALL_TERMINATED ({len(term_patterns)} patterns ended early)"

    print(f"\n{'=' * 60}")
    print(f"[porpoise-v2] OVERALL: {overall}")
    print(f"[porpoise-v2]   patterns: {list(patterns.keys())}")
    print(f"[porpoise-v2]   failures: {fail_patterns}")
    print(f"[porpoise-v2]   marginal: {marginal_patterns}")
    print(f"[porpoise-v2]   early term: {term_patterns}")
    print(f"{'=' * 60}")

    os.makedirs = os.makedirs if hasattr(os, "makedirs") else None  # noqa
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "seconds": args.seconds,
            "seed": args.seed,
            "patterns": {k: v for k, v in results.items()},
            "overall": overall,
        }, f, indent=2)
    print(f"[porpoise-v2] saved to {args.out}")


if __name__ == "__main__":
    main()