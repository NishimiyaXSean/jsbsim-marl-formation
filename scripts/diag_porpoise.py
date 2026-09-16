"""L1.2: 海豚跳专项测试（porpoising detection）。

应用 hold-heading, hold-altitude, hold-speed 命令 30s，记录
俯仰/高度振荡。这是 Sean 之前反复头疼的问题。

通过条件：俯仰变化 < ±2°、高度变化 < ±20m

Usage:
  python scripts/diag_porpoise.py --seconds 30 --seed 42
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
from src.environment.singlecombat_shoot_task import SingleCombatShootTask


# Hold-everything action: speed=1 (middle), heading=2 (middle),
# altitude=0 (only option), fire=0
# If aircraft drifts despite this, try other patterns.
HOLD_ACTIONS = {
    "speed_hold_hdg_hold": np.array([1, 2, 0, 0], dtype=np.int64),
    # Alt: slight speed-up + slight hdg-left (test controller response)
    "speed_up_slight_lh": np.array([2, 1, 0, 0], dtype=np.int64),
    "speed_hold_slight_lh": np.array([1, 1, 0, 0], dtype=np.int64),
}


def run_porpoise(action, seconds, seed, sim_freq_hz=60):
    """Apply `action` for `seconds` at `sim_freq_hz` Hz. Return trace + stats."""
    max_steps = int(seconds * sim_freq_hz)
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)

    log = []
    initial = None

    for step in range(max_steps):
        obs, rews, terms, truncs, info = env.step({"p0": action})

        # Pull aircraft state
        try:
            state = env.pursuers[0].aircraft.state
            pitch = float(state["pitch_deg"])
            roll = float(state["roll_deg"])
            alt = float(state["alt_m"])
            spd = float(state["airspeed_mps"])
            hdg = float(state.get("heading_deg", 0.0))
        except Exception as e:
            print(f"  step {step}: state read failed: {e}")
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
        })

        if terms.get("__all__") or truncs.get("__all__"):
            break
        if not np.isfinite(np.array([pitch, alt, spd])).all():
            break

    env.close()
    return log, initial


def detect_oscillation(trace, axis_key, threshold_range):
    """Detect oscillation: peak-to-peak amplitude above threshold.
    Returns (peak_to_peak, oscillation_detected)."""
    vals = np.array([r[axis_key] for r in trace])
    peak_to_peak = float(vals.max() - vals.min())
    return peak_to_peak, peak_to_peak > threshold_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/health_check/porpoise.json")
    args = parser.parse_args()

    print(f"[porpoise] seconds={args.seconds}, seed={args.seed}")
    print(f"[porpoise] running 3 hold-action patterns...")

    results = {}
    for name, action in HOLD_ACTIONS.items():
        print(f"\n[porpoise] === pattern: {name}, action={action.tolist()} ===")
        trace, initial = run_porpoise(action, args.seconds, args.seed)

        if not trace:
            print(f"  NO TRACE — env reset/terminated immediately")
            results[name] = {"verdict": "FAIL: no trace"}
            continue

        pitch_pp, pitch_osc = detect_oscillation(trace, "pitch_dev", threshold_range=4.0)
        alt_pp, alt_osc = detect_oscillation(trace, "alt_dev", threshold_range=40.0)
        spd_pp, spd_osc = detect_oscillation(trace, "spd_dev", threshold_range=10.0)

        # Compute oscillation frequency (zero crossings per second)
        pitch_vals = np.array([r["pitch_dev"] for r in trace])
        zero_crossings = ((pitch_vals[:-1] * pitch_vals[1:]) < 0).sum()
        n_seconds = len(trace) / 60.0
        osc_freq_hz = zero_crossings / max(n_seconds, 1) / 2  # full cycles per second

        verdict = "PASS"
        if pitch_pp > 4.0 or alt_pp > 40.0:
            verdict = "FAIL: oscillation > > threshold"
        elif pitch_pp > 2.0 or alt_pp > 20.0:
            verdict = "MARGINAL"
        if osc_freq_hz > 0.3:
            verdict += f" + OSC_FREQ={osc_freq_hz:.2f}Hz (porpoise pattern)"

        results[name] = {
            "n_steps": len(trace),
            "initial_state": initial,
            "pitch_pp_deg": pitch_pp,
            "pitch_osc_detected": pitch_osc,
            "alt_pp_m": alt_pp,
            "alt_osc_detected": alt_osc,
            "spd_pp_mps": spd_pp,
            "spd_osc_detected": spd_osc,
            "osc_freq_hz": osc_freq_hz,
            "verdict": verdict,
            "trace": trace,
        }

        print(f"  ran {len(trace)} steps")
        print(f"  initial state: pitch={initial['pitch']:.1f}° alt={initial['alt']:.0f}m spd={initial['spd']:.1f}mps")
        print(f"  pitch peak-to-peak: {pitch_pp:.2f}°  ({'OSC' if pitch_osc else 'no'})")
        print(f"  alt peak-to-peak:   {alt_pp:.1f}m   ({'OSC' if alt_osc else 'no'})")
        print(f"  spd peak-to-peak:   {spd_pp:.1f}mps ({'OSC' if spd_osc else 'no'})")
        print(f"  osc freq: {osc_freq_hz:.2f}Hz")
        print(f"  VERDICT: {verdict}")

    # Overall verdict
    overall = "PASS"
    for name, r in results.items():
        if "FAIL" in r.get("verdict", ""):
            overall = "FAIL"
            break
        elif "MARGINAL" in r.get("verdict", ""):
            overall = "MARGINAL"

    print(f"\n{'=' * 60}")
    print(f"[porpoise] OVERALL: {overall}")
    print(f"{'=' * 60}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "seconds": args.seconds,
            "seed": args.seed,
            "patterns": list(HOLD_ACTIONS.keys()),
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                       for k, v in results.items()},
            "overall": overall,
            "raw_traces": {k: v.get("trace", []) for k, v in results.items()},
        }, f)
    print(f"[porpoise] saved to {args.out}")


if __name__ == "__main__":
    main()