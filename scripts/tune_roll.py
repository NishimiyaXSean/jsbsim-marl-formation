"""Phase 2: Single-channel roll tuning (PD-only, no integral).

Lock elevator at speed-dependent trim, rudder at 0, throttle at 0.80.
Command roll step sequence: 0°→60°→0°→-60°→0°.
Tune roll_kp and roll_kd for crisp response without overshoot.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/tune_roll.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig

os.environ.setdefault("JSBSIM_DEBUG", "0")

# ── Test parameters ────────────────────────────────────────────────────────

DT = 1.0 / 60.0
WARMUP_S = 2.0
HOLD_S = 3.0            # hold each target for 3 seconds
THROTTLE = 0.80
INIT_ALT_FT = 9842
INIT_HEADING_DEG = 0.0
SPEED_KTS = 400

# Roll step sequence
ROLL_STEPS = [0.0, 60.0, 0.0, -60.0, 0.0]   # degrees


def _load_trim_for_speed(speed_mps: float) -> float:
    """Load trim elevator from Phase 1 data, or compute via 1/V^2 law."""
    try:
        with open("data/trim_table.json") as f:
            data = json.load(f)
        ref_V = data["ref_speed_mps"]
        ref_E = data["ref_elevator"]
        return ref_E * (ref_V / max(speed_mps, 80.0)) ** 2
    except FileNotFoundError:
        # Fallback: use 1/V^2 from hardcoded reference
        return -0.05 * (176.0 / max(speed_mps, 80.0)) ** 2


def _run_roll_test(kp: float, kd: float, label: str, plot: bool = False
                   ) -> dict:
    """Run roll step sequence with given gains, return metrics."""
    config = BFMAutopilotConfig(roll_kp=kp, roll_kd=kd, roll_ki=0.0)
    ap = BFMAutopilot(config)

    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
             heading_deg=INIT_HEADING_DEG, speed_kts=SPEED_KTS)

    all_t = []
    all_roll = []
    all_cmd = []
    all_ail = []

    t = 0.0
    speed_mps = ac.state["airspeed_mps"]
    trim_elev = _load_trim_for_speed(speed_mps)
    print(f"  [{label}] trim_elev={trim_elev:+.4f} @ {speed_mps:.0f} m/s, "
          f"kp={kp:.2f}, kd={kd:.3f}")

    for target_deg in ROLL_STEPS:
        target_rad = np.deg2rad(target_deg)
        for _ in range(int(HOLD_S / DT)):
            s = ac.state
            thr, elev, ail, rud = ap.step(
                n_x=0.0, n_n=1.0, mu=target_rad, dt=DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
            )
            ac.set_controls(throttle=thr, elevator=trim_elev, aileron=ail, rudder=0.0)
            ac.run()
            all_t.append(t)
            all_roll.append(s["roll_deg"])
            all_cmd.append(target_deg)
            all_ail.append(ail)
            t += DT

    all_t = np.array(all_t)
    all_roll = np.array(all_roll)
    all_cmd = np.array(all_cmd)

    # Metrics: rise time and overshoot for the 0→60° step
    step_start = int(WARMUP_S / DT)
    step_end = step_start + int(HOLD_S / DT)
    seg_roll = all_roll[step_start:step_end]
    seg_cmd = all_cmd[step_start:step_end]

    # Rise time: time from 10% to 90% of step
    step_mag = 60.0
    try:
        t10_idx = np.where(seg_roll >= 6.0)[0][0]   # 10% of 60
        t90_idx = np.where(seg_roll >= 54.0)[0][0]   # 90% of 60
        rise_time = (t90_idx - t10_idx) * DT
    except IndexError:
        rise_time = float("inf")

    overshoot = max(0.0, np.max(seg_roll) - step_mag)

    # Steady-state oscillation (last 1s of each hold)
    ss_samples = int(1.0 / DT)
    steady_rms = float(np.std(all_roll[-ss_samples:]))

    return {
        "kp": kp, "kd": kd,
        "rise_time": rise_time,
        "overshoot": overshoot,
        "steady_rms": steady_rms,
        "t": all_t, "roll": all_roll, "cmd": all_cmd, "ail": all_ail,
    }


def main():
    os.makedirs("results/phase2_tuning", exist_ok=True)
    print("=" * 60)
    print("PHASE 2a: Roll Channel Tuning")
    print("=" * 60)

    # Start conservative, work up
    # JSBSim inner-loop roll-rate PID has kp=3.0 (very aggressive).
    # Our outer loop should be slower to avoid cascaded oscillation.
    gain_sets = [
        (1.0, 0.0,   "kp=1.0  (baseline)"),
        (1.5, 0.0,   "kp=1.5  (V10 default)"),
        (1.5, 0.08,  "kp=1.5, kd=0.08"),
        (2.0, 0.0,   "kp=2.0"),
        (2.0, 0.10,  "kp=2.0, kd=0.10"),
        (2.5, 0.0,   "kp=2.5"),
        (2.5, 0.12,  "kp=2.5, kd=0.12"),
        (3.0, 0.0,   "kp=3.0"),
        (3.0, 0.15,  "kp=3.0, kd=0.15"),
    ]

    best = None
    results = []
    for kp, kd, label in gain_sets:
        r = _run_roll_test(kp, kd, label)
        results.append(r)
        ok = (r["rise_time"] < 0.8 and r["overshoot"] < 6.0 and r["steady_rms"] < 1.0)
        print(f"    rise={r['rise_time']:.3f}s  overshoot={r['overshoot']:.1f}°  "
              f"steady_rms={r['steady_rms']:.1f}°  {'OK' if ok else 'FAIL'}")
        if ok and (best is None or r["rise_time"] < best["rise_time"]):
            best = r

    if best is None:
        # Pick best compromise
        best = min(results, key=lambda r: r["rise_time"] + 10 * r["overshoot"])
        print(f"\n  No perfect set — best compromise: kp={best['kp']}, kd={best['kd']}")
    else:
        print(f"\n  Best: kp={best['kp']}, kd={best['kd']} "
              f"(rise={best['rise_time']:.3f}s, overshoot={best['overshoot']:.1f}°)")

    # ── Plot best result ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(best["t"], best["cmd"], "k--", lw=1.5, alpha=0.5, label="Command")
    ax1.plot(best["t"], best["roll"], "b-", lw=1.5, label=f"Roll (kp={best['kp']}, kd={best['kd']})")
    ax1.set_ylabel("Roll Angle (deg)")
    ax1.set_title("Roll Channel Tuning — Best Result")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(best["t"], best["ail"], "r-", lw=1.0, label="Aileron")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Aileron Cmd")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/phase2_tuning/roll_tuning.png", dpi=150, bbox_inches="tight")
    print("  Plot saved → results/phase2_tuning/roll_tuning.png")
    plt.close("all")

    print(f"\n  Recommended: roll_kp={best['kp']:.2f}, roll_kd={best['kd']:.3f}")


if __name__ == "__main__":
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
