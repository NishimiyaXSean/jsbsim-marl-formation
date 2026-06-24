"""Phase 2: Speed channel tuning.

Wings-level at trim, command 176→200 m/s step.  Speed dynamics are slow
(thrust → acceleration integration); derivative is noisy so kd=0.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/tune_speed.py
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

DT = 1.0 / 60.0
WARMUP_S = 2.0
HOLD_S = 10.0           # speed changes are slow — longer hold
THROTTLE = 0.80
INIT_ALT_FT = 9842
INIT_HEADING_DEG = 90.0
SPEED_KTS = 400
TARGET_SPEED = 200.0     # m/s


def _load_trim_for_speed(speed_mps: float) -> float:
    try:
        with open("data/trim_table.json") as f:
            data = json.load(f)
        ref_V = data["ref_speed_mps"]
        ref_E = data["ref_elevator"]
        return ref_E * (ref_V / max(speed_mps, 80.0)) ** 2
    except FileNotFoundError:
        return -0.05 * (176.0 / max(speed_mps, 80.0)) ** 2


def _run_speed_test(kp: float, label: str) -> dict:
    config = BFMAutopilotConfig(speed_kp=kp, speed_ki=0.0, speed_kd=0.0)
    ap = BFMAutopilot(config)

    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
             heading_deg=INIT_HEADING_DEG, speed_kts=SPEED_KTS)

    all_t, all_spd, all_thr = [], [], []
    t = 0.0
    speed_mps = ac.state["airspeed_mps"]
    trim_elev = _load_trim_for_speed(speed_mps)
    print(f"  [{label}] trim_elev={trim_elev:+.4f}, start_speed={speed_mps:.0f} m/s")

    # Warmup at 1G level flight
    for _ in range(int(WARMUP_S / DT)):
        s = ac.state
        thr, elev, ail, rud = ap.step(
            n_x=0.0, n_n=1.0, mu=0.0, dt=DT,
            n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
            airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
        )
        ac.set_controls(throttle=thr, elevator=trim_elev, aileron=0.0, rudder=0.0)
        ac.run()

    # Command speed step: n_x = 1.0 (gentle acceleration) for the whole hold
    for _ in range(int(HOLD_S / DT)):
        s = ac.state
        thr, elev, ail, rud = ap.step(
            n_x=1.0, n_n=1.0, mu=0.0, dt=DT,
            n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
            airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
        )
        ac.set_controls(throttle=thr, elevator=trim_elev, aileron=0.0, rudder=0.0)
        ac.run()
        all_t.append(t)
        all_spd.append(s["airspeed_mps"])
        all_thr.append(thr)
        t += DT

    all_t = np.array(all_t)
    all_spd = np.array(all_spd)
    all_thr = np.array(all_thr)

    # Throttle oscillation metric: std of last 3s
    ss_samples = int(3.0 / DT)
    throttle_rms = float(np.std(all_thr[-ss_samples:]))

    return {
        "kp": kp,
        "final_speed": float(all_spd[-1]),
        "throttle_rms": throttle_rms,
        "t": all_t, "spd": all_spd, "thr": all_thr,
    }


def main():
    os.makedirs("results/phase2_tuning", exist_ok=True)
    print("=" * 60)
    print("PHASE 2c: Speed Channel Tuning")
    print("=" * 60)

    gain_sets = [(0.005, "kp=0.005"), (0.010, "kp=0.010"),
                 (0.015, "kp=0.015"), (0.020, "kp=0.020")]

    results = []
    for kp, label in gain_sets:
        r = _run_speed_test(kp, label)
        results.append(r)
        ok = r["throttle_rms"] < 0.05
        print(f"    final_speed={r['final_speed']:.0f} m/s  "
              f"throttle_rms={r['throttle_rms']:.4f}  {'OK' if ok else 'OSCILLATING'}")

    # Best: highest kp without oscillation
    best = None
    for r in results:
        if r["throttle_rms"] < 0.05:
            best = r
        else:
            break
    if best is None:
        best = results[0]

    print(f"\n  Best: kp={best['kp']:.3f} "
          f"(final_speed={best['final_speed']:.0f} m/s, "
          f"throttle_rms={best['throttle_rms']:.4f})")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(best["t"], best["spd"], "b-", lw=1.5)
    ax1.axhline(y=TARGET_SPEED, color="k", ls="--", alpha=0.5, label="target")
    ax1.set_ylabel("Airspeed (m/s)")
    ax1.set_title(f"Speed Channel — kp={best['kp']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(best["t"], best["thr"], "g-", lw=1.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Throttle")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/phase2_tuning/speed_tuning.png", dpi=150, bbox_inches="tight")
    print("  Plot saved → results/phase2_tuning/speed_tuning.png")
    plt.close("all")

    print(f"\n  Recommended: speed_kp={best['kp']:.3f}")


if __name__ == "__main__":
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
