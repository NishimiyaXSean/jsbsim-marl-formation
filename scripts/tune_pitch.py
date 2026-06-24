"""Phase 2: Single-channel pitch tuning (PD-only, no integral).

Lock aileron at 0, rudder at 0, throttle at 0.80.
Command G-load step sequence: 1G→3G→1G→5G→1G.
Tune nz_kp and nz_kd for fast G-tracking without overshoot.

JSBSim inner-loop G-load PID has kp=0.3 — cascaded outer loop must be
conservatively tuned to avoid oscillation.

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/tune_pitch.py
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
INIT_HEADING_DEG = 90.0
SPEED_KTS = 400

# G-load step sequence (positive only — no negative G pushover)
G_STEPS = [1.0, 3.0, 1.0, 5.0, 1.0]


def _load_trim_for_speed(speed_mps: float) -> float:
    try:
        with open("data/trim_table.json") as f:
            data = json.load(f)
        ref_V = data["ref_speed_mps"]
        ref_E = data["ref_elevator"]
        return ref_E * (ref_V / max(speed_mps, 80.0)) ** 2
    except FileNotFoundError:
        return -0.05 * (176.0 / max(speed_mps, 80.0)) ** 2


def _run_g_test(kp: float, kd: float, label: str) -> dict:
    """Run G-step sequence with given Nz gains, locked wings-level."""
    config = BFMAutopilotConfig(nz_kp=kp, nz_kd=kd, nz_ki=0.0)
    ap = BFMAutopilot(config)

    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
             heading_deg=INIT_HEADING_DEG, speed_kts=SPEED_KTS)

    all_t = []
    all_nz = []
    all_cmd = []
    all_elev = []
    all_alpha = []

    t = 0.0
    speed_mps = ac.state["airspeed_mps"]
    trim_elev = _load_trim_for_speed(speed_mps)
    print(f"  [{label}] trim_elev={trim_elev:+.4f} @ {speed_mps:.0f} m/s, "
          f"kp={kp:.2f}, kd={kd:.3f}")

    # Warmup
    for _ in range(int(WARMUP_S / DT)):
        s = ac.state
        thr, elev, ail, rud = ap.step(
            n_x=0.0, n_n=1.0, mu=0.0, dt=DT,
            n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
            airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
        )
        ac.set_controls(throttle=THROTTLE, elevator=elev, aileron=0.0, rudder=0.0)
        ac.run()

    for target_g in G_STEPS:
        for _ in range(int(HOLD_S / DT)):
            s = ac.state
            thr, elev, ail, rud = ap.step(
                n_x=0.0, n_n=target_g, mu=0.0, dt=DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
            )
            ac.set_controls(throttle=THROTTLE, elevator=elev, aileron=0.0, rudder=0.0)
            ac.run()
            all_t.append(t)
            all_nz.append(-s["n_z_g"])   # perceived G
            all_cmd.append(target_g)
            all_elev.append(elev)
            all_alpha.append(s["alpha_deg"])
            t += DT

    all_t = np.array(all_t)
    all_nz = np.array(all_nz)
    all_cmd = np.array(all_cmd)

    # Metrics for 1G→3G step
    step_start = int(WARMUP_S / DT)
    step_end = step_start + int(HOLD_S / DT)
    seg_nz = all_nz[step_start:step_end]
    seg_cmd = all_cmd[step_start:step_end]

    try:
        t10_idx = np.where(seg_nz >= 1.0 + 0.2 * 2.0)[0][0]  # 10% of 2G delta
        t90_idx = np.where(seg_nz >= 1.0 + 0.9 * 2.0)[0][0]  # 90%
        rise_time = (t90_idx - t10_idx) * DT
    except IndexError:
        rise_time = float("inf")

    overshoot = max(0.0, np.max(seg_nz) - 3.0)
    ss_error = abs(np.mean(seg_nz[-int(1.0 / DT):]) - 3.0)

    # Alpha check
    max_alpha = float(np.max(all_alpha))

    return {
        "kp": kp, "kd": kd,
        "rise_time": rise_time,
        "overshoot": overshoot,
        "ss_error": ss_error,
        "max_alpha": max_alpha,
        "t": all_t, "nz": all_nz, "cmd": all_cmd, "elev": all_elev, "alpha": all_alpha,
    }


def main():
    os.makedirs("results/phase2_tuning", exist_ok=True)
    print("=" * 60)
    print("PHASE 2b: Pitch (Nz) Channel Tuning")
    print("=" * 60)

    gain_sets = [
        (0.08, 0.0,   "kp=0.08 (baseline)"),
        (0.10, 0.0,   "kp=0.10"),
        (0.12, 0.0,   "kp=0.12"),
        (0.15, 0.0,   "kp=0.15 (V10 default)"),
        (0.15, 0.010, "kp=0.15, kd=0.010"),
        (0.18, 0.0,   "kp=0.18"),
        (0.18, 0.012, "kp=0.18, kd=0.012"),
        (0.20, 0.0,   "kp=0.20"),
        (0.20, 0.015, "kp=0.20, kd=0.015"),
    ]

    best = None
    results = []
    for kp, kd, label in gain_sets:
        r = _run_g_test(kp, kd, label)
        results.append(r)
        ok = (r["rise_time"] < 0.5 and r["overshoot"] < 0.3
              and r["max_alpha"] < 20.0)
        print(f"    rise={r['rise_time']:.3f}s  overshoot={r['overshoot']:.2f}G  "
              f"ss_err={r['ss_error']:.2f}G  max_alpha={r['max_alpha']:.1f}°  "
              f"{'OK' if ok else 'FAIL'}")
        if ok and (best is None or r["rise_time"] < best["rise_time"]):
            best = r

    if best is None:
        best = min(results, key=lambda r: r["rise_time"] + 5 * r["overshoot"])
        print(f"\n  No perfect set — best compromise: kp={best['kp']}, kd={best['kd']}")
    else:
        print(f"\n  Best: kp={best['kp']}, kd={best['kd']} "
              f"(rise={best['rise_time']:.3f}s, overshoot={best['overshoot']:.2f}G)")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(best["t"], best["cmd"], "k--", lw=1.5, alpha=0.5, label="Command")
    axes[0].plot(best["t"], best["nz"], "b-", lw=1.5, label=f"n_z (kp={best['kp']}, kd={best['kd']})")
    axes[0].set_ylabel("Perceived G")
    axes[0].set_title("Pitch Channel Tuning — Best Result")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(best["t"], best["elev"], "r-", lw=1.0)
    axes[1].set_ylabel("Elevator Cmd")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(best["t"], best["alpha"], "orange", lw=1.0)
    axes[2].axhline(y=20, color="red", ls="--", alpha=0.5, label="alpha limit")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Alpha (deg)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/phase2_tuning/pitch_tuning.png", dpi=150, bbox_inches="tight")
    print("  Plot saved → results/phase2_tuning/pitch_tuning.png")
    plt.close("all")

    print(f"\n  Recommended: nz_kp={best['kp']:.3f}, nz_kd={best['kd']:.3f}")


if __name__ == "__main__":
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
