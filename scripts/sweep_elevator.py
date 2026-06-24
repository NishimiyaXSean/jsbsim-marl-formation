"""Phase 1: Open-loop elevator sweep for speed-to-trim characterisation.

Sweeps elevator at 3 reference speeds (150, 200, 250 m/s), recording
steady-state n_z_g, pitch rate, and alpha to build a speed-to-trim lookup
table.  All PIDs are bypassed — direct control surface commands.

Outputs:
    data/trim_table.json          — speed-to-elevator-trim lookup
    results/phase1_sweep/         — matplotlib plots

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/sweep_elevator.py
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
from src.utils.units import kts_to_mps

os.environ.setdefault("JSBSIM_DEBUG", "0")

# ── Sweep parameters ──────────────────────────────────────────────────────

DT = 1.0 / 60.0
SETTLE_TIME = 3.0          # seconds at trim before collecting data
WARMUP_STEPS = 120          # additional warmup steps (JSBSim IC transient)
THROTTLE_FIXED = 0.80
INIT_ALT_FT = 9842          # 3000 m
INIT_HEADING_DEG = 90.0
MAX_ALPHA_DEG = 25.0        # safety: skip if alpha exceeds this

# Speed reference points (knots → m/s)
SPEED_KTS = [300, 400, 500]   # 154, 206, 257 m/s (rounded to 150/200/250 below)

# Elevator sweep range
ELEVATOR_SWEEP = np.arange(-0.20, 0.22, 0.02).tolist()  # 21 points

# ── Helpers ────────────────────────────────────────────────────────────────


def _find_trim_elevator(ac: Aircraft, speed_kts: float,
                        max_iters: int = 30) -> float:
    """Binary search for the elevator that gives n_z_g ≈ -1.0 at level flight.

    Returns the best elevator command (normalised [-1, 1]).
    """
    lo, hi = -0.5, 0.3  # reasonable range for 1G trim
    best_elev = -0.05    # fallback
    best_error = 999.0

    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                 heading_deg=INIT_HEADING_DEG, speed_kts=speed_kts)
        # Warmup
        for _ in range(60):
            ac.set_controls(throttle=THROTTLE_FIXED, elevator=mid, aileron=0.0, rudder=0.0)
            ac.run()
        # Measure
        nz_sum = 0.0
        for _ in range(120):
            ac.set_controls(throttle=THROTTLE_FIXED, elevator=mid, aileron=0.0, rudder=0.0)
            ac.run()
            nz_sum += ac.state["n_z_g"]
        nz_avg = nz_sum / 120.0
        error = abs(nz_avg + 1.0)  # target: n_z_g = -1.0 (1G level flight)

        if error < best_error:
            best_error = error
            best_elev = mid

        # n_z_g < -1.0 → too much pull → elevator too negative → need more positive
        if nz_avg < -1.0:
            lo = mid
        else:
            hi = mid

        if error < 0.005:  # converged
            break

    return float(best_elev)


def _run_step(ac: Aircraft, elevator_cmd: float, duration_s: float
              ) -> dict:
    """Run a single elevator step and return steady-state metrics."""
    nz_vals = []
    q_vals = []
    alpha_vals = []
    speed_vals = []
    alt_vals = []

    steps = int(duration_s / DT)
    for _ in range(steps):
        ac.set_controls(throttle=THROTTLE_FIXED, elevator=elevator_cmd,
                        aileron=0.0, rudder=0.0)
        ac.run()
        s = ac.state
        nz_vals.append(s["n_z_g"])
        q_vals.append(s["q_rps"])
        alpha_vals.append(s["alpha_deg"])
        speed_vals.append(s["airspeed_mps"])
        alt_vals.append(s["alt_m"])

    # Steady-state: mean of last 1.0s
    ss_samples = int(1.0 / DT)
    return {
        "n_z_g_mean": float(np.mean(nz_vals[-ss_samples:])),
        "n_z_g_std": float(np.std(nz_vals[-ss_samples:])),
        "q_rps_mean": float(np.mean(q_vals[-ss_samples:])),
        "alpha_deg_mean": float(np.mean(alpha_vals[-ss_samples:])),
        "airspeed_mean": float(np.mean(speed_vals[-ss_samples:])),
        "alt_start": alt_vals[0],
        "alt_end": alt_vals[-1],
        "alpha_max": float(np.max(np.abs(alpha_vals))),
    }


# ── Main sweep ─────────────────────────────────────────────────────────────


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/phase1_sweep", exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Open-Loop Elevator Sweep")
    print("=" * 60)

    all_results = {}   # speed_kts → {elevator_cmd → metrics}
    trim_entries = []  # for JSON

    for speed_kts in SPEED_KTS:
        speed_mps = kts_to_mps(speed_kts)
        print(f"\n{'─' * 60}")
        print(f"Speed: {speed_kts} kts ({speed_mps:.0f} m/s)")
        print(f"{'─' * 60}")

        # 1. Find trim elevator
        trim_elev = _find_trim_elevator(Aircraft(), speed_kts)
        trim_ac = Aircraft()
        trim_ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                      heading_deg=INIT_HEADING_DEG, speed_kts=speed_kts)
        for _ in range(WARMUP_STEPS):
            trim_ac.set_controls(throttle=THROTTLE_FIXED, elevator=trim_elev,
                                 aileron=0.0, rudder=0.0)
            trim_ac.run()
        trim_state = trim_ac.state
        print(f"  Trim elevator: {trim_elev:+.4f}")
        print(f"  At trim: n_z_g={trim_state['n_z_g']:.3f}, "
              f"alpha={trim_state['alpha_deg']:.1f}°, "
              f"airspeed={trim_state['airspeed_mps']:.0f} m/s")

        trim_entries.append({
            "speed_kts": speed_kts,
            "speed_mps": round(speed_mps, 1),
            "elevator_trim": round(trim_elev, 4),
            "alpha_deg": round(trim_state["alpha_deg"], 2),
            "n_z_g": round(trim_state["n_z_g"], 3),
            "airspeed_mps": round(trim_state["airspeed_mps"], 1),
        })

        # 2. Sweep elevator
        speed_results = {}
        safe_elevators = []
        for elev_cmd in ELEVATOR_SWEEP:
            ac = Aircraft()
            ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                     heading_deg=INIT_HEADING_DEG, speed_kts=speed_kts)
            # Warmup at trim
            for _ in range(WARMUP_STEPS):
                ac.set_controls(throttle=THROTTLE_FIXED, elevator=trim_elev,
                                aileron=0.0, rudder=0.0)
                ac.run()

            metrics = _run_step(ac, elev_cmd, 5.0)

            if metrics["alpha_max"] < MAX_ALPHA_DEG:
                safe_elevators.append(elev_cmd)
                speed_results[elev_cmd] = metrics
                print(f"  elev={elev_cmd:+.2f}: n_z={metrics['n_z_g_mean']:+.3f}  "
                      f"q={metrics['q_rps_mean']:+.4f} rad/s  "
                      f"alpha={metrics['alpha_deg_mean']:.1f}°  "
                      f"V={metrics['airspeed_mean']:.0f} m/s  "
                      f"dh={metrics['alt_end'] - metrics['alt_start']:+.0f}m")
            else:
                print(f"  elev={elev_cmd:+.2f}: SKIPPED (alpha_max={metrics['alpha_max']:.1f}° > {MAX_ALPHA_DEG}°)")

        all_results[speed_kts] = speed_results

    # ── Save trim table ───────────────────────────────────────────────────
    trim_table = {
        "description": "F-16 speed-to-elevator-trim lookup from Phase 1 sweep",
        "throttle_fixed": THROTTLE_FIXED,
        "init_alt_ft": INIT_ALT_FT,
        "ref_speed_mps": kts_to_mps(SPEED_KTS[1]),   # 400 kts reference
        "ref_elevator": trim_entries[1]["elevator_trim"],
        "ref_throttle": THROTTLE_FIXED,
        "entries": trim_entries,
        "1_v2_fit": {},  # populated below
    }

    # Validate 1/V^2 fit
    speeds = np.array([e["speed_mps"] for e in trim_entries])
    elevators = np.array([e["elevator_trim"] for e in trim_entries])
    ref_V = trim_table["ref_speed_mps"]
    ref_E = trim_table["ref_elevator"]
    predicted = ref_E * (ref_V / speeds) ** 2
    rms_error = float(np.sqrt(np.mean((elevators - predicted) ** 2)))
    trim_table["1_v2_fit"] = {
        "formula": "ref_elevator * (ref_speed_mps / V)^2",
        "rms_error_elevator_units": round(rms_error, 6),
        "fit_quality": "GOOD" if rms_error < 0.01 else "POOR — consider 2D lookup",
    }
    print(f"\n  1/V^2 fit RMS error: {rms_error:.6f} elevator units "
          f"({'OK' if rms_error < 0.01 else 'BAD'})")

    with open("data/trim_table.json", "w") as f:
        json.dump(trim_table, f, indent=2)
    print("  Trim table saved → data/trim_table.json")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {SPEED_KTS[0]: "blue", SPEED_KTS[1]: "green", SPEED_KTS[2]: "red"}
    for speed_kts in SPEED_KTS:
        speed_results = all_results[speed_kts]
        elevs = sorted(speed_results.keys())
        nz = [-speed_results[e]["n_z_g_mean"] for e in elevs]   # perceived G
        q = [speed_results[e]["q_rps_mean"] for e in elevs]
        c = colors[speed_kts]
        label = f"{speed_kts} kts ({kts_to_mps(speed_kts):.0f} m/s)"
        ax1.plot(elevs, nz, "o-", color=c, lw=2, label=label)
        ax2.plot(elevs, q, "o-", color=c, lw=2, label=label)

    for ax, ylabel, title in [
        (ax1, "Perceived G (-n_z_g)", "Elevator vs. Steady-State G-load"),
        (ax2, "Pitch Rate q (rad/s)", "Elevator vs. Steady-State Pitch Rate"),
    ]:
        ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
        ax.axvline(x=-0.05, color="gray", ls=":", alpha=0.3, label="hardcoded trim")
        ax.set_xlabel("Elevator Command (normalised)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/phase1_sweep/elevator_sweep.png", dpi=150, bbox_inches="tight")
    print("  Plots saved → results/phase1_sweep/elevator_sweep.png")
    plt.close("all")

    print(f"\nPhase 1 complete. "
          f"{'1/V^2 fit VALID' if rms_error < 0.01 else '1/V^2 fit MARGINAL — consider 2D table'}")


if __name__ == "__main__":
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
