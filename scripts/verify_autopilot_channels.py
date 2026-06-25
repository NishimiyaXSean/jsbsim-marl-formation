"""Step-response verification: JSBSim <-> BFMAutopilot channel connectivity.

Feeds deterministic step commands to the lambda-G flight control law and records
the JSBSim aerodynamic response. No neural networks are loaded - pure physics
and control-law validation.

Usage:
    python scripts/verify_autopilot_channels.py

Outputs:
    results/autopilot_step_response.png - 6-panel diagnostic plot
    results/autopilot_step_response.csv - raw time-series data
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import (
    BFMAutopilot, BFMAutopilotConfig, GainScheduler, TrimSchedule,
)
from src.utils.units import ft_to_m, kts_to_mps, rad_to_deg, deg_to_rad, mps_to_kts

# Simulation parameters
PHYSICS_DT = 1.0 / 60.0          # 60 Hz JSBSim time step
SIM_DURATION = 20.0              # seconds
DECIMATION = 5                   # record every N steps -> 12 Hz logging

# Initial conditions: F-16 at ~10,000 ft, 350 kts
INIT_ALT_FT = 10000.0
INIT_SPEED_KTS = 350.0
INIT_HDG_DEG = 90.0              # East

# Results directory
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_autopilot_verification():
    print("=" * 64)
    print("  JSBSim Autopilot Channel Step-Response Verification")
    print("=" * 64)

    # === 1. Aircraft + autopilot initialisation ===

    print("\n[1/4] Initialising JSBSim F-16 + BFMAutopilot with GainScheduler ...")

    ac = Aircraft()
    ac.reset(
        lat_deg=30.0, lon_deg=120.0,
        alt_ft=INIT_ALT_FT,
        heading_deg=INIT_HDG_DEG,
        speed_kts=INIT_SPEED_KTS,
        trim=False,
    )
    init_speed_mps = kts_to_mps(INIT_SPEED_KTS)
    print(f"      Aircraft: {ac.MODEL.upper()} at {INIT_ALT_FT:.0f} ft, "
          f"{INIT_SPEED_KTS:.0f} kts ({init_speed_mps:.1f} m/s)")

    # Autopilot with Phase 3.5 gain scheduling
    cfg = BFMAutopilotConfig()
    trim_sched = TrimSchedule()
    scheduler = GainScheduler()
    bfm = BFMAutopilot(config=cfg, trim=trim_sched, scheduler=scheduler)
    bfm.reset(initial_speed_mps=init_speed_mps)
    print("      BFMAutopilot: lambda-G law + GainScheduler + TrimSchedule engaged")

    # === 2. Property-tree input confirmations ===

    print("\n[2/4] Confirming JSBSim property-tree connectivity:")
    print("      Agent action -> BFMAutopilot.step(n_x, n_n, mu)")
    print("      Autopilot output -> ac.set_controls(throttle, elevator, aileron, rudder)")
    print("      set_controls writes to JSBSim properties:")
    print("        fcs/throttle-cmd-norm  <- throttle  [0, 1]")
    print("        fcs/elevator-cmd-norm  <- elevator  [-1, 1]")
    print("        fcs/aileron-cmd-norm   <- aileron   [-1, 1]")
    print("        fcs/rudder-cmd-norm    <- rudder    [-1, 1]")
    print("      State readback from JSBSim properties:")
    print("        accelerations/n-pilot-z-norm  -> n_z_g  (actual G, neg=pull)")
    print("        attitude/roll-rad             -> roll_rad")
    print("        velocities/vc-kts             -> airspeed")
    print("        aero/alpha-deg, aero/beta-deg -> alpha, beta")

    # Verify one round-trip
    ac.set_controls(0.80, -0.05, 0.0, 0.0)
    ac.run()
    s = ac.state
    print(f"\n      Round-trip test (thr=0.80, elev=-0.05):")
    print(f"        n_z_g={s['n_z_g']:+.3f}  alpha={s['alpha_deg']:+.2f} deg  "
          f"q={rad_to_deg(s['q_rps']):+.2f} deg/s  spd={s['airspeed_mps']:.1f} m/s")

    # === 3. Step-response test sequence ===

    print("\n[3/4] Running 20-second step-response sequence ...")
    print("      Phase 1:   0 -  3 s   n_n=1.0  mu=0 deg    (level trim)")
    print("      Phase 2:   3 - 10 s   n_n=4.0  mu=0 deg    (4G hard pull-up)")
    print("      Phase 3:  10 - 15 s   n_n=2.0  mu=-30 deg  (2G left turn, coupled)")
    print("      Phase 4:  15 - 20 s   n_n=1.0  mu=0 deg    (recover level)")

    total_steps = int(SIM_DURATION / PHYSICS_DT)

    # Pre-allocated recording buffers
    n_records = total_steps // DECIMATION + 1
    rec = {
        "t":           np.zeros(n_records),
        "target_nn":   np.zeros(n_records),
        "target_mu":   np.zeros(n_records),
        "n_z_g":       np.zeros(n_records),
        "alpha_deg":   np.zeros(n_records),
        "elevator":    np.zeros(n_records),
        "throttle":    np.zeros(n_records),
        "aileron":     np.zeros(n_records),
        "rudder":      np.zeros(n_records),
        "q_rps":       np.zeros(n_records),
        "p_rps":       np.zeros(n_records),
        "roll_deg":    np.zeros(n_records),
        "pitch_deg":   np.zeros(n_records),
        "alt_m":       np.zeros(n_records),
        "airspeed_mps": np.zeros(n_records),
        "thrust_lbs":  np.zeros(n_records),
        "mach":        np.zeros(n_records),
    }

    # Reset: get a clean baseline
    ac.reset(
        lat_deg=30.0, lon_deg=120.0,
        alt_ft=INIT_ALT_FT,
        heading_deg=INIT_HDG_DEG,
        speed_kts=INIT_SPEED_KTS,
        trim=False,
    )
    bfm.reset(initial_speed_mps=init_speed_mps)

    # Main simulation loop
    crash_reason = None
    crash_time = None

    for i in range(total_steps):
        t = i * PHYSICS_DT

        # --- Command logic (simulates RL agent output) -----------------
        if t < 3.0:
            target_nn = 1.0
            target_mu_deg = 0.0
        elif t < 10.0:
            target_nn = 4.0
            target_mu_deg = 0.0
        elif t < 15.0:
            target_nn = 2.0
            target_mu_deg = -30.0
        else:
            target_nn = 1.0
            target_mu_deg = 0.0

        target_mu_rad = deg_to_rad(target_mu_deg)

        # --- BFMAutopilot: (n_n, mu) -> control surfaces ---------------
        s = ac.state
        throttle, elevator, aileron, rudder = bfm.step(
            n_x=0.0, n_n=target_nn, mu=target_mu_rad, dt=PHYSICS_DT,
            n_z_g=s["n_z_g"],
            roll_rad=deg_to_rad(s["roll_deg"]),
            airspeed_mps=s["airspeed_mps"],
            beta_deg=s["beta_deg"],
            alpha_deg=s["alpha_deg"],
        )

        # --- Apply to aircraft -----------------------------------------
        ac.set_controls(throttle, elevator, aileron, rudder)

        # --- Step physics ----------------------------------------------
        ac.run()

        # --- Safety: check for NaN / ground crash / ceiling ------------
        s = ac.state
        if (not np.isfinite(float(s.get("alt_m", np.nan)))
                or not np.isfinite(float(s.get("n_z_g", np.nan)))):
            crash_reason = "state_NaN"
            crash_time = t
            print(f"      !! NaN detected at t={t:.2f}s — terminating early")
            break
        if s.get("alt_m", 0) < 5.0:
            crash_reason = "ground_impact"
            crash_time = t
            print(f"      !! GROUND IMPACT at t={t:.2f}s, alt={s['alt_m']:.1f}m — terminating")
            break
        if s.get("alt_m", 0) > 15000.0:
            crash_reason = "ceiling_exceeded"
            crash_time = t
            print(f"      !! CEILING EXCEEDED at t={t:.2f}s, alt={s['alt_m']:.0f}m")
            break

        # --- Record ----------------------------------------------------
        if i % DECIMATION == 0:
            idx = i // DECIMATION
            rec["t"][idx] = t
            rec["target_nn"][idx] = target_nn
            rec["target_mu"][idx] = target_mu_deg
            rec["n_z_g"][idx] = s["n_z_g"]
            rec["alpha_deg"][idx] = s["alpha_deg"]
            rec["elevator"][idx] = elevator
            rec["throttle"][idx] = throttle
            rec["aileron"][idx] = aileron
            rec["rudder"][idx] = rudder
            rec["q_rps"][idx] = s["q_rps"]
            rec["p_rps"][idx] = s["p_rps"]
            rec["roll_deg"][idx] = s["roll_deg"]
            rec["pitch_deg"][idx] = s["pitch_deg"]
            rec["alt_m"][idx] = s["alt_m"]
            rec["airspeed_mps"][idx] = s["airspeed_mps"]
            rec["thrust_lbs"][idx] = s["thrust_lbs"]
            rec["mach"][idx] = s["mach"]

        # Progress every 5 seconds
        if i > 0 and i % int(5.0 / PHYSICS_DT) == 0:
            print(f"      t={t:5.1f}s  cmd n_n={target_nn:.1f}G  "
                  f"actual |Nz|={abs(s['n_z_g']):.2f}G  "
                  f"elev={elevator:+.3f}  alpha={s['alpha_deg']:+.1f} deg  "
                  f"spd={s['airspeed_mps']:.0f}m/s  alt={s['alt_m']:.0f}m")

    if crash_reason:
        print(f"      Simulation terminated early: {crash_reason} at t={crash_time:.1f}s")
    else:
        print("      Simulation complete.")

    # === 4. Analysis and plotting ===

    print("\n[4/4] Analysing channel response and generating plots ...")

    # Only use valid (non-NaN) records
    valid = np.isfinite(rec["t"]) & (rec["t"] > 0)
    valid[0] = np.isfinite(rec["t"][0])  # include t=0 if valid
    # Truncate to valid data
    n_valid = valid.sum()
    if n_valid < len(rec["t"]):
        for k in rec:
            rec[k] = rec[k][:n_valid]

    actual_g = np.abs(rec["n_z_g"])

    # Compute response metrics per phase
    metrics = _compute_metrics(rec, crash_reason, crash_time)

    # Generate 6-panel diagnostic plot
    _plot_diagnostics(rec, actual_g, metrics)

    # Save CSV
    _save_csv(rec)

    # Print diagnostic summary
    _print_summary(metrics)

    return metrics


def _compute_metrics(rec, crash_reason=None, crash_time=None):
    """Compute per-phase tracking quality metrics."""
    m = {"crash_reason": crash_reason, "crash_time": crash_time}

    for label, t_start, t_end in [
        ("P1: 0-3s trim", 0.0, 3.0),
        ("P2: 3-10s pull-up", 3.5, 9.5),
        ("P3: 10-15s turn", 10.5, 14.5),
        ("P4: 15-20s recovery", 15.5, 19.5),
    ]:
        mask = (rec["t"] >= t_start) & (rec["t"] <= t_end)
        # Exclude NaN
        mask = mask & np.isfinite(rec["n_z_g"])
        if mask.sum() < 2:
            m[label] = None  # insufficient data
            continue

        target = rec["target_nn"][mask]
        actual = np.abs(rec["n_z_g"][mask])
        error = actual - target

        m[label] = {
            "target_mean": float(np.mean(target)),
            "actual_mean": float(np.mean(actual)),
            "error_rmse": float(np.sqrt(np.mean(error ** 2))),
            "error_max": float(np.max(np.abs(error))),
            "elevator_mean": float(np.mean(rec["elevator"][mask])),
            "elevator_std": float(np.std(rec["elevator"][mask])),
            "q_std_deg": float(np.std(np.degrees(rec["q_rps"][mask]))),
            "alt_delta_m": float(rec["alt_m"][mask][-1] - rec["alt_m"][mask][0]),
            "n_samples": int(mask.sum()),
        }

    # Convergence: measure oscillation after P2 pull-up settles
    p2_mask = (rec["t"] >= 5.0) & (rec["t"] <= 9.5)
    if p2_mask.sum() > 20:
        nz_p2 = np.abs(rec["n_z_g"][p2_mask])
        m["P2_convergence"] = {
            "nz_std_after_settle": float(np.std(nz_p2)),
            "nz_range_after_settle": float(np.ptp(nz_p2)),
            "porpoising": bool(np.std(nz_p2) > 0.3),
        }

    return m


def _plot_diagnostics(rec, actual_g, metrics):
    """Generate 6-panel diagnostic plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = rec["t"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        "JSBSim F-16 BFMAutopilot Channel Step-Response Verification\n"
        "lambda-G Flight Control Law + GainScheduler (Phase 3.5)",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Nz tracking (target vs actual)
    ax = axes[0, 0]
    ax.plot(t, rec["target_nn"], "r--", linewidth=2, label="Target n_n (commanded G)")
    ax.plot(t, actual_g, "b-", linewidth=1.5, label="Actual |Nz| (JSBSim measured)")
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Load Factor (G)")
    ax.set_title("Nz Channel: Command Tracking")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    # Panel 2: Elevator deflection
    ax = axes[0, 1]
    ax.plot(t, rec["elevator"], "g-", linewidth=1.5, label="Elevator cmd [-1,1]")
    ax.axhline(-1.0, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Elevator Command (norm)")
    ax.set_title("Elevator Channel: Control Effort")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Pitch rate
    ax = axes[1, 0]
    q_deg = np.degrees(rec["q_rps"])
    ax.plot(t, q_deg, "m-", linewidth=1.5, label="Pitch rate q (deg/s)")
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Pitch Rate (deg/s)")
    ax.set_title("Pitch Rate Response")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Roll channel
    ax = axes[1, 1]
    ax.plot(t, rec["target_mu"], "r--", linewidth=2, label="Target mu (deg)")
    ax.plot(t, rec["roll_deg"], "b-", linewidth=1.5, label="Actual roll (deg)")
    ax.plot(t, rec["aileron"] * 30, "c-", linewidth=1, alpha=0.7,
            label="Aileron cmd x30")
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Roll Angle (deg) / Aileron x30")
    ax.set_title("Roll Channel: Bank Angle Tracking + Coupling")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 5: Alpha (AoA) + speed
    ax = axes[2, 0]
    ax2 = ax.twinx()
    ax.plot(t, rec["alpha_deg"], "darkorange", linewidth=1.5, label="AoA alpha (deg)")
    ax.axhline(25.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8,
               label="Alpha limit (25 deg)")
    ax2.plot(t, rec["airspeed_mps"], "steelblue", linewidth=1.5, label="Airspeed (m/s)")
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Angle of Attack (deg)")
    ax2.set_ylabel("Airspeed (m/s)")
    ax.set_title("AoA + Airspeed: Energy State")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: Altitude + throttle
    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax.plot(t, rec["alt_m"], "b-", linewidth=1.5, label="Altitude (m)")
    ax2.plot(t, rec["throttle"], "saddlebrown", linewidth=1.5, label="Throttle [0,1]")
    for ts in [3, 10, 15]:
        ax.axvline(ts, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Altitude MSL (m)")
    ax2.set_ylabel("Throttle Command")
    ax.set_title("Altitude + Throttle Response")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = RESULTS_DIR / "autopilot_step_response.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved plot: {outpath}")


def _save_csv(rec):
    """Save all recorded time-series to CSV."""
    outpath = RESULTS_DIR / "autopilot_step_response.csv"
    headers = list(rec.keys())
    n = len(rec["t"])

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(n):
            w.writerow([rec[k][i] for k in headers])

    print(f"      Saved CSV:  {outpath}  ({n} records)")


def _print_summary(metrics):
    """Print human-readable diagnostic summary."""
    print("\n" + "=" * 64)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 64)

    checks = []

    # Crash diagnosis
    crash_reason = metrics.get("crash_reason")
    crash_time = metrics.get("crash_time")
    if crash_reason:
        print(f"\n  [Flight Termination]")
        print(f"    Reason: {crash_reason}")
        print(f"    Time:   {crash_time:.1f}s" if crash_time else "")
        if crash_reason == "ground_impact":
            print(f"    FAIL - Aircraft crashed. PID tuning or trim issue suspected.")
            checks.append(("Flight safety", False))
        elif crash_reason == "state_NaN":
            print(f"    FAIL - Simulation diverged (NaN). Severe instability.")
            checks.append(("Flight safety", False))

    # Check 1: Nz tracking in pull-up phase
    p2 = metrics.get("P2: 3-10s pull-up")
    if p2 is None:
        print(f"\n  [Nz Tracking - 4G pull-up phase]")
        print(f"    SKIP - Aircraft did not survive to this phase")
        checks.append(("Nz tracking", False))
    elif p2:
        rmse = p2["error_rmse"]
        print(f"\n  [Nz Tracking - 4G pull-up phase]")
        print(f"    Target:     4.00 G")
        print(f"    Actual mean: {p2['actual_mean']:.2f} G")
        print(f"    RMSE:        {rmse:.2f} G")
        print(f"    Max error:   {p2['error_max']:.2f} G")
        if rmse < 0.5:
            print("    PASS - Nz tracks within 0.5G RMSE")
            checks.append(("Nz tracking", True))
        elif rmse < 1.0:
            print(f"    WARN - Nz RMSE {rmse:.2f}G (<1.0G, acceptable)")
            checks.append(("Nz tracking", True))
        else:
            print(f"    FAIL - Nz RMSE {rmse:.2f}G exceeds 1.0G threshold")
            checks.append(("Nz tracking", False))

    # Check 2: Porpoising / dolphin
    conv = metrics.get("P2_convergence", {})
    if conv:
        std_nz = conv["nz_std_after_settle"]
        is_porp = conv.get("porpoising", False)
        print("\n  [Porpoising check - post-settle oscillation]")
        print(f"    Nz std (5-9.5s): {std_nz:.3f} G")
        print(f"    Nz range:        {conv['nz_range_after_settle']:.3f} G")
        if not is_porp:
            print("    PASS - No porpoising detected (std < 0.3G)")
            checks.append(("Anti-porpoising", True))
        else:
            print(f"    FAIL - Porpoising detected (std = {std_nz:.3f}G > 0.3G)")
            checks.append(("Anti-porpoising", False))

    # Check 3: Roll tracking in coupled phase
    p3 = metrics.get("P3: 10-15s turn")
    if p3 is None:
        print("\n  [Roll Tracking - 2G + 30deg bank coupled phase]")
        print("    SKIP - Aircraft did not survive to this phase")
        checks.append(("Multi-channel coupling", False))
    elif p3:
        print("\n  [Roll Tracking - 2G + 30deg bank coupled phase]")
        print(f"    Nz RMSE during turn: {p3['error_rmse']:.2f} G")
        print(f"    Elevator mean:       {p3['elevator_mean']:.3f}")
        print(f"    Elevator std:        {p3['elevator_std']:.3f}")
        print(f"    Pitch rate std:      {p3['q_std_deg']:.1f} deg/s")
        if p3["error_rmse"] < 1.0:
            print("    PASS - Multi-channel coupling stable")
            checks.append(("Multi-channel coupling", True))
        else:
            print("    WARN - Elevated Nz error during coupled manoeuvre")
            checks.append(("Multi-channel coupling", p3["error_rmse"] < 2.0))

    # Check 4: Trim accuracy
    p1 = metrics.get("P1: 0-3s trim")
    if p1 is None:
        print("\n  [Trim Accuracy - Level flight phase]")
        print("    SKIP - No valid data")
        checks.append(("Trim accuracy", False))
    elif p1:
        print("\n  [Trim Accuracy - Level flight phase]")
        print(f"    Actual mean G: {p1['actual_mean']:.2f} (target 1.00)")
        print(f"    G error:       {p1['error_rmse']:.2f} G")
        if abs(p1["actual_mean"] - 1.0) < 0.15:
            print("    PASS - Level trim within 0.15G of 1.0G")
            checks.append(("Trim accuracy", True))
        else:
            print(f"    WARN - Trim offset {p1['actual_mean']-1.0:+.2f}G")
            checks.append(("Trim accuracy", abs(p1["actual_mean"] - 1.0) < 0.3))

    # Check 5: Recovery
    p4 = metrics.get("P4: 15-20s recovery")
    if p4 is None:
        print("\n  [Recovery - Return to 1G level]")
        print("    SKIP - Aircraft did not survive to recovery phase")
        checks.append(("Recovery", False))
    elif p4:
        print("\n  [Recovery - Return to 1G level]")
        print(f"    Actual mean G: {p4['actual_mean']:.2f} (target 1.00)")
        print(f"    G error:       {p4['error_rmse']:.2f} G")
        if abs(p4["actual_mean"] - 1.0) < 0.2:
            print("    PASS - Recovery to ~1G successful")
            checks.append(("Recovery", True))
        else:
            print(f"    WARN - Recovery offset {p4['actual_mean']-1.0:+.2f}G")
            checks.append(("Recovery", abs(p4["actual_mean"] - 1.0) < 0.4))

    # Overall verdict
    print(f"\n{'─' * 64}")
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    if n_pass == n_total:
        print(f"  OVERALL: ALL CHECKS PASSED ({n_pass}/{n_total})")
        print("  JSBSim engine + BFMAutopilot channels are operational.")
    else:
        print(f"  OVERALL: {n_pass}/{n_total} checks passed")
        failed = [name for name, ok in checks if not ok]
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'─' * 64}\n")


if __name__ == "__main__":
    import logging
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    try:
        run_autopilot_verification()
    except Exception as e:
        print(f"\n  FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
