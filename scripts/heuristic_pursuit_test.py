"""Heuristic Pure Pursuit — discrete BFM actions driven by ATA logic.

Places an F-16 pursuer 2 km behind a straight-flying target and uses a
simple rule-based policy to select from the 9-action pursuit set:

    ATA < -5 deg  → Action 4 (Turn Left)
    ATA > +5 deg  → Action 3 (Turn Right)
    |ATA| < 5 deg → Action 1 (Accelerate)

Each action is held for a minimum of 1.5 s (macro-action) to prevent
high-frequency switching that causes departure.  No neural networks —
pure deterministic pursuit logic.

Outputs:
    results/heuristic_pursuit/
        pursuit_trajectory.png     — 2D top-down + altitude profile
        pursuit.acmi               — Tacview 3D trajectory
        pursuit_metrics.csv        — time-series data

Usage:
    python scripts/heuristic_pursuit_test.py
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.bfm_actions import PURSUIT_ACTIONS, describe_pursuit_action
from src.utils.geometry import compute_los, compute_tactical_angles, compute_forward_vector
from src.utils.units import m_to_ft, mps_to_kts, kts_to_mps, deg_to_rad, rad_to_deg

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

# ── Scenario parameters ─────────────────────────────────────────────────
DT = 1.0 / 60.0
TARGET_SPD_KTS = 300.0          # slower target — easier to catch
TARGET_ALT_FT = 10000.0
TARGET_HDG_DEG = 90.0
PURSUER_SPD_KTS = 400.0
PURSUER_ALT_FT = 10000.0
SEPARATION_M = 1500.0          # pursuer starts 1.5 km behind
MAX_TIME_S = 90.0
INTERCEPT_RADIUS_M = 100.0
MIN_ACTION_HOLD_S = 1.0        # slightly faster tactical switching
ATA_DEADBAND_DEG = 5.0
ALT_GUARD_MARGIN_M = 800.0     # if pursuer climbs this far above AND |ATA|<30 -> Descend

# ── Results directory ────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "heuristic_pursuit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _select_action(ata_deg: float, dist_m: float) -> int:
    """Choose a BFM pursuit action based on ATA and range.

    Three regimes:
    - Far off-boresight (|ATA| > 30°): aggressive turn toward target
    - Moderate offset (5° < |ATA| < 30°): gentle correction (coast
      or slight turn — use Accelerate which keeps wings level and
      lets the aircraft gradually converge)
    - On-target (|ATA| < 5°): Accelerate to close range
    - Close range (< 500m): Decelerate to avoid overshoot
    """
    if dist_m < 500:
        return 2   # Decelerate — prevent overshoot
    if abs(ata_deg) > 30:
        return 4 if ata_deg < 0 else 3  # Aggressive turn
    if abs(ata_deg) > ATA_DEADBAND_DEG:
        return 1   # Accelerate (wings-level, gradual convergence)
    return 1       # Accelerate (on-target)


def run_pursuit() -> dict:
    """Run the heuristic pure-pursuit scenario.  Returns result dict."""

    # ── Create aircraft ────────────────────────────────────────────────
    target = Aircraft()
    target.reset(lat_deg=30.0, lon_deg=120.0,
                 alt_ft=TARGET_ALT_FT, heading_deg=TARGET_HDG_DEG,
                 speed_kts=TARGET_SPD_KTS)
    t_ned = np.array([0.0, 0.0, TARGET_ALT_FT * 0.3048], dtype=np.float64)
    target.position_ned = t_ned

    pursuer = Aircraft()
    pursuer.reset(lat_deg=30.0, lon_deg=120.0,
                  alt_ft=PURSUER_ALT_FT, heading_deg=TARGET_HDG_DEG,
                  speed_kts=PURSUER_SPD_KTS)
    p_ned = np.array([0.0, -SEPARATION_M, PURSUER_ALT_FT * 0.3048], dtype=np.float64)
    pursuer.position_ned = p_ned

    # ── Autopilot + envelope ───────────────────────────────────────────
    cfg = BFMAutopilotConfig()
    ap = BFMAutopilot(cfg, trim=TrimSchedule(), scheduler=GainScheduler())
    ap.reset(initial_speed_mps=kts_to_mps(PURSUER_SPD_KTS))
    env = FlightEnvelope(EnvelopeConfig())

    # ── Target: FlightController for clean altitude+speed+heading hold ─
    target_fc = FlightController()
    target_fc.reset()
    target_fc_target = FlightControlTargets(
        heading_deg=TARGET_HDG_DEG,
        altitude_m=TARGET_ALT_FT * 0.3048,
        speed_mps=kts_to_mps(TARGET_SPD_KTS),
    )

    # ── Warmup: 3 s level flight ──────────────────────────────────────
    for _ in range(180):
        # Pursuer
        ps = pursuer.state
        p_nx, p_nn, p_mu = env.step(0.0, 1.0, 0.0,
            speed_mps=ps["airspeed_mps"], alt_m=ps["alt_m"],
            vz_mps=ps["h_dot_fps"] * 0.3048,
            current_roll_rad=deg_to_rad(ps["roll_deg"]), dt=DT)
        p_thr, p_elev, p_ail, p_rud = ap.step(
            p_nx, p_nn, p_mu, DT,
            n_z_g=ps["n_z_g"], roll_rad=deg_to_rad(ps["roll_deg"]),
            airspeed_mps=ps["airspeed_mps"], beta_deg=ps["beta_deg"],
            alpha_deg=ps["alpha_deg"], q_rps=ps["q_rps"])
        pursuer.set_controls(throttle=p_thr, elevator=p_elev,
                             aileron=p_ail, rudder=p_rud)

        # Target: FlightController hold
        ts = target.state
        t_thr, t_elev, t_ail, t_rud = target_fc.compute(ts, target_fc_target, DT)
        target.set_controls(throttle=t_thr, elevator=t_elev,
                            aileron=t_ail, rudder=t_rud)

        pursuer.run()
        target.run()

        p_ned[0:2] += pursuer.velocity_ned[0:2] * DT
        p_ned[2] = pursuer.state["alt_m"]
        pursuer.position_ned = p_ned
        t_ned[0:2] += target.velocity_ned[0:2] * DT
        t_ned[2] = target.state["alt_m"]
        target.position_ned = t_ned

    # ── Pursuit phase ──────────────────────────────────────────────────
    print(f"{'Time':>6s} {'Action':>20s} {'Dist':>7s} {'ATA':>7s} "
          f"{'P_Alt':>7s} {'P_Spd':>6s} {'P_Hdg':>6s} {'P_Roll':>6s}")

    rec = {"t": [], "action": [], "dist": [], "ata": [],
           "p_alt": [], "p_spd": [], "p_hdg": [], "p_roll": [],
           "t_alt": [], "t_spd": [], "t_hdg": [],
           "p_n": [], "p_e": []}

    current_action = 1
    action_hold_remaining = 0.0
    min_dist = SEPARATION_M
    reason = "timeout"
    sim_time = 0.0
    step_count = 0
    last_print = -5.0

    while sim_time < MAX_TIME_S:
        step_count += 1
        sim_time = step_count * DT

        # ── LOS / ATA computation ────────────────────────────────────
        _, los_dir, dist = compute_los(p_ned, t_ned)
        ps = pursuer.state
        ts = target.state
        own_fwd = compute_forward_vector(np.array([
            deg_to_rad(ps["roll_deg"]),
            deg_to_rad(ps["pitch_deg"]),
            deg_to_rad(ps["yaw_deg"]),
        ]))
        tgt_fwd = compute_forward_vector(np.array([
            deg_to_rad(ts["roll_deg"]),
            deg_to_rad(ts["pitch_deg"]),
            deg_to_rad(ts["yaw_deg"]),
        ]))
        angles = compute_tactical_angles(own_fwd, tgt_fwd, los_dir)
        cos_ata = angles["cos_ata"]
        ata_deg = float(np.degrees(np.arccos(np.clip(cos_ata, -1.0, 1.0))))
        # Determine left/right sign via cross product
        fwd_2d = own_fwd[:2] / (np.linalg.norm(own_fwd[:2]) + 1e-8)
        los_2d = los_dir[:2] / (np.linalg.norm(los_dir[:2]) + 1e-8)
        cross_z = fwd_2d[0] * los_2d[1] - fwd_2d[1] * los_2d[0]
        if cross_z < 0:
            ata_deg = -ata_deg  # target is to the left

        # ── Action selection with minimum hold time ──────────────────
        if action_hold_remaining <= 0.0:
            # Altitude guard: only override when roughly on-target
            # (turning has tactical priority over altitude management)
            if (p_ned[2] > t_ned[2] + ALT_GUARD_MARGIN_M
                    and abs(ata_deg) < 30.0):
                new_action = 6  # Descend — bleed altitude while tracking
            else:
                new_action = _select_action(ata_deg, dist)
            if new_action != current_action:
                current_action = new_action
                action_hold_remaining = MIN_ACTION_HOLD_S
        action_hold_remaining -= DT

        n_x_raw, n_n_raw, mu_raw = PURSUIT_ACTIONS[current_action]

        # ── Pursuer step ─────────────────────────────────────────────
        ps = pursuer.state
        p_nx, p_nn, p_mu = env.step(n_x_raw, n_n_raw, mu_raw,
            speed_mps=ps["airspeed_mps"], alt_m=ps["alt_m"],
            vz_mps=ps["h_dot_fps"] * 0.3048,
            current_roll_rad=deg_to_rad(ps["roll_deg"]), dt=DT)
        p_thr, p_elev, p_ail, p_rud = ap.step(
            p_nx, p_nn, p_mu, DT,
            n_z_g=ps["n_z_g"], roll_rad=deg_to_rad(ps["roll_deg"]),
            airspeed_mps=ps["airspeed_mps"], beta_deg=ps["beta_deg"],
            alpha_deg=ps["alpha_deg"], q_rps=ps["q_rps"])
        pursuer.set_controls(throttle=p_thr, elevator=p_elev,
                             aileron=p_ail, rudder=p_rud)

        # ── Target step: FlightController hold ──────────────────────
        ts = target.state
        t_thr, t_elev, t_ail, t_rud = target_fc.compute(ts, target_fc_target, DT)
        target.set_controls(throttle=t_thr, elevator=t_elev,
                            aileron=t_ail, rudder=t_rud)

        pursuer.run()
        target.run()

        # ── Update NED ───────────────────────────────────────────────
        p_ned[0:2] += pursuer.velocity_ned[0:2] * DT
        p_ned[2] = pursuer.state["alt_m"]
        pursuer.position_ned = p_ned
        t_ned[0:2] += target.velocity_ned[0:2] * DT
        t_ned[2] = target.state["alt_m"]
        target.position_ned = t_ned

        if dist < min_dist:
            min_dist = dist

        # ── Recording (10 Hz) ────────────────────────────────────────
        if step_count % 6 == 0:
            rec["t"].append(sim_time)
            rec["action"].append(current_action)
            rec["dist"].append(dist)
            rec["ata"].append(ata_deg)
            rec["p_alt"].append(p_ned[2])
            rec["p_spd"].append(ps["airspeed_mps"])
            rec["p_hdg"].append(ps["yaw_deg"])
            rec["p_roll"].append(ps["roll_deg"])
            rec["t_alt"].append(t_ned[2])
            rec["t_spd"].append(ts["airspeed_mps"])
            rec["t_hdg"].append(ts["yaw_deg"])
            rec["p_n"].append(p_ned[0])
            rec["p_e"].append(p_ned[1])

        # ── Progress print ───────────────────────────────────────────
        if sim_time - last_print >= 10.0:
            name = describe_pursuit_action(current_action)
            print(f"{sim_time:6.1f} {name:>20s} {dist:7.0f} {ata_deg:+7.1f} "
                  f"{p_ned[2]:7.0f} {ps['airspeed_mps']:6.0f} "
                  f"{ps['yaw_deg']:6.1f} {ps['roll_deg']:6.1f}")
            last_print = sim_time

        # ── Termination ──────────────────────────────────────────────
        if dist < INTERCEPT_RADIUS_M:
            reason = "intercept"
            break
        if p_ned[2] < 10.0:
            reason = "ground_crash"
            break
        if p_ned[2] > 12000.0:
            reason = "ceiling"
            break
        if dist > 10000.0:
            reason = "lost_target"
            break
        ps_check = pursuer.state
        if any(not np.isfinite(float(ps_check.get(k, 0)))
               for k in ["n_z_g", "airspeed_mps", "alt_m"]):
            reason = "nan"
            break

    # ── Summary ───────────────────────────────────────────────────────
    name = describe_pursuit_action(current_action)
    print(f"\n  Result: {reason} | time={sim_time:.1f}s | min_dist={min_dist:.0f}m | "
          f"final action={name}")

    rec["reason"] = reason
    rec["min_dist"] = min_dist
    rec["sim_time"] = sim_time
    rec["final_action"] = current_action
    rec["p_ned"] = p_ned
    rec["t_ned"] = t_ned

    return rec


def _plot_pursuit(rec: dict) -> None:
    """Generate 2D top-down trajectory and altitude profile."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Heuristic Pure Pursuit — Discrete BFM Action Policy\n"
                 f"Result: {rec['reason']} | min_dist={rec['min_dist']:.0f}m | "
                 f"time={rec['sim_time']:.1f}s",
                 fontsize=12, fontweight="bold")

    # ── Top-down view ────────────────────────────────────────────────
    p_n = np.array(rec["p_n"])
    p_e = np.array(rec["p_e"])
    t_n = np.cumsum([kts_to_mps(TARGET_SPD_KTS) * (1.0/60.0) * 6] * len(p_n))

    ax1.plot(p_e, p_n, "b-", lw=1.2, label="Pursuer")
    ax1.plot(t_n, np.zeros_like(t_n) + rec["t_alt"][0], "r--", lw=1.0, label="Target")
    ax1.plot(p_e[0], p_n[0], "go", ms=8, label="Start")
    ax1.plot(p_e[-1], p_n[-1], "ro", ms=8, label="End")
    ax1.set_xlabel("East (m)"); ax1.set_ylabel("North (m)")
    ax1.set_title("Top-Down Trajectory")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_aspect("equal")

    # ── Altitude profile ─────────────────────────────────────────────
    t = np.array(rec["t"])
    ax2.plot(t, rec["p_alt"], "b-", lw=1.5, label="Pursuer Alt")
    ax2.plot(t, rec["t_alt"], "r--", lw=1.0, label="Target Alt")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Altitude Profile")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    path = OUT_DIR / "pursuit_trajectory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def _save_csv(rec: dict) -> None:
    """Save time-series to CSV."""
    path = OUT_DIR / "pursuit_metrics.csv"
    keys = ["t", "action", "dist", "ata", "p_alt", "p_spd", "p_hdg", "p_roll",
            "t_alt", "t_spd", "t_hdg", "p_n", "p_e"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(rec["t"])):
            w.writerow([rec[k][i] for k in keys])
    print(f"  CSV saved → {path}")


def _save_tacview(rec: dict) -> None:
    """Write Tacview ACMI for pursuer and target trajectories."""
    path = OUT_DIR / "pursuit.acmi"
    t = np.array(rec["t"])
    if len(t) < 2:
        return

    dt_acmi = t[1] - t[0]
    p_hdg_rad = np.deg2rad(rec["p_hdg"])
    p_spd = np.array(rec["p_spd"])
    p_vn = p_spd * np.cos(p_hdg_rad)
    p_ve = p_spd * np.sin(p_hdg_rad)
    p_n = np.cumsum(np.concatenate([[rec["p_n"][0]], p_vn[1:] * dt_acmi]))
    p_e = np.cumsum(np.concatenate([[rec["p_e"][0]], p_ve[1:] * dt_acmi]))

    ref_lat, ref_lon = 30.0, 120.0
    m_per_deg = 111320.0 * np.cos(np.radians(ref_lat))

    with open(path, "w") as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
        f.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")
        f.write("# Heuristic Pure Pursuit Test\n")
        f.write("0,Name=F-16 Pursuer\n0,Color=Blue\n")
        f.write("1,Name=F-16 Target\n1,Color=Red\n")

        for i in range(0, len(t), 3):
            p_lat = ref_lat + p_n[i] / m_per_deg
            p_lon = ref_lon + p_e[i] / m_per_deg
            # ACMI T= format: Longitude|Latitude|Altitude|Roll|Pitch|Yaw
            f.write(f"#{t[i]:.2f}\n")
            f.write(f"0,T={p_lon:.6f}|{p_lat:.6f}|{rec['p_alt'][i]:.1f}"
                    f"|{rec['p_roll'][i]:.1f}|0.0|{rec['p_hdg'][i]:.1f}\n")
            # Target: straight East at constant speed
            t_lat = ref_lat
            t_lon = ref_lon + (kts_to_mps(TARGET_SPD_KTS) * t[i]) / m_per_deg
            f.write(f"1,T={t_lon:.6f}|{t_lat:.6f}|{rec['t_alt'][i]:.1f}"
                    f"|0.0|0.0|{TARGET_HDG_DEG:.1f}\n")

    print(f"  Tacview saved → {path}")


def main() -> None:
    print("=" * 64)
    print("  Heuristic Pure Pursuit — Discrete BFM Action Policy")
    print(f"  Pursuer: {PURSUER_SPD_KTS:.0f} kts, Target: {TARGET_SPD_KTS:.0f} kts")
    print(f"  Initial separation: {SEPARATION_M:.0f} m")
    print(f"  Policy: ATA < -{ATA_DEADBAND_DEG}°→Turn Left, "
          f"ATA > +{ATA_DEADBAND_DEG}°→Turn Right, "
          f"|ATA|<{ATA_DEADBAND_DEG}°→Accelerate")
    print(f"  Min action hold: {MIN_ACTION_HOLD_S:.1f} s")
    print("=" * 64)

    rec = run_pursuit()

    _plot_pursuit(rec)
    _save_csv(rec)
    _save_tacview(rec)

    # ── Final verdict ──────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    if rec["reason"] == "intercept":
        print(f"  VERDICT: SUCCESS — intercept in {rec['sim_time']:.1f}s, "
              f"min_dist={rec['min_dist']:.0f}m")
    elif rec["reason"] == "timeout":
        print(f"  VERDICT: TIMEOUT — min_dist={rec['min_dist']:.0f}m "
              f"(target not reached in {MAX_TIME_S}s)")
    else:
        print(f"  VERDICT: FAILURE — {rec['reason']}, "
              f"min_dist={rec['min_dist']:.0f}m")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
