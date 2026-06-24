"""Phase 4: BFM discrete command suite full validation.

Executes all 9 PURSUIT_ACTIONS through the complete Phase 3 pipeline
(FlightEnvelope → BFMAutopilot → JSBSim FCS) and verifies:
1. Each action produces a stable trajectory (5s hold)
2. Random switching never causes loss of control
3. Composite actions (accelerating turns) are stable
4. Tacview + trajectory plots for visual inspection

Usage:
    /c/Users/Sean/anaconda3/envs/jsbsim_rl/python scripts/verify_bfm_actions.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.bfm_actions import PURSUIT_ACTIONS, describe_pursuit_action

os.environ.setdefault("JSBSIM_DEBUG", "0")


# ── Test parameters ────────────────────────────────────────────────────────

DT = 1.0 / 60.0
INIT_ALT_FT = 9842      # 3000 m
INIT_HEADING_DEG = 90.0
INIT_SPEED_KTS = 400
ACTION_HOLD_S = 5.0      # hold each action for 5s
STRESS_DURATION_S = 60.0  # random switching stress test
STRESS_SEEDS = [42, 123, 456]
STRESS_SWITCH_INTERVAL = (0.5, 2.0)  # seconds between random switches


def _run_action(ac: Aircraft, ap: BFMAutopilot, envelope: FlightEnvelope,
                action_idx: int, duration_s: float) -> dict:
    """Execute one BFM action for *duration_s* seconds, return telemetry."""
    n_x_raw, n_n_raw, mu_raw = PURSUIT_ACTIONS[action_idx]

    t_vals, alt_vals, spd_vals = [], [], []
    nz_vals, alpha_vals, roll_vals, hdg_vals, thrust_vals = [], [], [], [], []
    elev_vals, ail_vals, thr_vals = [], [], []

    t = 0.0
    n_steps = int(duration_s / DT)
    for _ in range(n_steps):
        s = ac.state
        # Run through FlightEnvelope
        n_x_env, n_n_env, mu_env = envelope.step(
            n_x_raw, n_n_raw, mu_raw,
            speed_mps=s["airspeed_mps"], alt_m=s["alt_m"],
            vz_mps=s["h_dot_fps"] * 0.3048,  # ft/s → m/s
            current_roll_rad=np.deg2rad(s["roll_deg"]), dt=DT,
        )
        # Autopilot
        thr, elev, ail, rud = ap.step(
            n_x_env, n_n_env, mu_env, DT,
            n_z_g=s["n_z_g"],
            roll_rad=np.deg2rad(s["roll_deg"]),
            airspeed_mps=s["airspeed_mps"],
            beta_deg=s["beta_deg"],
            alpha_deg=s["alpha_deg"],
        )
        ac.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
        ac.run()

        t_vals.append(t)
        alt_vals.append(s["alt_m"])
        spd_vals.append(s["airspeed_mps"])
        nz_vals.append(-s["n_z_g"])  # perceived G
        alpha_vals.append(s["alpha_deg"])
        roll_vals.append(s["roll_deg"])
        hdg_vals.append(s["yaw_deg"])
        thrust_vals.append(s["thrust_lbs"])
        elev_vals.append(elev)
        ail_vals.append(ail)
        thr_vals.append(thr)
        t += DT

    return {
        "action": action_idx,
        "t": np.array(t_vals),
        "alt": np.array(alt_vals), "spd": np.array(spd_vals),
        "nz": np.array(nz_vals), "alpha": np.array(alpha_vals),
        "roll": np.array(roll_vals), "hdg": np.array(hdg_vals),
        "thrust": np.array(thrust_vals),
        "elev": np.array(elev_vals), "ail": np.array(ail_vals),
        "thr": np.array(thr_vals),
        "alpha_max": float(np.max(np.abs(alpha_vals))),
        "nz_max": float(np.max(np.abs(nz_vals))),
        "alt_min": float(np.min(alt_vals)),
        "alt_max": float(np.max(alt_vals)),
        "spd_min": float(np.min(spd_vals)),
        "spd_max": float(np.max(spd_vals)),
    }


def _save_tacview_single(r: dict, out_dir: str):
    """Write a standalone Tacview .acmi file for one action, with spatial tracking."""
    action_idx = r["action"]
    action_name = describe_pursuit_action(action_idx)
    safe_name = f"action_{action_idx}_{action_name.replace(' ','_').replace('+','plus')}"
    path = os.path.join(out_dir, f"{safe_name}.txt.acmi")

    # Reconstruct lat/lon by integrating heading + speed (flat-earth, 3000m reference)
    t = r["t"]
    hdg_rad = np.deg2rad(r["hdg"])
    spd = r["spd"]
    dt = t[1] - t[0] if len(t) > 1 else 1.0 / 60.0

    # Integrate NED position from velocity
    v_n = spd * np.cos(hdg_rad)
    v_e = spd * np.sin(hdg_rad)
    n = np.cumsum(v_n * dt)
    e = np.cumsum(v_e * dt)

    # Convert meters back to lat/lon offset from reference point
    ref_lat, ref_lon = 30.0, 120.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref_lat))
    lat_vals = ref_lat + n / m_per_deg_lat
    lon_vals = ref_lon + e / m_per_deg_lon

    with open(path, "w") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")
        f.write(f"# Action {action_idx}: {action_name} — 5s hold at 3000m/400kts\n")
        f.write("0,Name=F-16\n")
        f.write("0,Color=Red\n")

        for i in range(0, len(t), 5):  # downsample to 12 Hz for smooth playback
            f.write(f"#{t[i]:.2f}\n")
            f.write(f"0,T={lat_vals[i]:.6f}|{lon_vals[i]:.6f}|{r['alt'][i]:.1f}"
                    f"|{r['roll'][i]:.1f}|{0.0}|{r['hdg'][i]:.1f}\n")

    print(f"  Tacview → {path}")


def _save_tacview_all(all_results: list[dict], out_dir: str):
    """Save individual Tacview files for each action."""
    for r in all_results:
        _save_tacview_single(r, out_dir)


def main():
    out_dir = "results/bfm_validation"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 4: BFM Discrete Command Suite Validation")
    print("=" * 60)

    cfg = BFMAutopilotConfig()
    trim = TrimSchedule()
    envelope = FlightEnvelope(EnvelopeConfig())

    # ── 1. Individual action tests ─────────────────────────────────────
    print("\n--- Individual Action Tests (5s each) ---")
    all_results = []
    failures = []

    for action_idx in sorted(PURSUIT_ACTIONS.keys()):
        ap = BFMAutopilot(cfg, trim=trim)
        ac = Aircraft()
        ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                 heading_deg=INIT_HEADING_DEG, speed_kts=INIT_SPEED_KTS)
        # Warmup
        for _ in range(60):
            s = ac.state
            thr, elev, ail, rud = ap.step(
                0.0, 1.0, 0.0, DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"],
            )
            ac.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            ac.run()

        r = _run_action(ac, ap, envelope, action_idx, ACTION_HOLD_S)
        all_results.append(r)

        ok = (r["alpha_max"] < 28.0 and r["nz_max"] < 9.5
              and r["alt_min"] > 500.0 and r["alt_max"] < 6000.0
              and r["spd_min"] > 80.0 and r["spd_max"] < 400.0)
        status = "OK" if ok else f"FAIL (a_max={r['alpha_max']:.1f}, nz_max={r['nz_max']:.1f})"
        print(f"  Action {action_idx}: {describe_pursuit_action(action_idx):25s}  "
              f"alpha_max={r['alpha_max']:.1f}°  nz_max={r['nz_max']:.1f}G  "
              f"dh={r['alt_max']-r['alt_min']:.0f}m  {status}")
        if not ok:
            failures.append((action_idx, status))

    # ── 2. Random switching stress test ────────────────────────────────
    print(f"\n--- Random Switching Stress Test ({STRESS_DURATION_S}s, "
          f"{len(STRESS_SEEDS)} seeds) ---")
    stress_ok = True
    for seed in STRESS_SEEDS:
        rng = np.random.default_rng(seed)
        ap = BFMAutopilot(cfg, trim=trim)
        ac = Aircraft()
        ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                 heading_deg=INIT_HEADING_DEG, speed_kts=INIT_SPEED_KTS)
        # Warmup
        for _ in range(60):
            s = ac.state
            thr, elev, ail, rud = ap.step(
                0.0, 1.0, 0.0, DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"],
            )
            ac.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            ac.run()

        alt_min, spd_min = 99999.0, 99999.0
        alt_max, spd_max = 0.0, 0.0
        alpha_max_global = 0.0

        action_idx = 0
        next_switch = rng.uniform(*STRESS_SWITCH_INTERVAL)
        t_since_switch = 0.0
        n_steps = int(STRESS_DURATION_S / DT)

        for _ in range(n_steps):
            if t_since_switch >= next_switch:
                action_idx = rng.integers(0, len(PURSUIT_ACTIONS))
                t_since_switch = 0.0
                next_switch = rng.uniform(*STRESS_SWITCH_INTERVAL)

            n_x_raw, n_n_raw, mu_raw = PURSUIT_ACTIONS[action_idx]
            s = ac.state
            n_x_env, n_n_env, mu_env = envelope.step(
                n_x_raw, n_n_raw, mu_raw,
                speed_mps=s["airspeed_mps"], alt_m=s["alt_m"],
                vz_mps=s["h_dot_fps"] * 0.3048,
                current_roll_rad=np.deg2rad(s["roll_deg"]), dt=DT,
            )
            thr, elev, ail, rud = ap.step(
                n_x_env, n_n_env, mu_env, DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"],
            )
            ac.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            ac.run()

            alt = s["alt_m"]
            spd = s["airspeed_mps"]
            alt_min = min(alt_min, alt)
            alt_max = max(alt_max, alt)
            spd_min = min(spd_min, spd)
            spd_max = max(spd_max, spd)
            alpha_max_global = max(alpha_max_global, abs(s["alpha_deg"]))
            t_since_switch += DT

        seed_ok = (alt_min > 500.0 and alt_max < 6000.0
                   and spd_min > 80.0 and spd_max < 400.0
                   and alpha_max_global < 28.0)
        print(f"  Seed {seed}: alt=[{alt_min:.0f}, {alt_max:.0f}]m  "
              f"spd=[{spd_min:.0f}, {spd_max:.0f}]m/s  "
              f"alpha_max={alpha_max_global:.1f}°  "
              f"{'OK' if seed_ok else 'FAIL'}")
        if not seed_ok:
            stress_ok = False

    # ── 3. Deep test for composite actions (7, 8) ──────────────────────
    print("\n--- Composite Action Deep Test (10s each) ---")
    for action_idx in [7, 8]:
        ap = BFMAutopilot(cfg, trim=trim)
        ac = Aircraft()
        ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=INIT_ALT_FT,
                 heading_deg=INIT_HEADING_DEG, speed_kts=INIT_SPEED_KTS)
        for _ in range(60):
            s = ac.state
            thr, elev, ail, rud = ap.step(
                0.0, 1.0, 0.0, DT,
                n_z_g=s["n_z_g"], roll_rad=np.deg2rad(s["roll_deg"]),
                airspeed_mps=s["airspeed_mps"], beta_deg=s["beta_deg"],
                alpha_deg=s["alpha_deg"],
            )
            ac.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            ac.run()

        r = _run_action(ac, ap, envelope, action_idx, 10.0)
        ok = (r["alpha_max"] < 28.0 and r["nz_max"] < 9.5
              and r["alt_min"] > 500.0 and r["alt_max"] < 6000.0)
        print(f"  Action {action_idx} ({describe_pursuit_action(action_idx)}): "
              f"alpha_max={r['alpha_max']:.1f}°  nz_max={r['nz_max']:.1f}G  "
              f"dh={r['alt_max']-r['alt_min']:.0f}m  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append((action_idx, f"Composite: a_max={r['alpha_max']:.1f}"))

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    n_fail = len(failures) + (0 if stress_ok else 1)
    print(f"  Individual actions: {9 - len([f for f in failures if f[0] < 9])}/9 OK")
    print(f"  Composite actions:  {2 - len([f for f in failures if f[0] in (7,8)])}/2 OK")
    print(f"  Stress test:        {'OK' if stress_ok else 'FAIL'}")
    print(f"  TOTAL:              {'ALL PASSED' if n_fail == 0 else f'{n_fail} FAILURES'}")

    # ── Tacview output (one file per action with spatial tracking) ──────
    _save_tacview_all(all_results, out_dir)

    # ── Summary plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for i, r in enumerate(all_results):
        ax = axes[i // 3][i % 3]
        ax.plot(r["t"], r["alt"], "b-", lw=1.0, alpha=0.7, label="Alt (m)")
        ax2 = ax.twinx()
        ax2.plot(r["t"], r["spd"], "r-", lw=1.0, alpha=0.5, label="Spd (m/s)")
        ax.set_title(f"Action {r['action']}: {describe_pursuit_action(r['action'])}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (m)", color="b")
        ax2.set_ylabel("Speed (m/s)", color="r")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bfm_validation_summary.png"),
                dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {os.path.join(out_dir, 'bfm_validation_summary.png')}")
    plt.close("all")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
