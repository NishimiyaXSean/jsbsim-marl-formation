"""Verify F-16 pursuit: can a simple guidance law achieve target intercept?

Each scenario starts pursuer 2km behind a target flying East at 180 m/s.
Key question: with proper control, can the F-16 close 2km and get within 50m?

Usage:
    JSBSIM_DEBUG=0 python scripts/verify_pursuit.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.bfm_actions import get_bfm_action
from src.utils.geometry import compute_forward_vector, compute_los
from src.utils.units import m_to_ft, mps_to_kts, deg_to_rad

PHYSICS_DT = 1.0 / 60.0
TARGET_HDG = 90.0
TARGET_SPD = 180.0    # m/s (~350 kts)
TARGET_ALT = 3000.0    # m (~9842 ft)
SEPARATION = 2000.0    # m
MAX_TIME = 120.0


def create_target():
    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=m_to_ft(TARGET_ALT),
             heading_deg=TARGET_HDG, speed_kts=mps_to_kts(TARGET_SPD), trim=False)
    ned = np.array([0.0, 0.0, TARGET_ALT], dtype=np.float64)
    ac.position_ned = ned
    return ac, ned


def create_pursuer(speed_mps=250.0):
    """Pursuer starts faster to have energy advantage."""
    ac = Aircraft()
    ac.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=m_to_ft(TARGET_ALT),
             heading_deg=TARGET_HDG, speed_kts=mps_to_kts(speed_mps), trim=False)
    ned = np.array([0.0, -SEPARATION, TARGET_ALT], dtype=np.float64)
    ac.position_ned = ned
    return ac, ned


def hold_target(ac):
    """Simple P-correction to keep target on rails."""
    s = ac.state
    he = (TARGET_HDG - s["yaw_deg"] + 180) % 360 - 180
    ae = TARGET_ALT - s["alt_m"]
    se = TARGET_SPD - s["airspeed_mps"]
    ac.set_controls(
        throttle=np.clip(0.80 + se * 0.005, 0.5, 1.0),
        elevator=np.clip(-0.05 - ae * 0.001, -0.15, 0.05),
        aileron=np.clip(he * 0.02, -0.06, 0.06),
        rudder=0.0,
    )


def run_one(label, control_fn, pursuer_spd=250.0):
    pursuer, p_ned = create_pursuer(pursuer_spd)
    target, t_ned = create_target()

    min_dist = SEPARATION
    reason = "timeout"
    sim_time = 0.0
    step_count = 0

    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"  Pursuer starts at {pursuer_spd} m/s ({mps_to_kts(pursuer_spd):.0f} kts)")
    print(f"{'─'*58}")
    print(f"{'Time':>6s} {'Dist':>7s} {'MinD':>7s} {'P_Alt':>6s} {'P_Spd':>6s} {'P_Hdg':>6s} {'P_Roll':>6s} {'Thr':>5s} {'Elev':>6s}")
    print("-" * 78)

    last_print = -2.0

    while sim_time < MAX_TIME:
        step_count += 1
        sim_time = step_count * PHYSICS_DT

        # Control
        thr, elev, ail, rud = control_fn(pursuer, p_ned, t_ned, PHYSICS_DT)
        pursuer.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
        hold_target(target)

        # Step
        pursuer.run()
        target.run()

        # NED
        p_ned[0:2] += pursuer.velocity_ned[0:2] * PHYSICS_DT
        p_ned[2] = pursuer.state["alt_m"]
        pursuer.position_ned = p_ned
        t_ned[0:2] += target.velocity_ned[0:2] * PHYSICS_DT
        t_ned[2] = target.state["alt_m"]
        target.position_ned = t_ned

        dist = float(np.linalg.norm(p_ned - t_ned))
        if dist < min_dist:
            min_dist = dist

        # Termination
        if dist < 50.0:
            reason = "success"
            break
        if p_ned[2] < 10.0:
            reason = "ground_crash"
            break
        if p_ned[2] > 12000.0 or dist > 10000.0:
            reason = "lost_target" if dist > 10000.0 else "out_of_bounds"
            break
        ps = pursuer.state
        if any(not np.isfinite(float(ps.get(k, 0))) for k in ["n_z_g", "airspeed_mps", "alt_m"]):
            reason = "nan"
            break

        if sim_time - last_print >= 5.0:
            ps = pursuer.state
            print(f"{sim_time:6.1f} {dist:7.0f} {min_dist:7.0f} {p_ned[2]:6.0f} "
                  f"{ps['airspeed_mps']:6.0f} {ps['yaw_deg']:6.1f} {ps['roll_deg']:6.1f} "
                  f"{thr:5.2f} {elev:6.3f}")
            last_print = sim_time

    ps = pursuer.state
    print(f"\n  Result: {reason} | time={sim_time:.1f}s | min_dist={min_dist:.0f}m | "
          f"final spd={ps['airspeed_mps']:.0f}m/s")
    return {"label": label, "reason": reason, "min_dist": min_dist, "time": sim_time}


# ═══════════════════════════════════════════════════════════════════════════════
#  Scenario A: Level-flight acceleration test
#  Can the F-16 sustain >180 m/s in LEVEL flight at full throttle?
# ═══════════════════════════════════════════════════════════════════════════════

def test_level_speed():
    """Just fly straight at full throttle. What speed can we hold in level flight?"""
    # Need to find elevator that gives level flight at full throttle
    # Trim at thr=0.80, elev=-0.05 → level at ~176 m/s
    # At thr=1.00, more thrust → need less nose-up (more positive elev)
    # Let's try elev=-0.02 (slightly nose-down from trim to hold level at higher speed)

    def control(pursuer, p_ned, t_ned, dt):
        # Altitude hold via elevator
        ae = TARGET_ALT - pursuer.state["alt_m"]
        elev = np.clip(-0.02 - ae * 0.003, -0.15, 0.10)
        return (1.0, elev, 0.0, 0.0)

    return run_one("Scenario A: Full throttle, level flight speed test", control, pursuer_spd=250.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Scenario B: Altitude-trading pursuit
#  Trade altitude for speed to catch target, then pull up
# ═══════════════════════════════════════════════════════════════════════════════

def test_dive_pursuit():
    """Start above target, dive to gain speed, pure pursuit in the horizontal plane."""

    pursuer = Aircraft()
    pursuer.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=m_to_ft(TARGET_ALT + 500),
                 heading_deg=TARGET_HDG, speed_kts=mps_to_kts(250.0), trim=False)
    p_ned = np.array([0.0, -SEPARATION, TARGET_ALT + 500.0], dtype=np.float64)
    pursuer.position_ned = p_ned
    target, t_ned = create_target()

    min_dist = SEPARATION
    reason = "timeout"
    sim_time = 0.0

    print(f"\n{'─'*58}")
    print(f"  Scenario B: Dive from +500m, pure pursuit heading guidance")
    print(f"  Pursuer starts at 250 m/s, 500m above target")
    print(f"{'─'*58}")
    print(f"{'Time':>6s} {'Dist':>7s} {'MinD':>7s} {'P_Alt':>6s} {'P_Spd':>6s} {'P_Hdg':>6s} {'P_Roll':>6s} {'Thr':>5s} {'Elev':>6s}")
    print("-" * 78)

    last_print = -2.0
    step = 0

    while sim_time < MAX_TIME:
        step += 1
        sim_time = step * PHYSICS_DT

        # Pure pursuit heading
        _, los_dir, _ = compute_los(p_ned, t_ned)
        desired_hdg = np.degrees(np.arctan2(los_dir[1], los_dir[0])) % 360.0
        hdg_err = (desired_hdg - pursuer.state["yaw_deg"] + 180) % 360 - 180

        # Elevator: dive to target altitude, but don't crash
        alt_target = max(t_ned[2], 500.0)
        ae = alt_target - p_ned[2]
        elev = np.clip(-0.02 - ae * 0.003, -0.20, 0.15)
        ail = np.clip(hdg_err * 0.04, -0.3, 0.3)
        thr = 1.0

        pursuer.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=0.0)
        hold_target(target)
        pursuer.run()
        target.run()

        p_ned[0:2] += pursuer.velocity_ned[0:2] * PHYSICS_DT
        p_ned[2] = pursuer.state["alt_m"]
        pursuer.position_ned = p_ned
        t_ned[0:2] += target.velocity_ned[0:2] * PHYSICS_DT
        t_ned[2] = target.state["alt_m"]
        target.position_ned = t_ned

        dist = float(np.linalg.norm(p_ned - t_ned))
        if dist < min_dist:
            min_dist = dist

        if dist < 50.0:
            reason = "success"
            break
        if p_ned[2] < 10.0:
            reason = "ground_crash"
            break
        if p_ned[2] > 12000.0 or dist > 10000.0:
            reason = "lost_target" if dist > 10000.0 else "out_of_bounds"
            break
        ps = pursuer.state
        if any(not np.isfinite(float(ps.get(k, 0))) for k in ["n_z_g", "airspeed_mps", "alt_m"]):
            reason = "nan"
            break

        if sim_time - last_print >= 5.0:
            ps = pursuer.state
            print(f"{sim_time:6.1f} {dist:7.0f} {min_dist:7.0f} {p_ned[2]:6.0f} "
                  f"{ps['airspeed_mps']:6.0f} {ps['yaw_deg']:6.1f} {ps['roll_deg']:6.1f} "
                  f"{thr:5.2f} {elev:6.3f}")
            last_print = sim_time

    ps = pursuer.state
    print(f"\n  Result: {reason} | time={sim_time:.1f}s | min_dist={min_dist:.0f}m | "
          f"final spd={ps['airspeed_mps']:.0f}m/s")
    return {"label": "Scenario B: Dive pursuit (+500m start)", "reason": reason,
            "min_dist": min_dist, "time": sim_time}


# ═══════════════════════════════════════════════════════════════════════════════
#  Scenario C: BFM autopilot with BFM actions
#  Use the actual BFM action space (action 1 = accelerate, action 9/10 = gentle turn)
# ═══════════════════════════════════════════════════════════════════════════════

def test_bfm_actions():
    """Use discrete BFM actions through the full envelope + autopilot pipeline."""

    bfm = BFMAutopilot()
    bfm.reset(initial_speed_mps=250.0)
    envelope = FlightEnvelope()

    def control(pursuer, p_ned, t_ned, dt):
        _, los_dir, dist = compute_los(p_ned, t_ned)

        # Choose BFM action based on tactical situation
        desired_hdg = np.degrees(np.arctan2(los_dir[1], los_dir[0])) % 360.0
        hdg_err = (desired_hdg - pursuer.state["yaw_deg"] + 180) % 360 - 180

        if abs(hdg_err) < 5 and dist > 500:
            # Target ahead — accelerate straight (action 1)
            n_x, n_n, mu = get_bfm_action(1)
        elif abs(hdg_err) < 30:
            # Small correction — gentle turn toward target
            if hdg_err > 0:
                n_x, n_n, mu = get_bfm_action(10)  # gentle left turn
            else:
                n_x, n_n, mu = get_bfm_action(9)   # gentle right turn
        else:
            # Large correction — climbing/diving turn toward target
            if hdg_err > 0:
                n_x, n_n, mu = get_bfm_action(5)   # left climbing turn (gentler)
                # Override with gentler params for pursuit
                n_x, n_n, mu = 1.0, 3.0, np.pi / 6
            else:
                n_x, n_n, mu = 1.0, 3.0, -np.pi / 6

        # Pass through flight envelope
        ps = pursuer.state
        roll_rad = np.deg2rad(ps["roll_deg"])
        vz_mps = -ps["w_fps"] * 0.3048
        n_x_e, n_n_e, mu_e = envelope.step(
            n_x, n_n, mu,
            speed_mps=ps["airspeed_mps"], alt_m=ps["alt_m"], vz_mps=vz_mps,
            current_roll_rad=roll_rad, dt=dt, is_attacker=True,
        )

        thr, elev, ail, rud = bfm.step(
            n_x_e, n_n_e, mu_e, dt,
            n_z_g=ps["n_z_g"], roll_rad=roll_rad,
            airspeed_mps=ps["airspeed_mps"], beta_deg=ps["beta_deg"],
            q_rps=ps["q_rps"],
        )
        return (thr, elev, ail, rud)

    return run_one("Scenario C: BFM actions via envelope + autopilot", control, pursuer_spd=250.0)


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("F-16 Pursuit Verification")
    print(f"Target: {TARGET_HDG} deg East, {TARGET_SPD} m/s, {TARGET_ALT}m")
    print(f"Pursuer starts {SEPARATION}m behind")
    print("=" * 60)

    results = []
    results.append(test_level_speed())
    results.append(test_dive_pursuit())
    results.append(test_bfm_actions())

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r["reason"] == "success" else f"FAIL ({r['reason']})"
        print(f"  {status:20s} | min_dist={r['min_dist']:6.0f}m | {r['label']}")

    passes = sum(1 for r in results if r["reason"] == "success")
    print(f"\n  {passes}/{len(results)} scenarios achieved intercept.")


if __name__ == "__main__":
    import logging, warnings
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
