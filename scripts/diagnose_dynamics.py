"""Diagnostic script: test F-16 dynamics, maneuverability, and control chain.

Checks:
1. Can the F-16 perform sustained turns?
2. Can it climb and dive effectively?
3. What are the physical limits (max turn rate, climb rate, speed range)?
4. Does the PN guidance produce reasonable heading commands?
5. What control authority does the RL agent actually have?
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dynamics.aircraft import Aircraft
from src.dynamics.flight_controller import FlightController, FlightControlTargets, AltitudeStabilizer, SpeedStabilizer, HeadingStabilizer, ELEVATOR_TRIM, THROTTLE_TRIM
from src.utils.pn_guidance import compute_pn_heading
from src.utils.geometry import compute_forward_vector, compute_tactical_angles, compute_los

os.environ.setdefault("JSBSIM_DEBUG", "0")


def test_basic_flight():
    """Verify F-16 can maintain stable flight at trim."""
    print("=" * 60)
    print("TEST 1: Basic Trim Flight (3 seconds)")
    print("=" * 60)

    ac = Aircraft()
    ac.reset(lat_deg=30, lon_deg=120, alt_ft=9842, heading_deg=90, speed_kts=400)
    ac.position_ned = np.array([0.0, 0.0, 3000.0])

    for i in range(180):  # 3s at 60Hz
        ac.set_controls(throttle=0.80, elevator=-0.05, aileron=0.0, rudder=0.0)
        ac.run()
        ac.position_ned[0:2] += ac.velocity_ned[0:2] * (1/60)
        ac.position_ned[2] = ac.state["alt_m"]

    s = ac.state
    print(f"  Altitude:    {s['alt_m']:.1f} m")
    print(f"  Airspeed:    {s['airspeed_mps']:.1f} m/s ({s['airspeed_kts']:.1f} kts)")
    print(f"  n_z:         {s['n_z_g']:.3f} G")
    print(f"  Pitch:       {s['pitch_deg']:.1f}°")
    print(f"  Roll:        {s['roll_deg']:.1f}°")
    print(f"  Alpha:       {s['alpha_deg']:.1f}°")
    print(f"  Thrust:      {s['thrust_lbs']:.0f} lbs")
    print(f"  Mach:        {s['mach']:.3f}")
    alt_stable = abs(s['alt_m'] - 3000.0) < 100
    print(f"  Altitude stable: {alt_stable} (within 100m of 3000m)")
    return alt_stable


def test_max_turn_rate():
    """Test maximum sustained turn rate with full aileron."""
    print("\n" + "=" * 60)
    print("TEST 2: Maximum Sustained Turn Rate")
    print("=" * 60)

    for aileron_cmd in [0.1, 0.2, 0.3, 0.5, 1.0]:
        ac = Aircraft()
        ac.reset(lat_deg=30, lon_deg=120, alt_ft=9842, heading_deg=0, speed_kts=400)
        ac.position_ned = np.array([0.0, 0.0, 3000.0])

        headings = []
        for i in range(300):  # 5s
            # Use FC for altitude + speed, direct aileron
            fc = FlightController()
            fc.reset()
            # Simplified: just set controls directly
            ac.set_controls(throttle=0.80, elevator=-0.05, aileron=aileron_cmd, rudder=0.0)
            ac.run()
            headings.append(ac.state["yaw_deg"])

        # Compute turn rate from heading change
        d_hdg = (headings[-1] - headings[0] + 180) % 360 - 180
        turn_rate = d_hdg / 5.0  # deg/s over 5s
        max_roll = max(abs(ac.state["roll_deg"]) for _ in [1])  # approximate
        print(f"  Aileron={aileron_cmd:4.1f}:  turn_rate={turn_rate:6.1f}°/s  "
              f"final_roll={ac.state['roll_deg']:.0f}°  "
              f"final_alt={ac.state['alt_m']:.0f}m  "
              f"final_speed={ac.state['airspeed_mps']:.0f}m/s")


def test_climb_dive():
    """Test climb and dive capability."""
    print("\n" + "=" * 60)
    print("TEST 3: Climb and Dive Performance")
    print("=" * 60)

    for elev_cmd, label in [(-0.3, "Climb (elev=-0.3)"), (-0.1, "Gentle climb (-0.1)"),
                              (0.1, "Gentle dive (0.1)"), (0.3, "Dive (0.3)")]:
        ac = Aircraft()
        ac.reset(lat_deg=30, lon_deg=120, alt_ft=9842, heading_deg=90, speed_kts=400)
        ac.position_ned = np.array([0.0, 0.0, 3000.0])

        alts = []
        speeds = []
        for i in range(300):  # 5s
            ac.set_controls(throttle=0.80, elevator=elev_cmd, aileron=0.0, rudder=0.0)
            ac.run()
            ac.position_ned[2] = ac.state["alt_m"]
            alts.append(ac.state["alt_m"])
            speeds.append(ac.state["airspeed_mps"])

        vs_fps = ac.state["h_dot_fps"]
        vs_mps = vs_fps * 0.3048
        print(f"  {label:25s}: Δalt={alts[-1]-alts[0]:+6.0f}m  "
              f"final_vs={vs_mps:+5.1f}m/s  "
              f"speed={speeds[-1]:.0f}m/s  "
              f"n_z={ac.state['n_z_g']:.2f}G")


def test_pn_guidance():
    """Test PN guidance law with typical engagement geometry."""
    print("\n" + "=" * 60)
    print("TEST 4: PN Guidance Logic")
    print("=" * 60)

    # Typical scenario: pursuer at origin heading 90°, target at 1km NE moving east
    pursuer_ned = np.array([0.0, 0.0, 3000.0])
    pursuer_vel = np.array([0.0, 206.0, 0.0])  # heading 90° at 206 m/s
    target_ned = np.array([800.0, 800.0, 3000.0])
    target_vel = np.array([0.0, 130.0, 0.0])   # heading 90° at 130 m/s
    current_heading = 90.0

    for i in range(10):
        desired = compute_pn_heading(pursuer_ned, pursuer_vel, target_ned, target_vel,
                                     current_heading, dt=0.5, nav_constant=3.0, max_turn_rate_dps=15.0)
        heading_err = (desired - current_heading + 180) % 360 - 180
        print(f"  Step {i}: bearing from pursuer to target, "
              f"heading={current_heading:.0f}° desired={desired:.1f}° error={heading_err:+.1f}°")

        # Move both aircraft
        current_heading = desired
        pursuer_ned[:2] += pursuer_vel[:2] * 0.5
        target_ned[:2] += target_vel[:2] * 0.5


def test_action_flow():
    """Trace the actual control flow from RL action to aircraft surface."""
    print("\n" + "=" * 60)
    print("TEST 5: Action Flow Analysis")
    print("=" * 60)

    # Simulate what ResidualExpertWrapper outputs
    expert_ail = 0.15   # typical PN output for a moderate heading error
    expert_alt = 0.0    # expert doesn't control altitude
    expert_spd = 1.0    # expert always commands full speed

    # Agent residual (typical small values based on clip_fraction=0)
    agent_action = np.array([0.02, -0.01, 0.03])  # very small corrections

    residual = agent_action * 0.5  # RESIDUAL_SCALE
    combined = np.clip(np.array([expert_ail, expert_alt, expert_spd]) + residual, -1.0, 1.0)

    print(f"  Expert output:      ail={expert_ail:.3f}, alt={expert_alt:.3f}, spd={expert_spd:.3f}")
    print(f"  Agent action (raw): {agent_action}")
    print(f"  Residual (×0.5):    {residual}")
    print(f"  Combined (clipped): {combined}")

    # Now trace through env.step()
    MAX_D_HEADING_DEG = 10.0
    MAX_D_ALT_M = 15.0
    MAX_D_SPEED_MPS = 10.0

    raw_ail = float(combined[0])
    d_alt = float(combined[1]) * MAX_D_ALT_M
    d_spd = float(combined[2]) * MAX_D_SPEED_MPS

    print(f"\n  After env.step() parsing:")
    print(f"  raw_ail={raw_ail:.3f} → aileron surface = {raw_ail*0.3:.3f}")
    print(f"  d_alt={d_alt:.1f}m per decision (0.5s) → {d_alt/0.5:.1f} m/s climb rate")
    print(f"  d_spd={d_spd:.1f}m/s per decision → speed target increases by {d_spd:.1f} m/s")

    # Expert always sets d_spd=1.0 → MAX_D_SPEED_MPS=10 m/s per 0.5s = 20 m/s/s
    # This means the speed target keeps increasing until it hits the max (250 m/s)
    print(f"\n  !! CRITICAL: Expert always sets d_spd=1.0")
    print(f"  → speed target increases by {MAX_D_SPEED_MPS:.0f} m/s every 0.5s")
    print(f"  → reaches max (250 m/s) in {(250-250)/(MAX_D_SPEED_MPS/0.5):.1f}s")
    print(f"  → agent residual on d_spd is clipped to [0.5, 1.0] → CANNOT reduce speed!")


def test_heading_stabilizer():
    """Test the full FlightController heading response."""
    print("\n" + "=" * 60)
    print("TEST 6: FlightController Heading Response")
    print("=" * 60)

    ac = Aircraft()
    ac.reset(lat_deg=30, lon_deg=120, alt_ft=9842, heading_deg=0, speed_kts=400)
    ac.position_ned = np.array([0.0, 0.0, 3000.0])

    fc = FlightController()
    fc.reset()

    target = FlightControlTargets(heading_deg=90.0, altitude_m=3000.0, speed_mps=180.0)

    headings = []
    rolls = []
    for i in range(600):  # 10s
        thr, elev, ail, rud = fc.compute(ac.state, target, dt=1/60)
        ac.set_controls(thr, elev, ail, rud)
        ac.run()
        ac.position_ned[0:2] += ac.velocity_ned[0:2] * (1/60)
        ac.position_ned[2] = ac.state["alt_m"]
        headings.append(ac.state["yaw_deg"])
        rolls.append(ac.state["roll_deg"])

    final_hdg = headings[-1]
    hdg_err = (target.heading_deg - final_hdg + 180) % 360 - 180
    max_roll = max(abs(r) for r in rolls)
    print(f"  Target heading: {target.heading_deg}°")
    print(f"  Final heading:  {final_hdg:.1f}° (error={hdg_err:.1f}°)")
    print(f"  Max roll:       {max_roll:.0f}°")
    print(f"  Capture time:   ~{len([h for h in headings if abs((h - target.heading_deg + 180)%360-180) < 5]) / 60:.1f}s")
    print(f"  Final alt:      {ac.state['alt_m']:.0f}m")
    print(f"  Final speed:    {ac.state['airspeed_mps']:.0f}m/s")


def test_combined_maneuver():
    """Test a complex combined maneuver: turn + climb simultaneously."""
    print("\n" + "=" * 60)
    print("TEST 7: Combined Turn + Climb (Pursuit-like)")
    print("=" * 60)

    ac = Aircraft()
    ac.reset(lat_deg=30, lon_deg=120, alt_ft=9842, heading_deg=0, speed_kts=400)
    ac.position_ned = np.array([0.0, 0.0, 3000.0])

    fc = FlightController()
    fc.reset()

    # Simulate a pursuit scenario: target is at NE, above, moving fast
    # FC drives heading 45°, climb to 3500m, speed 200 m/s
    target = FlightControlTargets(heading_deg=45.0, altitude_m=3500.0, speed_mps=200.0)

    positions = []
    for i in range(600):  # 10s
        thr, elev, ail, rud = fc.compute(ac.state, target, dt=1/60)
        ac.set_controls(thr, elev, ail, rud)
        ac.run()
        ac.position_ned[0:2] += ac.velocity_ned[0:2] * (1/60)
        ac.position_ned[2] = ac.state["alt_m"]
        if i % 30 == 0:
            positions.append(ac.position_ned.copy())

    s = ac.state
    print(f"  Start: NED=[0, 0, 3000]")
    p = positions[-1]
    print(f"  End:   NED=[{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}]")
    print(f"  Heading: {s['yaw_deg']:.1f}° (target 45°)")
    print(f"  Altitude: {s['alt_m']:.0f}m (target 3500m)")
    print(f"  Speed: {s['airspeed_mps']:.0f}m/s (target 200m/s)")
    print(f"  Max roll during maneuver: check above")


if __name__ == "__main__":
    import logging
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    print("=" * 60)
    print("F-16 DYNAMICS & CONTROL CHAIN DIAGNOSTIC")
    print("=" * 60)

    test_basic_flight()
    test_max_turn_rate()
    test_climb_dive()
    test_pn_guidance()
    test_action_flow()
    test_heading_stabilizer()
    test_combined_maneuver()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
