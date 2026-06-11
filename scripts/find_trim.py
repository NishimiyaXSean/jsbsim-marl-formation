"""Find the elevator trim needed for level flight at full throttle.

Tests systematic elevator values at throttle=1.0 to find the setting
that gives steady level flight and measures the equilibrium speed.

Usage:
    JSBSIM_DEBUG=0 python scripts/find_trim.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dynamics.aircraft import Aircraft
from src.utils.units import m_to_ft


def test_trim(throttle, elevator, alt_ft=9842.0, speed_kts=400, duration_s=60):
    """Fly straight for duration_s and return final speed and altitude trend."""
    ac = Aircraft()
    ac.reset(alt_ft=alt_ft, heading_deg=90.0, speed_kts=speed_kts, trim=False)

    speeds = []
    alts = []
    vz_values = []
    dt = 1.0 / 60.0

    for step in range(int(duration_s / dt)):
        ac.set_controls(throttle=throttle, elevator=elevator, aileron=0.0, rudder=0.0)
        ac.run()
        if step % 30 == 0:
            s = ac.state
            speeds.append(s["airspeed_mps"])
            alts.append(s["alt_m"])
            vz_values.append(-s["w_fps"] * 0.3048)

    speeds = np.array(speeds)
    alts = np.array(alts)
    vz_values = np.array(vz_values)

    # Average speed over last 10s (near steady-state)
    final_speed = float(np.mean(speeds[-20:]))
    final_alt = float(np.mean(alts[-20:]))

    # Vertical speed trend over last half
    alt_trend = (alts[-1] - alts[len(alts)//2]) / (duration_s / 2)

    return {
        "final_speed_mps": final_speed,
        "final_alt_m": final_alt,
        "alt_trend_mps": alt_trend,
        "vz_mean": float(np.mean(vz_values[-20:])),
        "speed_std": float(np.std(speeds[-20:])),
    }


def main():
    print("=" * 70)
    print("Elevator Trim Scan: Find level-flight elevator for throttle=1.0")
    print(f"Target altitude: 3000 m (9842 ft), starting speed: 400 kts (206 m/s)")
    print("=" * 70)

    throttle = 1.0
    print(f"\n{'Elev':>8s}  {'FinalSpd':>9s}  {'FinalAlt':>9s}  {'AltTrend':>9s}  {'VzMean':>8s}  {'SpdStd':>7s}")
    print("-" * 65)

    best_elev = None
    best_trend = float('inf')

    for elevator in [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.02, 0.04, 0.06]:
        r = test_trim(throttle, elevator, alt_ft=9842.0, speed_kts=400, duration_s=40)

        marker = ""
        if abs(r["alt_trend_mps"]) < abs(best_trend):
            best_trend = r["alt_trend_mps"]
            best_elev = elevator
        if abs(r["alt_trend_mps"]) < 2.0:
            marker = " <-- LEVEL"

        print(f"{elevator:8.3f}  {r['final_speed_mps']:9.1f}  {r['final_alt_m']:9.0f}  "
              f"{r['alt_trend_mps']:9.2f}  {r['vz_mean']:8.2f}  {r['speed_std']:7.2f}{marker}")

    print(f"\nBest level-flight elevator at throttle=1.0: {best_elev:.3f}")

    # Now retest with that elevator for longer duration
    print(f"\n{'='*70}")
    print(f"Long-duration test: throttle={throttle}, elevator={best_elev:.3f}")
    r = test_trim(throttle, best_elev, alt_ft=9842.0, speed_kts=400, duration_s=120)
    print(f"  Final speed: {r['final_speed_mps']:.1f} m/s ({r['final_speed_mps']*1.944:.0f} kts)")
    print(f"  Final altitude: {r['final_alt_m']:.0f} m")
    print(f"  Altitude trend: {r['alt_trend_mps']:.2f} m/s")
    print(f"  Vz mean: {r['vz_mean']:.2f} m/s")

    # Also test: can we reach higher speed by starting fast?
    print(f"\n{'='*70}")
    print("High-speed entry test: start at 600 kts (309 m/s)")
    r = test_trim(throttle, best_elev, alt_ft=9842.0, speed_kts=600, duration_s=60)
    print(f"  Final speed: {r['final_speed_mps']:.1f} m/s ({r['final_speed_mps']*1.944:.0f} kts)")
    print(f"  Altitude trend: {r['alt_trend_mps']:.2f} m/s")

    # Test at lower altitude where thrust is higher
    print(f"\n{'='*70}")
    print("Lower altitude test: 1500m (4921 ft)")
    for elevator in [-0.08, -0.06, -0.05, -0.04, -0.03, -0.02, 0.00]:
        r = test_trim(throttle, elevator, alt_ft=4921.0, speed_kts=400, duration_s=30)
        marker = ""
        if abs(r["alt_trend_mps"]) < 2.0:
            marker = " <-- LEVEL"
        print(f"  elev={elevator:6.3f}  spd={r['final_speed_mps']:6.0f}m/s  "
              f"altTrend={r['alt_trend_mps']:6.2f}m/s  vz={r['vz_mean']:6.2f}{marker}")


if __name__ == "__main__":
    import logging, warnings
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    main()
