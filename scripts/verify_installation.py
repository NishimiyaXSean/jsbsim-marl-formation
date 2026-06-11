"""Verify JSBSim installation and F-16 model.

Run: python scripts/verify_installation.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.aircraft import Aircraft


def main():
    print("=" * 50)
    print("JSBSim F-16 Installation Verification")
    print("=" * 50)

    # 1. Check JSBSim import
    print("\n[1/4] Checking JSBSim Python bindings...")
    try:
        import jsbsim
        print(f"  OK: jsbsim {jsbsim.__version__ if hasattr(jsbsim, '__version__') else '(version unknown)'}")
    except ImportError:
        print("  FAIL: jsbsim not installed. Run: pip install jsbsim")
        sys.exit(1)

    # 2. Check aircraft data
    print("\n[2/4] Checking aircraft data...")
    data_dir = os.environ.get("JSBSIM_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "jsbsim"))
    if not os.path.isdir(data_dir):
        print(f"  WARN: data/jsbsim/ not found at {data_dir}")
        print("  Download from: https://github.com/JSBSim-Team/jsbsim")
        print("  Place aircraft/, engines/, systems/ under data/jsbsim/")
        print("  Continuing with JSBSim default search path...")
        data_dir = None
    else:
        print(f"  OK: data directory found at {data_dir}")

    # 3. Create aircraft and load F-16
    print("\n[3/4] Loading F-16 model...")
    try:
        ac = Aircraft(jsbsim_data_dir=data_dir)
        print(f"  OK: F-16 model loaded successfully")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # 4. Run a short simulation
    print("\n[4/4] Running 1000-step test simulation...")
    try:
        ac.reset(alt_ft=15000.0, heading_deg=90.0, speed_kts=400.0)

        # Straight and level for 500 steps
        for _ in range(500):
            ac.set_controls(throttle=0.8, elevator=0.0, aileron=0.0, rudder=0.0)
            ac.run()

        s = ac.state
        print(f"  After 500 steps (level flight):")
        print(f"    Altitude: {s['alt_ft']:.0f} ft ({s['alt_m']:.0f} m)")
        print(f"    Airspeed: {s['airspeed_kts']:.0f} kts ({s['airspeed_mps']:.0f} m/s)")
        print(f"    Roll/Pitch/Yaw: {s['roll_deg']:.1f}°/{s['pitch_deg']:.1f}°/{s['yaw_deg']:.1f}°")
        print(f"    Nz (G): {s['n_z_g']:.2f}")

        # Gentle left turn for 500 steps
        for _ in range(500):
            ac.set_controls(throttle=0.8, elevator=0.05, aileron=-0.3, rudder=0.05)
            ac.run()

        s2 = ac.state
        print(f"\n  After 500 more steps (left turn):")
        print(f"    Altitude: {s2['alt_ft']:.0f} ft")
        print(f"    Airspeed: {s2['airspeed_kts']:.0f} kts")
        print(f"    Roll/Pitch/Yaw: {s2['roll_deg']:.1f}°/{s2['pitch_deg']:.1f}°/{s2['yaw_deg']:.1f}°")
        print(f"    Heading change: {s2['yaw_deg'] - s['yaw_deg']:.1f}°")

        print("\n" + "=" * 50)
        print("ALL CHECKS PASSED!")
        print("JSBSim F-16 is ready for use.")
        print("=" * 50)

    except Exception as e:
        print(f"  FAIL during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
