"""Test JSBSim Aircraft wrapper."""

from src.dynamics.aircraft import Aircraft


def test_aircraft_creation():
    """Aircraft can be created and F-16 model loaded."""
    ac = Aircraft()
    assert ac.MODEL == "f16"
    assert ac.fdm is not None


def test_aircraft_reset():
    """Aircraft.reset() sets initial conditions correctly."""
    ac = Aircraft()
    ac.reset(alt_ft=15000.0, heading_deg=90.0, speed_kts=400.0)

    s = ac.state
    assert 14000 <= s["alt_ft"] <= 16000  # Allow trim deviation
    assert 300 <= s["airspeed_kts"] <= 500


def test_aircraft_run():
    """Aircraft.run() advances simulation time."""
    ac = Aircraft()
    ac.reset()

    t0 = ac.get_sim_time()
    for _ in range(60):  # 1 second at 60 Hz
        ac.run()
    t1 = ac.get_sim_time()

    assert t1 > t0
    assert abs((t1 - t0) - 1.0) < 0.1


def test_set_controls():
    """set_controls applies values within valid range."""
    ac = Aircraft()
    ac.reset()
    ac.set_controls(throttle=0.5, elevator=-0.2, aileron=0.1, rudder=-0.05)
    ac.run()

    s = ac.state
    assert isinstance(s["alt_ft"], float)
    assert isinstance(s["airspeed_kts"], float)
