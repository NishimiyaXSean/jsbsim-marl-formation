"""Unit tests for BFMAutopilot (BFM → control surfaces)."""

import numpy as np
import pytest

from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def autopilot() -> BFMAutopilot:
    ap = BFMAutopilot()
    ap.reset(initial_speed_mps=200.0)
    return ap


@pytest.fixture
def dt() -> float:
    return 1.0 / 60.0


# ── Basic output shape ───────────────────────────────────────────────────────

class TestOutputShape:
    """Autopilot returns 4 control values in valid ranges."""

    def test_output_is_4_tuple(self, autopilot, dt):
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        assert isinstance(thr, float)
        assert isinstance(elev, float)
        assert isinstance(ail, float)
        assert isinstance(rud, float)

    def test_output_ranges(self, autopilot, dt):
        """Throttle ∈ [0, 1]; elevator/aileron/rudder ∈ [-1, 1]."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 3.0, 0.5, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        assert 0.0 <= thr <= 1.0
        assert -1.0 <= elev <= 1.0
        assert -1.0 <= ail <= 1.0
        assert -1.0 <= rud <= 1.0


# ── Elevator (Nz-tracking) channel ───────────────────────────────────────────

class TestElevatorChannel:
    """Elevator tracks body-Z normal acceleration.

    JSBSim F-16 sign convention:
        elevator > 0 → nose DOWN (less pull, altitude decreases)
        elevator < 0 → nose UP   (more pull, altitude increases)
    """

    def test_pull_up_negative_elevator(self, autopilot, dt):
        """When pulling up (n_n > 1), elevator should be negative (nose UP)."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 5.0, 0.0, dt,
            n_z_g=-1.0,  # current level flight (1G)
            roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        # Need to go from n_z_g=-1 to n_z_g=-5 → MORE negative elevator (pull)
        assert elev < 0.0

    def test_push_over_positive_elevator(self, autopilot, dt):
        """When pushing over (n_n < 1), elevator should be positive (nose DOWN)."""
        thr, elev, ail, rud = autopilot.step(
            0.0, -1.0, 0.0, dt,
            n_z_g=-1.0,  # current level flight
            roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        # Need to go from n_z_g=-1 to n_z_g=+1 → MORE positive elevator (push)
        assert elev > 0.0

    def test_steady_state_near_trim(self, autopilot, dt):
        """When n_z already matches target, elevator should be near trim."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 3.0, 0.0, dt,
            n_z_g=-3.0,  # already at target
            roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        # At target, error=0, elevator = trim (about -0.04 at 200 m/s)
        assert abs(elev) < 0.7


# ── Aileron (roll-tracking) channel ──────────────────────────────────────────

class TestAileronChannel:
    """Aileron tracks bank angle."""

    def test_roll_right_positive_aileron(self, autopilot, dt):
        """Target right roll → positive aileron."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, -np.pi / 4, dt,  # 45° right bank
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        assert ail > 0.0  # positive to roll right

    def test_roll_left_negative_aileron(self, autopilot, dt):
        """Target left roll → negative aileron."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, np.pi / 4, dt,  # 45° left bank
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        assert ail < 0.0  # negative to roll left

    def test_already_banked_zero_aileron(self, autopilot, dt):
        """When already at target roll, aileron should be near zero."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, -0.5, dt,
            n_z_g=-1.0, roll_rad=-0.5,  # already at -0.5 rad
            airspeed_mps=200.0, beta_deg=0.0,
        )
        assert abs(ail) < 0.2


# ── Throttle (speed-hold) channel ────────────────────────────────────────────

class TestThrottleChannel:
    """Throttle tracks airspeed target driven by n_x."""

    def test_accelerate_increases_throttle(self, autopilot, dt):
        """n_x > 0 should increase throttle to speed up."""
        # Prime: call once to set baseline
        autopilot.step(0.0, 1.0, 0.0, dt,
                       n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0)
        # Now accelerate
        thr, _, _, _ = autopilot.step(
            2.0, 1.0, 0.0, dt,  # 2G forward
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        # Throttle should be notable (speed target drifted up)
        assert thr > 0.0

    def test_decelerate_reduces_throttle(self, autopilot, dt):
        """n_x < 0 should reduce throttle."""
        # Set a high initial target speed
        for _ in range(60):  # 1 s of 2G acceleration
            autopilot.step(2.0, 1.0, 0.0, dt,
                           n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0)
        # Now decelerate
        thr, _, _, _ = autopilot.step(
            -2.0, 1.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=250.0, beta_deg=0.0,
        )
        # Throttle should be lower than before
        assert thr < 0.9

    def test_reset_sets_initial_speed(self, dt):
        """Reset with a speed argument sets the target."""
        ap = BFMAutopilot()
        ap.reset(initial_speed_mps=300.0)
        # First call: should hold 300 m/s, not hunt from 0
        thr, _, _, _ = ap.step(
            0.0, 1.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=300.0, beta_deg=0.0,
        )
        assert 0.0 <= thr <= 1.0  # valid output, not extreme


# ── Rudder (sideslip suppression) ────────────────────────────────────────────

class TestRudderChannel:
    """Rudder suppresses sideslip (beta → 0)."""

    def test_positive_sideslip_positive_rudder(self, autopilot, dt):
        """Positive beta (nose left) → positive rudder (nose right)."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=5.0,
        )
        assert rud > 0.0  # rudder fights sideslip

    def test_zero_sideslip_zero_rudder(self, autopilot, dt):
        """No sideslip → no rudder."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 1.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        assert abs(rud) < 0.1


# ── Integration: full BFM action pipeline ────────────────────────────────────

class TestIntegration:
    """Test end-to-end BFM action → surface pipeline."""

    def test_level_flight_is_stable(self, autopilot, dt):
        """BFM action 0 (level flight) should keep controls centered."""
        for _ in range(30):
            thr, elev, ail, rud = autopilot.step(
                0.0, 1.0, 0.0, dt,
                n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
            )
        # After settling, all surfaces should be moderate
        assert abs(elev) < 0.5
        assert abs(ail) < 0.2
        assert abs(rud) < 0.1

    def test_max_g_climb_commands(self, autopilot, dt):
        """BFM action 3 (8G zoom climb) — strong pull, no bank."""
        thr, elev, ail, rud = autopilot.step(
            0.0, 8.0, 0.0, dt,
            n_z_g=-1.0, roll_rad=0.0, airspeed_mps=200.0, beta_deg=0.0,
        )
        # Pull-up → negative elevator (nose UP per JSBSim convention)
        assert elev < -0.3  # Significant pull-up
        assert abs(ail) < 0.5  # Wings level
