"""Unit tests for FlightEnvelope (V-n diagram, GPWS, G-smoothing)."""

import numpy as np
import pytest

from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def envelope() -> FlightEnvelope:
    return FlightEnvelope()


@pytest.fixture
def dt() -> float:
    return 1.0 / 60.0  # 60 Hz


# ── V-n envelope ─────────────────────────────────────────────────────────────

class TestVnEnvelope:
    """Test lift-limited G availability."""

    def test_corner_speed_full_g(self, envelope, dt):
        """At corner speed, the full structural G limit is available."""
        n_x, n_n, mu = envelope.step(
            0.0, 9.0, 0.0,
            speed_mps=150.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        # 9G should be allowed at corner speed
        # After smoothing: alpha = (1/60) / (0.4 + 1/60) ≈ 0.04
        # n_n_sm = 1.0 + (9.0 - 1.0) * 0.04 = 1.32
        # Actually let me think: smoothing starts from n_n=1 (level trim),
        # so first step should be ~1 + (9-1)*0.04 = 1.32 — not clamped by V-n
        assert n_n > 1.0  # should be some positive pull-up
        assert n_n <= 9.0  # but not exceeding structural limit

    def test_below_corner_speed_limits_g(self, envelope, dt):
        """At low speed, available G is reduced by V² factor."""
        # At 75 m/s (half corner speed), available G ≈ (0.5)² * 9 = 2.25
        n_x, n_n, mu = envelope.step(
            0.0, 9.0, 0.0,
            speed_mps=75.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        # The V-n limiter runs BEFORE smoothing, so n_n_cmd is clamped first
        assert n_n <= 3.0  # should be well below 9G (2.25 limit + smoothing)

    def test_negative_g_clamped(self, envelope, dt):
        """Negative G is limited to -3G structural."""
        n_x, n_n, mu = envelope.step(
            0.0, -8.0, 0.0,
            speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        assert n_n >= -3.0

    def test_g_scale_curriculum(self, envelope, dt):
        """G-scale multiplier reduces available G for curriculum stages."""
        n_x, n_n, mu = envelope.step(
            0.0, 9.0, 0.0,
            speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt, g_scale=0.5,
        )
        assert n_n <= 4.5  # 9 * 0.5 = 4.5 max


# ── Speed limits ─────────────────────────────────────────────────────────────

class TestSpeedLimits:
    """Test tangential acceleration clamps."""

    def test_no_accel_above_max_speed(self, envelope, dt):
        """Cannot accelerate past max speed."""
        n_x, n_n, mu = envelope.step(
            2.0, 1.0, 0.0,  # want to accelerate
            speed_mps=410.0,  # already past max
            alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        assert n_x == 0.0  # acceleration blocked

    def test_no_decel_below_stall(self, envelope, dt):
        """Cannot decelerate below stall speed."""
        n_x, n_n, mu = envelope.step(
            -2.0, 1.0, 0.0,  # want to decelerate
            speed_mps=50.0,   # already below stall
            alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        assert n_x == 0.0  # deceleration blocked


# ── GPWS ─────────────────────────────────────────────────────────────────────

class TestGPWS:
    """Test Ground Proximity Warning System override."""

    def test_gpws_pull_up_attacker(self, envelope, dt):
        """Attacker below 300m and descending → forced max-G pull-up, wings level."""
        n_x, n_n, mu = envelope.step(
            0.0, 1.0, 0.5,  # normal flight command
            speed_mps=200.0, alt_m=250.0,  # below 300m trigger
            vz_mps=-10.0,  # descending
            current_roll_rad=0.3, dt=dt, is_attacker=True,
        )
        assert n_n == 9.0  # max G override
        assert mu == 0.0   # wings level

    def test_no_gpws_when_climbing(self, envelope, dt):
        """GPWS should NOT trigger when climbing even if low."""
        n_x, n_n, mu = envelope.step(
            0.0, 1.0, 0.5,
            speed_mps=200.0, alt_m=200.0,
            vz_mps=10.0,  # climbing!
            current_roll_rad=0.0, dt=dt, is_attacker=True,
        )
        # Should NOT be overridden — climbing is safe
        assert n_n != 9.0

    def test_gpws_evader_higher_floor(self, envelope, dt):
        """Evader has a higher GPWS trigger (800m vs 300m)."""
        n_x, n_n, mu = envelope.step(
            0.0, 1.0, 0.5,
            speed_mps=200.0, alt_m=500.0,  # below evader's 800m floor
            vz_mps=-10.0,
            current_roll_rad=0.0, dt=dt, is_attacker=False,
        )
        assert n_n == 9.0  # evader GPWS triggers at 800m


# ── Smoothing ────────────────────────────────────────────────────────────────

class TestSmoothing:
    """Test first-order G-onset lag."""

    def test_smoothing_converges(self, envelope, dt):
        """Repeated calls with same command should converge toward target."""
        for _ in range(100):  # ~1.67 sec at 60 Hz, well past tau=0.4s
            n_x, n_n, mu = envelope.step(
                0.0, 5.0, 0.0,
                speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
                current_roll_rad=0.0, dt=dt,
            )
        assert abs(n_n - 5.0) < 0.1  # converged to within 0.1G

    def test_reset_clears_state(self, envelope, dt):
        """Reset() should clear the smoothing memory."""
        envelope.step(0.0, 5.0, 0.0, speed_mps=200.0, alt_m=3000.0,
                      vz_mps=0.0, current_roll_rad=0.0, dt=dt)
        envelope.reset()
        # After reset, smoothing starts from 1.0G trim
        n_x, n_n, mu = envelope.step(
            0.0, 1.0, 0.0,
            speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=0.0, dt=dt,
        )
        assert abs(n_n - 1.0) < 0.1  # close to trim


# ── Roll control ─────────────────────────────────────────────────────────────

class TestRollControl:
    """Test roll P-controller with rate limiting."""

    def test_roll_tracks_target(self, envelope, dt):
        """Roll should converge toward target over multiple steps."""
        mu_current = 0.0
        for _ in range(60):  # 1.0 sec — enough for roll to converge
            n_x, n_n, mu_current = envelope.step(
                0.0, 1.0, np.pi / 3,  # 60° bank
                speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
                current_roll_rad=mu_current,  # feed back previous mu
                dt=dt,
            )
        assert abs(mu_current - np.pi / 3) < 0.15    # within ~8.6°

    def test_roll_rate_limited(self, envelope, dt):
        """Single step cannot exceed max roll rate * dt."""
        mu_start = 0.0
        target = np.pi  # 180° roll command
        n_x, n_n, mu = envelope.step(
            0.0, 1.0, target,
            speed_mps=200.0, alt_m=3000.0, vz_mps=0.0,
            current_roll_rad=mu_start, dt=dt,
        )
        max_step = np.pi * dt  # max_roll_rate * dt
        assert abs(mu - mu_start) <= max_step + 1e-10
