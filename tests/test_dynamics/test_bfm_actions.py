"""Unit tests for BFM action definitions."""

import numpy as np
import pytest

from src.dynamics.bfm_actions import (
    BFM_ACTION_MAPPING,
    NUM_BFM_ACTIONS,
    get_bfm_action,
    describe_bfm_action,
)


class TestBFMActionMapping:
    """Verify the 13 discrete BFM actions."""

    def test_num_actions(self):
        """Exactly 13 BFM actions as in the original project."""
        assert NUM_BFM_ACTIONS == 13
        assert len(BFM_ACTION_MAPPING) == 13

    def test_all_indices_present(self):
        """All indices 0–12 should have an entry."""
        for i in range(13):
            assert i in BFM_ACTION_MAPPING

    def test_get_bfm_action_valid(self):
        """Valid indices return proper (n_x, n_n, mu) tuples."""
        n_x, n_n, mu = get_bfm_action(0)
        assert n_x == 0.0
        assert n_n == 1.0
        assert mu == 0.0

        n_x, n_n, mu = get_bfm_action(3)
        assert n_x == 0.0
        assert n_n == 8.0
        assert mu == 0.0

        n_x, n_n, mu = get_bfm_action(5)
        assert n_x == 0.0
        assert n_n == 8.0
        assert mu == pytest.approx(np.pi / 3.0)

    def test_get_bfm_action_invalid_fallback(self):
        """Out-of-range indices fall back to action 0."""
        n_x, n_n, mu = get_bfm_action(999)
        assert (n_x, n_n, mu) == (0.0, 1.0, 0.0)

    def test_describe_all_actions(self):
        """Every action should have a non-empty description."""
        for i in range(13):
            desc = describe_bfm_action(i)
            assert isinstance(desc, str)
            assert len(desc) > 0
            assert desc != f"Unknown action {i}"

    def test_action_semantics(self):
        """Verify expected semantics of key maneuvers."""
        # Action 0: wings-level level flight
        n_x, n_n, mu = get_bfm_action(0)
        assert mu == 0.0
        assert abs(n_n) == 1.0

        # Action 9: gentle right turn (60° right bank, 2G)
        n_x, n_n, mu = get_bfm_action(9)
        assert mu < 0  # right roll (negative)
        assert n_n == 2.0

        # Action 10: gentle left turn (60° left bank, 2G)
        n_x, n_n, mu = get_bfm_action(10)
        assert mu > 0  # left roll (positive)
        assert n_n == 2.0
