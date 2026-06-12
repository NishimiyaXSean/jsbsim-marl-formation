"""BFM action definitions — full 13-action combat set + 9-action pursuit subset.

Each action is a tuple of ``(n_x, n_n, mu)``:

``n_x``
    Tangential acceleration in G.  Positive → speed up; negative → slow down.
``n_n``
    Normal acceleration in G.  Positive → pull up (positive-G turn / climb);
    negative → push over (negative-G dive).
``mu``
    Bank angle in radians.  0 = wings level;  +π/3 = 60° left;  -π/3 = 60° right.

**Pursuit subset** — 9 safe, low-G actions designed for single-agent pursuit training:
    speed control (maintain / accelerate / decelerate)
    + steering (gentle left / right turns)
    + altitude (gentle climb / descent)
    + combined (accelerating turns)
No high-G (>3G) or diving-turn manoeuvres that cause ground strikes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# ── Full 13-action combat set ──────────────────────────────────────────────────

BFM_ACTION_MAPPING: Dict[int, Tuple[float, float, float]] = {
    0:  ( 0.0,  1.0,  0.0),              # a1:  Level flight (1G)
    1:  ( 2.0,  1.0,  0.0),              # a2:  Accelerate straight
    2:  (-2.0,  1.0,  0.0),              # a3:  Decelerate straight
    3:  ( 0.0,  8.0,  0.0),              # a4:  Max-G zoom climb (8G pull-up)
    4:  ( 0.0, -8.0,  0.0),              # a5:  Max-G split-S (8G push-down)
    5:  ( 0.0,  8.0,  np.pi / 3.0),      # a6:  Left climbing turn (8G, 60° left)
    6:  ( 0.0, -8.0, -np.pi / 3.0),      # a7:  Right diving turn (8G, 60° right)
    7:  ( 0.0,  8.0, -np.pi / 3.0),      # a8:  Right climbing turn (8G, 60° right)
    8:  ( 0.0, -8.0,  np.pi / 3.0),      # a9:  Left diving turn (8G, 60° left)
    9:  ( 0.0,  2.0, -np.pi / 3.0),      # a10: Gentle right turn (2G, 60° right)
    10: ( 0.0,  2.0,  np.pi / 3.0),      # a11: Gentle left turn (2G, 60° left)
    11: ( 0.0,  3.0,  0.0),              # a12: Gentle climb (3G pull-up)
    12: ( 0.0, -3.0,  0.0),              # a13: Gentle dive (3G push-down)
}

NUM_BFM_ACTIONS: int = len(BFM_ACTION_MAPPING)

# ── 9-action pursuit subset — safe, low-G, no diving turns ─────────────────────

PURSUIT_ACTIONS: Dict[int, Tuple[float, float, float]] = {
    0:  ( 0.0,  1.0,  0.0),              # p1:  Level flight (1G)
    1:  ( 2.0,  1.0,  0.0),              # p2:  Accelerate straight
    2:  (-2.0,  1.0,  0.0),              # p3:  Decelerate straight
    3:  ( 0.0,  2.0, -np.pi / 3.0),      # p4:  Gentle right turn (2G, 60° right)
    4:  ( 0.0,  2.0,  np.pi / 3.0),      # p5:  Gentle left turn (2G, 60° left)
    5:  ( 0.0,  3.0,  0.0),              # p6:  Gentle climb (3G pull-up)
    6:  ( 0.0, -2.0,  0.0),              # p7:  Gentle descent (2G push-down)
    7:  ( 1.0,  2.0, -np.pi / 3.0),      # p8:  Accelerating right turn
    8:  ( 1.0,  2.0,  np.pi / 3.0),      # p9:  Accelerating left turn
}

NUM_PURSUIT_ACTIONS: int = len(PURSUIT_ACTIONS)

# ── Descriptions ────────────────────────────────────────────────────────────────

_BFM_DESCRIPTIONS: Dict[int, str] = {
    0:  "Level flight (1G)",
    1:  "Accelerate straight",
    2:  "Decelerate straight",
    3:  "Max-G zoom climb",
    4:  "Max-G split-S",
    5:  "Left climbing turn",
    6:  "Right diving turn",
    7:  "Right climbing turn",
    8:  "Left diving turn",
    9:  "Gentle right turn",
    10: "Gentle left turn",
    11: "Gentle climb",
    12: "Gentle dive",
}

_PURSUIT_DESCRIPTIONS: Dict[int, str] = {
    0:  "Level flight",
    1:  "Accelerate",
    2:  "Decelerate",
    3:  "Turn right",
    4:  "Turn left",
    5:  "Climb",
    6:  "Descend",
    7:  "Accel + turn right",
    8:  "Accel + turn left",
}


# ── Public helpers ──────────────────────────────────────────────────────────────

def get_bfm_action(action_idx: int) -> Tuple[float, float, float]:
    """Return ``(n_x, n_n, mu)`` for *action_idx* (0–12).

    Out-of-range indices silently fall back to action 0 (level flight).
    """
    return BFM_ACTION_MAPPING.get(action_idx, BFM_ACTION_MAPPING[0])


def get_pursuit_action(action_idx: int) -> Tuple[float, float, float]:
    """Return ``(n_x, n_n, mu)`` for *action_idx* (0–8) in the pursuit set.

    Out-of-range indices fall back to action 0 (level flight).
    """
    return PURSUIT_ACTIONS.get(action_idx, PURSUIT_ACTIONS[0])


def describe_bfm_action(action_idx: int) -> str:
    """Human-readable label for a BFM action index."""
    return _BFM_DESCRIPTIONS.get(action_idx, f"Unknown action {action_idx}")


def describe_pursuit_action(action_idx: int) -> str:
    """Human-readable label for a pursuit action index."""
    return _PURSUIT_DESCRIPTIONS.get(action_idx, f"Unknown pursuit action {action_idx}")
