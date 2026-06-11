"""13 discrete Basic Fighter Maneuver (BFM) action definitions.

Migrated verbatim from the original PyBullet 1v1 marl_env.py.
Each action is a tuple of ``(n_x, n_n, mu)``:

``n_x``
    Tangential acceleration in G.  Positive → speed up; negative → slow down.
``n_n``
    Normal acceleration in G.  Positive → pull up (positive-G turn / climb);
    negative → push over (negative-G dive).
``mu``
    Bank angle in radians (roll angle of the lift vector about the velocity vector).
    0 = wings level;  +π/3 = 60° left bank;  -π/3 = 60° right bank.

The 13-action design is one of the **innovations** of the original research:
it discretises the continuous 3-DOF air combat control space into
semantically meaningful basic fighter manoeuvres, which dramatically
reduces the RL exploration burden.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# ── Action mapping ───────────────────────────────────────────────────────────

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

# Human-readable descriptions (useful for logging / debugging)
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


# ── Public helpers ───────────────────────────────────────────────────────────

def get_bfm_action(action_idx: int) -> Tuple[float, float, float]:
    """Return ``(n_x, n_n, mu)`` for *action_idx* (0–12).

    Out-of-range indices silently fall back to action 0 (level flight).
    """
    return BFM_ACTION_MAPPING.get(action_idx, BFM_ACTION_MAPPING[0])


def describe_bfm_action(action_idx: int) -> str:
    """Human-readable label for a BFM action index."""
    return _BFM_DESCRIPTIONS.get(action_idx, f"Unknown action {action_idx}")
