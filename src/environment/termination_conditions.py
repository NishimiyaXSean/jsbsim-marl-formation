"""Modular termination condition composables for formation tasks.

Each class implements __call__(task, env) → Optional[str]
Return a reason string if the condition is triggered, None otherwise.

Inspired by LAG's termination_conditions design.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════

class BaseTerminationCondition:
    """Abstract termination check."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def __call__(self, task, env) -> Optional[str]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════

class FlightEnvelopeTermination(BaseTerminationCondition):
    """Check stall, ground proximity, altitude limits, overload."""

    def __call__(self, task, env) -> Optional[str]:
        for ps, aid in zip(env.pursuers, task._agent_ids):
            alt_m = float(ps.aircraft.state["alt_m"])
            airspeed = float(ps.aircraft.state["airspeed_mps"])
            nz_g = float(ps.aircraft.state.get("n_z_g", 1.0))

            if alt_m < 100.0:
                return f"{aid}_low_altitude"
            if alt_m > 5000.0:
                return f"{aid}_high_altitude"
            if airspeed < 100.0:
                return f"{aid}_stall"
            if abs(nz_g) > 9.0:
                return f"{aid}_overload"
        return None


class TimeoutTermination(BaseTerminationCondition):
    """Episode length limit."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._max_steps = self.config.get("max_steps", 1000)

    def __call__(self, task, env) -> Optional[str]:
        step_counter = getattr(task, '_step_counter', env._step_counter)
        if step_counter >= self._max_steps:
            return "timeout"
        return None


class CooperativeSuccessTermination(BaseTerminationCondition):
    """OR-gate and AND-gate cooperative success detection."""

    _COOP_PHASE_OR = 0
    _COOP_PHASE_AND = 1
    _OR_DIST = 200.0
    _AND_ANGLE = 30.0
    _SUSTAIN_STEPS = 6

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._or_dist = self.config.get("or_dist", 200.0)

    def __call__(self, task, env) -> Optional[str]:
        t_pos = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - t_pos))
        d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - t_pos))

        coop_phase = getattr(task, '_coop_phase', self._COOP_PHASE_OR)

        if coop_phase == self._COOP_PHASE_OR:
            if d0 < self._or_dist or d1 < self._or_dist:
                return "cooperative_success_or"
        elif coop_phase == self._COOP_PHASE_AND:
            and_dist = getattr(task, '_and_dist', 800.0)
            pincer = getattr(task, '_last_pincer', 0.0)
            if d0 < and_dist and d1 < and_dist and pincer > self._AND_ANGLE:
                counter = getattr(task, '_coop_sustain_counter', 0) + 1
                task._coop_sustain_counter = counter
                if counter >= self._SUSTAIN_STEPS:
                    return "cooperative_success_and"
            else:
                task._coop_sustain_counter = 0
        return None
