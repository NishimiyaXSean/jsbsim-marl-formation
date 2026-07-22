"""Modular reward function composables for formation tasks.

Each class implements __call__(task, env) → Dict[str, float]
and can be independently configured, tested, and replaced.

Inspired by LAG's BaseRewardFunction design.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants (extracted from FormationTask for module self-containment)
# ═══════════════════════════════════════════════════════════════════════════════

_DT = 0.2          # decision interval (s)
_DECISION_STEPS = 12
_MAX_DIST = 10000.0
_MAX_VEL = 400.0
_COOP_PHASE_AND = 1


# ═══════════════════════════════════════════════════════════════════════════════

class BaseRewardFunction:
    """Abstract reward function with config-driven weight and potential-based shaping."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def __call__(self, task, env) -> Dict[str, float]:
        raise NotImplementedError

    def _scale(self) -> float:
        return float(self.config.get(f"{self.__class__.__name__}_scale", 1.0))


# ═══════════════════════════════════════════════════════════════════════════════

class ProgressReward(BaseRewardFunction):
    """Reward for closing distance to target."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._weight = self.config.get("progress_weight", 1.0)

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {}
        t_pos = env.targets[0].aircraft.position_ned
        for aid, ps in zip(task._agent_ids, env.pursuers):
            cur_dist = float(np.linalg.norm(ps.aircraft.position_ned - t_pos))
            delta = ps.prev_dist - cur_dist
            r = self._weight * delta * 0.5 * _DECISION_STEPS
            if cur_dist < 500.0:
                r += self._weight * delta * 5.0 * _DECISION_STEPS
            rewards[aid] = r
            ps.prev_dist = cur_dist
        return rewards


class ATAAlignmentReward(BaseRewardFunction):
    """Reward for nose-on-target (cos_ATA)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._weight = self.config.get("ata_weight", 8.0)

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {}
        t_pos = env.targets[0].aircraft.position_ned
        for aid, ps in zip(task._agent_ids, env.pursuers):
            cur_dist = float(np.linalg.norm(ps.aircraft.position_ned - t_pos))
            a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            t_fwd = compute_forward_vector(env.targets[0].aircraft.rpy_rad)
            _, los_dir, _ = compute_los(ps.aircraft.position_ned, t_pos)
            geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
            cos_ata = geo.get('cos_ata', 0.0)
            dist_factor = np.clip(1.0 - cur_dist / _MAX_DIST, 0.1, 1.0)
            rewards[aid] = self._weight * cos_ata * dist_factor * _DECISION_STEPS
        return rewards


class ProximityTierReward(BaseRewardFunction):
    """One-time bonuses for crossing distance milestones."""

    _TIERS = [(800.0, 25.0), (500.0, 50.0), (300.0, 100.0)]

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {aid: 0.0 for aid in task._agent_ids}
        t_pos = env.targets[0].aircraft.position_ned
        for aid, ps in zip(task._agent_ids, env.pursuers):
            cur_dist = float(np.linalg.norm(ps.aircraft.position_ned - t_pos))
            for tier_dist, bonus in self._TIERS:
                if cur_dist < tier_dist and tier_dist not in ps.proximity_awarded:
                    rewards[aid] += bonus
                    ps.proximity_awarded.add(tier_dist)
        return rewards


class StepPenaltyReward(BaseRewardFunction):
    """Small penalty per decision step to discourage time-wasting."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._penalty = self.config.get("step_penalty", 0.25)

    def __call__(self, task, env) -> Dict[str, float]:
        return {aid: -self._penalty * _DECISION_STEPS for aid in task._agent_ids}


class PincerShapingReward(BaseRewardFunction):
    """Cooperative pincer angle bonus in AND-gate phase."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._coeff = self.config.get("pincer_coeff", 35.0)

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {aid: 0.0 for aid in task._agent_ids}
        if getattr(task, '_coop_phase', 0) != _COOP_PHASE_AND:
            return rewards

        t_pos = env.targets[0].aircraft.position_ned
        los0 = t_pos - env.pursuers[0].aircraft.position_ned
        los1 = t_pos - env.pursuers[1].aircraft.position_ned
        d0, d1 = float(np.linalg.norm(los0)), float(np.linalg.norm(los1))
        and_dist = getattr(task, '_and_dist', 800.0)
        and_angle = getattr(task, '_and_angle', 30.0)

        if d0 < and_dist and d1 < and_dist:
            cos_pincer = np.dot(los0, los1) / max(d0 * d1, 1e-6)
            pincer_angle = float(np.degrees(np.arccos(np.clip(cos_pincer, -1, 1))))
            pincer_reward = self._coeff * min(pincer_angle, and_angle) * _DT * _DECISION_STEPS
            rewards["p0"] = pincer_reward
            rewards["p1"] = pincer_reward
            # Store for cooperative success termination check
            task._last_pincer = pincer_angle
        else:
            task._last_pincer = 0.0

        return rewards


class DistanceAsymmetryPenalty(BaseRewardFunction):
    """Penalize large distance gap between pursuers (anti-free-riding)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._threshold = self.config.get("asymmetry_threshold", 800.0)
        self._weight = self.config.get("asymmetry_weight", 0.3)
        self._norm = self.config.get("asymmetry_norm", 1000.0)

    def __call__(self, task, env) -> Dict[str, float]:
        t_pos = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - t_pos))
        d1 = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - t_pos))
        dist_diff = abs(d0 - d1)
        penalty = 0.0
        if dist_diff > self._threshold:
            penalty = self._weight * (dist_diff - self._threshold) / self._norm * _DT * _DECISION_STEPS
        return {"p0": -penalty, "p1": -penalty}


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-pursuit reward modules (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════════

class GroundWarningReward(BaseRewardFunction):
    """Linear penalty when flying below safe altitude (prevents CFIT)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._safe_alt = self.config.get("ground_safe_alt", 300.0)
        self._weight = self.config.get("ground_warning_weight", 0.1)

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {}
        for aid, ps in zip(task._agent_ids, env.pursuers):
            alt_m = float(ps.aircraft.state["alt_m"])
            if alt_m < self._safe_alt:
                rewards[aid] = -self._weight * (self._safe_alt - alt_m) / self._safe_alt
            else:
                rewards[aid] = 0.0
        return rewards


class LowSpeedTurnPenalty(BaseRewardFunction):
    """Penalize aggressive turns at low airspeed (stall-risk suppression)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._speed_warn = self.config.get("low_speed_warn", 150.0)
        self._weight = self.config.get("low_speed_turn_weight", 0.5)

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {}
        for aid, ps in zip(task._agent_ids, env.pursuers):
            airspeed = float(ps.aircraft.state["airspeed_mps"])
            last_actions = getattr(task, '_last_actions', {})
            act = last_actions.get(aid, {})
            delta_hdg = abs(act.get('delta_heading', 0.0))
            # Penalize only harsh turns (±30°) at low speed
            if airspeed < self._speed_warn and delta_hdg >= 30.0:
                rewards[aid] = -self._weight * (self._speed_warn - airspeed) / self._speed_warn
            else:
                rewards[aid] = 0.0
        return rewards


class CaptureSuccessReward(BaseRewardFunction):
    """One-shot bonus when pursuer reaches close range BEHIND target (dist + ATA)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._bonus = self.config.get("capture_bonus", 5000.0)
        self._dist_thresh = self.config.get("capture_dist", 300.0)
        self._ata_thresh = self.config.get("capture_ata", 30.0)  # degrees

    def __call__(self, task, env) -> Dict[str, float]:
        rewards = {}
        if env.M < 1:
            return {aid: 0.0 for aid in task._agent_ids}
        t_pos = env.targets[0].aircraft.position_ned
        for aid, ps in zip(task._agent_ids, env.pursuers):
            a_pos = ps.aircraft.position_ned
            cur_dist = float(np.linalg.norm(a_pos - t_pos))

            # Compute ATA: angle between pursuer nose and LOS to target
            a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
            los_vec = t_pos - a_pos
            los_norm = np.linalg.norm(los_vec) + 1e-6
            cos_ata = float(np.dot(a_fwd, los_vec / los_norm))
            ata_deg = float(np.degrees(np.arccos(np.clip(cos_ata, -1.0, 1.0))))

            awarded = getattr(ps, '_capture_awarded', False)
            if not awarded and cur_dist < self._dist_thresh and ata_deg < self._ata_thresh:
                rewards[aid] = self._bonus
                ps._capture_awarded = True
            else:
                rewards[aid] = 0.0
        return rewards
