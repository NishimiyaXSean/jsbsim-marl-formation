"""BaseEnv — RLlib MultiAgentEnv with physics-only responsibility.

Manages JSBSim aircraft lifecycle and the 12-micro-step physics loop.
All task-specific logic (observation, reward, termination, action masking)
is delegated to a BaseTask instance.

Architecture:
    BaseEnv.step(action_dict):
        1. task.apply_actions(env, action_dict)   # interpret RL actions → PID targets
        2. for _ in range(12):                     # physics micro-step loop
               aircraft.run() + position update
        3. task.step(env)                          # task-level logic
        4. obs = task.get_obs(env)                 # delegate
        5. rewards = task.get_reward(env)
        6. terminateds, truncateds, infos = task.get_termination(env)
        7. return obs, rewards, terminateds, truncateds, infos
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.dynamics.controller_base import FlightTarget
from src.dynamics.pid_controller import PIDFlightController
from src.dynamics.safety_interceptor import SafetyInterceptor
from src.utils.units import kts_to_mps

from .task_base import BaseTask
from .formation_task import FormationTask, DECISION_STEPS, PHYSICS_DT, CTRL_FREQ, N_SPEED

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal aircraft wrapper dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Pursuer:
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    controller: object = None  # SafetyInterceptor or BaseController (set after init)
    ref_hdg: float = 0.0
    ref_alt_m: float = 3000.0
    prev_dist: float = 0.0
    prev_ata_deg: Optional[float] = None
    prev_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_airspeed: float = 180.0
    proximity_awarded: set = field(default_factory=set)
    closure_rates: deque = field(default_factory=lambda: deque(maxlen=35))
    zone_death_counter: int = 0
    loiter_time: float = 0.0
    episode_start_dist: float = 0.0

    def reset_state(self):
        self.prev_ata_deg = None
        self.proximity_awarded.clear()
        self.closure_rates.clear()
        self.zone_death_counter = 0
        self.loiter_time = 0.0
        if self.controller is not None:
            self.controller.reset()


@dataclass
class _Target:
    aircraft: Aircraft
    fc: FlightController
    envelope: FlightEnvelope
    autopilot: BFMAutopilot
    ref_hdg: float = 0.0
    ref_alt_m: float = 3000.0


# ═══════════════════════════════════════════════════════════════════════════════
#  BaseEnv
# ═══════════════════════════════════════════════════════════════════════════════


class BaseEnv(MultiAgentEnv):
    """Generic RLlib-compatible environment for JSBSim air combat.

    Physics responsibility only. Task logic lives in self.task (BaseTask).

    Args:
        task: A BaseTask instance (e.g., FormationTask).
        env_config: Optional dict with keys:
            - jsbsim_data_dir: path to JSBSim aircraft data
            - difficulty_level: float [0, 1] target maneuver difficulty
    """

    metadata = {"name": "base_env_v0"}

    def __init__(self, task: BaseTask | None = None, env_config: dict | None = None):
        super().__init__()
        config = env_config or {}

        # Task (can be set after init for RLlib compatibility)
        self.task = task or FormationTask(config)

        # Build aircraft based on task configuration
        self.N = getattr(self.task, 'N', 2)  # pursuers
        self.M = getattr(self.task, 'M', 1)  # targets
        self._agent_ids = list(self.task.agent_ids)

        # Shared difficulty
        self._difficulty = float(np.clip(config.get("difficulty_level", 0.0), 0.0, 1.0))

        # Expose for Task
        self._last_asymmetric = False
        self._last_disadvantaged = 0

        # ── Build aircraft ──────────────────────────────────────────────
        jsbsim_data_dir = config.get("jsbsim_data_dir")

        # Choose controller type from config
        controller_type = config.get("controller_type", "pid")

        self.pursuers: List[_Pursuer] = []
        for _ in range(self.N):
            ac = Aircraft(jsbsim_data_dir)
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            # Build controller
            if controller_type == "neural":
                from src.dynamics.neural_controller import NeuralFlightController
                ctrl = SafetyInterceptor(NeuralFlightController())
            else:
                ctrl = SafetyInterceptor(PIDFlightController())
            self.pursuers.append(_Pursuer(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap, controller=ctrl))

        self.targets: List[_Target] = []
        for _ in range(self.M):
            ac = Aircraft(jsbsim_data_dir)
            fc = FlightController()
            envelope = FlightEnvelope(EnvelopeConfig())
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.targets.append(_Target(
                aircraft=ac, fc=fc, envelope=envelope, autopilot=ap))

        # ── Spaces (delegated to task) ──────────────────────────────────
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space
        self._step_counter = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  RLlib MultiAgentEnv interface
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        """Reset all aircraft and delegate to task.reset()."""
        rng = np.random.default_rng(seed)
        d = self._difficulty

        cluster = np.array([rng.uniform(-200, 200),
                           rng.uniform(-200, 200), 3000.0])

        # ── Asymmetric reset ────────────────────────────────────────────
        asymmetric = False
        disadvantaged_idx = 0
        if self.N >= 2 and rng.random() < 0.7:
            asymmetric = True
            disadvantaged_idx = rng.integers(0, self.N)
        self._last_asymmetric = asymmetric
        self._last_disadvantaged = disadvantaged_idx

        for i, ps in enumerate(self.pursuers):
            if asymmetric and i == disadvantaged_idx:
                behind_dir = rng.uniform(0, 2 * np.pi)
                far_offset = np.array([
                    1500.0 * np.cos(behind_dir),
                    1500.0 * np.sin(behind_dir), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster + far_offset
                away_hdg = float(np.degrees(np.arctan2(-far_offset[1], -far_offset[0]))) % 360.0
                away_hdg += rng.uniform(-60, 60)
                ps.ref_hdg = away_hdg % 360.0
            else:
                offset = np.array([rng.uniform(-100, 100),
                                  rng.uniform(-100, 100), 0.0])
                ps.aircraft.reset(
                    lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                    heading_deg=rng.uniform(0, 360), speed_kts=400, trim=False)
                ps.aircraft.position_ned = cluster + offset
                ps.ref_hdg = float(ps.aircraft.state["yaw_deg"])

            ps.fc.reset()
            ps.envelope.reset()
            ps.autopilot.reset(initial_speed_mps=200.0)
            ps.ref_alt_m = 3000.0
            ps.reset_state()

        # ── Target spawn ────────────────────────────────────────────────
        for j, ts in enumerate(self.targets):
            dist_min, dist_max = (900.0, 1300.0)  # task can override
            target_dist = rng.uniform(dist_min + d * 200, dist_max + d * 500)
            bearing_offset = rng.uniform(-d * 45.0, d * 45.0)
            heading_diff = rng.uniform(-d * 30.0, d * 30.0)

            pursuer_hdg = float(self.pursuers[0].aircraft.state["yaw_deg"])
            target_bearing = (pursuer_hdg + bearing_offset) % 360.0
            target_hdg = (pursuer_hdg + heading_diff) % 360.0

            target_ned = cluster + np.array([
                target_dist * np.cos(np.radians(target_bearing)),
                target_dist * np.sin(np.radians(target_bearing)), 0.0])
            target_ned[2] = 3000.0

            ts.aircraft.reset(
                lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000 * 3.28084),
                heading_deg=target_hdg, speed_kts=310, trim=False)
            ts.aircraft.position_ned = target_ned
            ts.fc.reset()
            ts.envelope.reset()
            ts.autopilot.reset(initial_speed_mps=160.0)
            ts.ref_hdg = target_hdg
            ts.ref_alt_m = 3000.0

        # ── Warmup: 3s level flight ─────────────────────────────────────
        warmup = int(3.0 * CTRL_FREQ)
        for _ in range(warmup):
            for ps in self.pursuers:
                s = ps.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ps.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(400))
                thr, elev, ail, rud = ps.fc.compute(s, tgt, PHYSICS_DT)
                ps.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += \
                    ps.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ps.aircraft.position_ned[2] = s["alt_m"]

            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ts.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, PHYSICS_DT)
                ts.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += \
                    ts.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ts.aircraft.position_ned[2] = s["alt_m"]

        self._step_counter = 0

        # Delegate task-specific reset
        self.task.reset(self)

        return self.task.get_obs(self), {}

    def step(self, action_dict: dict):
        """Execute one macro-action (0.2s, 12 physics sub-steps).

        1. Task interprets actions → PID targets
        2. Run 12 micro-step physics loop
        3. Task-level logic
        4. Collect obs / rewards / termination from Task
        """
        dt = PHYSICS_DT

        # ── ① Task applies actions: RL output → PID setpoints ────────────
        self.task.apply_actions(self, action_dict)

        # Per-step reward accumulators
        rewards = {aid: 0.0 for aid in self._agent_ids}

        # ── ② 12-step physics loop ──────────────────────────────────────
        for _ in range(DECISION_STEPS):
            # Control pursuers — delegate to controller (PID or Neural + Safety)
            for i, (ps, aid) in enumerate(zip(self.pursuers, self._agent_ids)):
                s = ps.aircraft.state
                cmd_speed = getattr(ps, '_cmd_speed', 250.0)
                target = FlightTarget(
                    heading_deg=ps.ref_hdg, altitude_m=ps.ref_alt_m,
                    speed_mps=cmd_speed)
                surfaces = ps.controller.predict(s, target, dt)
                ps.aircraft.set_controls(
                    throttle=surfaces.throttle, elevator=surfaces.elevator,
                    aileron=surfaces.aileron, rudder=surfaces.rudder)

            # Control targets — straight-and-level
            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(
                    heading_deg=ts.ref_hdg, altitude_m=3000.0,
                    speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, dt)
                ts.aircraft.set_controls(throttle=thr, elevator=elev,
                                        aileron=ail, rudder=rud)

            # Physics step
            for ps in self.pursuers:
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += \
                    ps.aircraft.velocity_ned[0:2] * dt
                ps.aircraft.position_ned[2] = ps.aircraft.state["alt_m"]
            for ts in self.targets:
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += \
                    ts.aircraft.velocity_ned[0:2] * dt
                ts.aircraft.position_ned[2] = ts.aircraft.state["alt_m"]

            # NaN guard
            for ps in self.pursuers:
                if any(not np.isfinite(float(ps.aircraft.state[k]))
                       for k in ["n_z_g", "airspeed_mps", "alt_m"]):
                    for aid in self._agent_ids:
                        rewards[aid] += -3000.0
                    obs = self.task.get_obs(self)
                    terminateds = {aid: True for aid in self._agent_ids}
                    terminateds["__all__"] = True
                    truncateds = {aid: False for aid in self._agent_ids}
                    truncateds["__all__"] = False
                    infos = {aid: {"termination_reason": "jsbsim_nan"} for aid in self._agent_ids}
                    return obs, rewards, terminateds, truncateds, infos

        # ── ③ Task-level logic ──────────────────────────────────────────
        self.task.step(self)

        # ── ④ ⑤ ⑥ Delegate to Task ──────────────────────────────────────
        obs = self.task.get_obs(self)
        task_rewards = self.task.get_reward(self)
        terminateds, truncateds, infos = self.task.get_termination(self)

        # Merge rewards
        for aid in self._agent_ids:
            rewards[aid] += task_rewards.get(aid, 0.0)

        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        """Clean up JSBSim instances."""
        for ps in self.pursuers:
            try:
                ps.aircraft.close()
            except AttributeError:
                pass
        for ts in self.targets:
            try:
                ts.aircraft.close()
            except AttributeError:
                pass

    def render(self, mode: str = "txt", filepath: str = "./recording.txt.acmi",
               tacview=None) -> None:
        """Render current frame to ACMI file or Tacview real-time."""
        # minimal stub — full render needs tacview_exporter
        pass
