"""LowLevelControlTask — LAG-style non-hierarchical control surface training.

Trains RL policy to directly output aileron/elevator/rudder/throttle indices.
This is the equivalent of LAG's BaselineActor training — the policy learns to
control F-16 from scratch on OUR JSBSim 1.3.1 dynamics.

Observation: 12-dim body-frame state, precisely normalized to [-1,1]
  [heading_err, pitch_err, beta_err, speed_err,
   p, q, r, roll, pitch, alpha, beta, airspeed]

Action: MultiDiscrete([21,21,21,15]) → direct control surfaces
  aileron:  idx 0..20 → (idx-10)/10 ∈ [-1, 1], center=0
  elevator: idx 0..20 → (idx-10)/10 ∈ [-1, 1], center=0
  rudder:   idx 0..20 → (idx-10)/10 ∈ [-1, 1], center=0
  throttle: idx 0..14 → idx/14 ∈ [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .task_base import BaseTask
from src.dynamics.controller_base import ControlSurfaces


class HeadingTrackingTask(BaseTask):
    """LAG-style low-level control — direct surface output from RL policy.

    Agent ID: "p0"
    Trains the equivalent of LAG's BaselineActor on our own JSBSim.
    """

    # ── Action resolution ───────────────────────────────────────────────
    N_AILERON  = 21   # [-1.0, +1.0], center=10 → 0.0
    N_ELEVATOR = 21   # [-1.0, +1.0], center=10 → 0.0
    N_RUDDER   = 21   # [-1.0, +1.0], center=10 → 0.0
    N_THROTTLE = 15   # [0.0, 1.0], step ~0.071

    # ── Normalization constants ─────────────────────────────────────────
    MAX_ROLL_RAD   = np.pi        # φ
    MAX_PITCH_RAD  = np.pi / 2    # θ
    MAX_BETA_RAD   = np.pi / 6    # β (sideslip)
    MAX_ALPHA_DEG  = 30.0         # α (AoA)
    MAX_ANG_VEL_RPS = np.pi       # p, q, r
    MAX_SPEED_MPS  = 400.0
    MAX_SPEED_ERR  = 100.0

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._agent_ids = ["p0"]
        self.N = 1; self.M = 0
        self.target_heading_deg = config.get("target_heading", 90.0) if config else 90.0
        self.target_altitude_m = 3000.0
        self.target_speed_mps = 250.0

        single_obs = gym.spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)
        single_act = gym.spaces.MultiDiscrete(
            [self.N_AILERON, self.N_ELEVATOR, self.N_RUDDER, self.N_THROTTLE])

        self._observation_space = gym.spaces.Dict({"p0": single_obs})
        self._action_space = gym.spaces.Dict({"p0": single_act})

    @property
    def observation_space(self): return self._observation_space
    @property
    def action_space(self): return self._action_space

    # ── Lifecycle ───────────────────────────────────────────────────────

    def reset(self, env) -> None:
        # Randomize targets per episode — use global seed for reproducibility
        self.target_heading_deg = float(np.random.uniform(0, 360))
        # Vary target altitude in 2500-3500m range → agent learns climb/descend
        self.target_altitude_m = float(np.random.uniform(2500, 3500))
        self.target_speed_mps = float(np.random.uniform(200, 300))

        for ps in env.pursuers:
            s = ps.aircraft.state
            # Sync PID refs to current state (not to target — that's the RL's job)
            ps.ref_hdg = float(s["yaw_deg"])
            ps.ref_alt_m = float(s["alt_m"])
            ps._cmd_speed = float(s["airspeed_mps"])
            ps._direct_surfaces = None

    def apply_actions(self, env, action_dict: Dict[str, np.ndarray]) -> None:
        """Direct control surface mapping — bypasses PID/Neural entirely."""
        for i, aid in enumerate(self._agent_ids):
            a = np.asarray(action_dict.get(aid, np.array([10, 10, 10, 7])), dtype=np.int64)
            ail_idx  = int(np.clip(a[0], 0, self.N_AILERON  - 1))
            elev_idx = int(np.clip(a[1], 0, self.N_ELEVATOR - 1))
            rud_idx  = int(np.clip(a[2], 0, self.N_RUDDER   - 1))
            thr_idx  = int(np.clip(a[3], 0, self.N_THROTTLE - 1))

            env.pursuers[i]._direct_surfaces = ControlSurfaces(
                aileron=(ail_idx  - 10) / 10.0,     # 10 → 0.0 center
                elevator=(elev_idx - 10) / 10.0,    # 10 → 0.0 center
                rudder=(rud_idx   - 10) / 10.0,     # 10 → 0.0 center
                throttle=thr_idx / 14.0,            # 0→0.0, 14→1.0
            )

    def step(self, env) -> None: pass

    # ── 12-dim LAG-aligned observation ──────────────────────────────────

    def get_obs(self, env) -> Dict[str, dict]:
        ps = env.pursuers[0]
        s = ps.aircraft.state

        # Fundamental flight states
        roll_rad  = math.radians(float(s["roll_deg"]))
        pitch_rad = math.radians(float(s["pitch_deg"]))
        yaw_rad   = math.radians(float(s["yaw_deg"]))
        airspeed  = float(s["airspeed_mps"])
        alpha_deg = float(s["alpha_deg"])

        # Sideslip (approximate from beta_deg or estimate from v_fps)
        beta_deg = float(s.get("beta_deg", 0.0))

        # Body angular rates — with JSBSim FDM fallback if state dict is missing keys
        fdm = ps.aircraft.fdm
        p_rps = float(s.get("p_rps",
                   fdm.get_property_value("velocities/p-rad_sec")))
        q_rps = float(s.get("q_rps",
                   fdm.get_property_value("velocities/q-rad_sec")))
        r_rps = float(s.get("r_rps",
                   fdm.get_property_value("velocities/r-rad_sec")))

        # Error signals
        heading_err_rad = ((self.target_heading_deg - float(s["yaw_deg"]) + 180.0) % 360.0 - 180.0) * np.pi / 180.0
        pitch_err_rad  = 0.0 - pitch_rad  # target: level flight (pitch=0)
        beta_err_rad   = 0.0 - math.radians(beta_deg)  # target: zero sideslip
        speed_err_mps  = self.target_speed_mps - airspeed

        obs = np.array([
            heading_err_rad / np.pi,
            pitch_err_rad / (np.pi / 2),
            beta_err_rad / (np.pi / 6),
            speed_err_mps / self.MAX_SPEED_ERR,
            p_rps / self.MAX_ANG_VEL_RPS,
            q_rps / self.MAX_ANG_VEL_RPS,
            r_rps / self.MAX_ANG_VEL_RPS,
            roll_rad / np.pi,
            pitch_rad / (np.pi / 2),
            alpha_deg / self.MAX_ALPHA_DEG,
            math.radians(beta_deg) / (np.pi / 6),
            airspeed / self.MAX_SPEED_MPS,
        ], dtype=np.float32)

        return {"p0": np.clip(obs, -1, 1)}

    # ── LAG-style geometric mean reward ─────────────────────────────────

    def get_reward(self, env) -> Dict[str, float]:
        ps = env.pursuers[0]
        s = ps.aircraft.state

        heading_err = abs((self.target_heading_deg - float(s["yaw_deg"]) + 180.0) % 360.0 - 180.0)
        pitch_deg   = float(s["pitch_deg"])
        beta_deg    = float(s.get("beta_deg", 0.0))
        airspeed    = float(s["airspeed_mps"])
        alt_m       = float(s["alt_m"])

        # ── Gaussian-shaped rewards ─────────────────────────────────────
        # IMPORTANT: we reward sideslip (β≈0 for coordinated turn), NOT roll.
        # Fixed-wing aircraft MUST bank to turn — penalizing |roll| would
        # teach the agent to never turn, which is physically wrong.
        head_r = math.exp(-(heading_err / 5.0) ** 2)
        alt_r  = math.exp(-((alt_m - self.target_altitude_m) / 15.24) ** 2)
        beta_r = math.exp(-(abs(beta_deg) / 5.0) ** 2)     # sideslip ≈ 0 in coordinated turn
        spd_r  = math.exp(-((airspeed - self.target_speed_mps) / 24.0) ** 2)

        # Geometric mean
        reward = (head_r * alt_r * beta_r * spd_r) ** 0.25

        # ── Bonuses / penalties ─────────────────────────────────────────
        if heading_err < 3.0:
            reward += 0.5
        if abs(pitch_deg) > 45.0:
            reward -= 1.0
        # Penalize aggressive pitch oscillations (q_rps jitter)
        q_mag = abs(float(s.get("q_rps", 0.0)))
        if q_mag > 0.5:
            reward -= 0.1 * (q_mag - 0.5)

        return {"p0": float(np.clip(reward, -5, 5))}

    # ── Termination ─────────────────────────────────────────────────────

    def get_termination(self, env) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        ps = env.pursuers[0]; s = ps.aircraft.state
        alt_m = float(s["alt_m"]); pitch_deg = float(s["pitch_deg"])
        terminateds = {"p0": False, "__all__": False}
        truncateds = {"p0": False, "__all__": False}
        infos: Dict[str, Any] = {"p0": {}}

        if alt_m < 500.0 or abs(pitch_deg) > 85.0:
            terminateds = {"p0": True, "__all__": True}
            infos["p0"]["termination_reason"] = "crash"
        if env._step_counter >= 500:
            truncateds = {"p0": True, "__all__": True}
            infos["p0"]["termination_reason"] = "timeout"
        return terminateds, truncateds, infos
