"""Formation MAPPO environment — RLlib MultiAgentEnv for 2v1 CTDE.

Each pursuer is an independent RL agent with its own Box(2) action
and local observation.  The target is scripted (straight-and-level).
Global state is provided for the centralized critic.

Agent IDs: "pursuer_0", "pursuer_1"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.dynamics.aircraft import Aircraft
from src.dynamics.autopilot import BFMAutopilot, BFMAutopilotConfig, TrimSchedule, GainScheduler
from src.dynamics.flight_envelope import FlightEnvelope, EnvelopeConfig
from src.dynamics.flight_controller import FlightController, FlightControlTargets
from src.utils.units import kts_to_mps
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles

# ── Constants (shared with FormationEnv) ─────────────────────────────────
CTRL_FREQ = 60.0; PHYSICS_DT = 1.0 / CTRL_FREQ
DECISION_DT = 0.5; DECISION_STEPS = int(DECISION_DT * CTRL_FREQ)
MAX_EPISODE_TIME = 120.0
MAX_DIST = 10000.0; MAX_HEIGHT = 5000.0; MAX_VEL = 400.0
MAX_ANG_VEL = np.pi; MAX_AOA = 30.0; MAX_LOS_RATE = 0.5
REWARD_SUCCESS = 5000.0; REWARD_TIMEOUT = -500.0
REWARD_CRASH = -200.0; REWARD_LOST_TARGET = -200.0
STEP_PENALTY = 0.25
ANTI_STALL_WINDOW = 35; ANTI_STALL_MIN_VC = 15.0
ANTI_STALL_MIN_DIST = 200.0; ANTI_STALL_PENALTY = 200.0
OBS_PER_PURSUER = 33
GLOBAL_DIM_PER_AIRCRAFT = 7  # pos(3) + vel(3) + heading(1)


class FormationMAPPOEnv(MultiAgentEnv):
    """2v1 formation pursuit for RLlib MAPPO (CTDE).

    Agents: "pursuer_0", "pursuer_1"
    Action per agent: Box(2) [turn_rate_factor, speed_factor]
    Obs per agent: Dict {"obs": Box(33), "global_state": Box(global_dim)}
    """

    metadata = {"name": "formation_mappo_v0"}

    def __init__(self, env_config: dict | None = None):
        super().__init__()
        config = env_config or {}
        self._difficulty = float(np.clip(config.get("difficulty_level", 0.0), 0.0, 1.0))
        self._lock_altitude = config.get("lock_altitude", True)
        self._record_tacview = config.get("record_tacview", False)
        self._ref_lla = (30.0, 120.0, 3000.0)
        self._tacview_frames: List[dict] = []

        self.N = 2  # pursuers
        self.M = 1  # targets
        self._agent_ids = ["pursuer_0", "pursuer_1"]

        # ── Build aircraft ────────────────────────────────────────────
        self.pursuers: List[_Pursuer] = []
        for i in range(self.N):
            ac = Aircraft(config.get("jsbsim_data_dir"))
            fc = FlightController()
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.pursuers.append(_Pursuer(aircraft=ac, fc=fc, autopilot=ap))

        self.targets: List[_Target] = []
        for i in range(self.M):
            ac = Aircraft(config.get("jsbsim_data_dir"))
            fc = FlightController()
            ap = BFMAutopilot(BFMAutopilotConfig(), trim=TrimSchedule(), scheduler=GainScheduler())
            self.targets.append(_Target(aircraft=ac, fc=fc, autopilot=ap))

        # ── Spaces ────────────────────────────────────────────────────
        self._obs_per_pursuer = OBS_PER_PURSUER
        self._global_dim = (self.N + self.M) * GLOBAL_DIM_PER_AIRCRAFT

        single_obs = gym.spaces.Dict({
            "obs": gym.spaces.Box(-1, 1, (self._obs_per_pursuer,), dtype=np.float32),
            "global_state": gym.spaces.Box(-1, 1, (self._global_dim,), dtype=np.float32),
        })
        single_act = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            aid: single_obs for aid in self._agent_ids
        })
        self.action_space = gym.spaces.Dict({
            aid: single_act for aid in self._agent_ids
        })
        self._agent_ids = list(self._agent_ids)  # ensure list for RLlib

        self._step_counter = 0

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        d = self._difficulty
        cluster = np.array([rng.uniform(-200, 200), rng.uniform(-200, 200), 3000.0])

        for i, ps in enumerate(self.pursuers):
            off = np.array([rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0])
            ps.aircraft.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000*3.28084),
                              heading_deg=rng.uniform(0,360), speed_kts=400, trim=False)
            ps.aircraft.position_ned = cluster + off
            ps.fc.reset(); ps.autopilot.reset(initial_speed_mps=200.0)
            ps.ref_hdg = float(ps.aircraft.state["yaw_deg"]); ps.ref_alt_m = 3000.0
            ps.prev_rpy = ps.aircraft.rpy_rad.copy()
            ps.prev_airspeed = 180.0; ps.closure_rates.clear()
            ps.prev_ata_deg = None; ps.proximity_awarded.clear()
            ps.loiter_time = 0.0; ps.zone_death_counter = 0

        for j, ts in enumerate(self.targets):
            dist = rng.uniform(900+d*1100, 1300+d*1700)
            bearing_off = rng.uniform(-d*45.0, d*45.0)
            hdg_diff = rng.uniform(-d*30.0, d*30.0)
            pursuer_hdg = float(self.pursuers[0].aircraft.state["yaw_deg"])
            tgt_bearing = (pursuer_hdg + bearing_off) % 360.0
            tgt_hdg = (pursuer_hdg + hdg_diff) % 360.0
            tgt_ned = cluster + np.array([dist*np.cos(np.radians(tgt_bearing)),
                                           dist*np.sin(np.radians(tgt_bearing)), 0.0])
            tgt_ned[2] = 3000.0
            ts.aircraft.reset(lat_deg=30.0, lon_deg=120.0, alt_ft=int(3000*3.28084),
                              heading_deg=tgt_hdg, speed_kts=310, trim=False)
            ts.aircraft.position_ned = tgt_ned
            ts.fc.reset(); ts.autopilot.reset(initial_speed_mps=160.0)
            ts.ref_hdg = tgt_hdg; ts.ref_alt_m = 3000.0

        # Warmup: 3s level flight
        warmup = int(3.0 * CTRL_FREQ)
        for _ in range(warmup):
            for ps in self.pursuers:
                s = ps.aircraft.state
                tgt = FlightControlTargets(heading_deg=ps.ref_hdg, altitude_m=3000.0, speed_mps=kts_to_mps(400))
                thr, elev, ail, rud = ps.fc.compute(s, tgt, PHYSICS_DT)
                ps.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ps.aircraft.run()
                ps.aircraft.position_ned[0:2] += ps.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ps.aircraft.position_ned[2] = s["alt_m"]
            for ts in self.targets:
                s = ts.aircraft.state
                tgt = FlightControlTargets(heading_deg=ts.ref_hdg, altitude_m=3000.0, speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, tgt, PHYSICS_DT)
                ts.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
                ts.aircraft.run()
                ts.aircraft.position_ned[0:2] += ts.aircraft.velocity_ned[0:2] * PHYSICS_DT
                ts.aircraft.position_ned[2] = s["alt_m"]

        for ps in self.pursuers:
            ps.prev_dist = float(np.linalg.norm(ps.aircraft.position_ned - self.targets[0].aircraft.position_ned))
            ps.episode_start_dist = ps.prev_dist

        self._step_counter = 0; self._tacview_frames = []
        return self._get_obs(), {}

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action_dict: dict):
        dt = PHYSICS_DT
        actions = {}
        for aid in self._agent_ids:
            a = np.clip(np.asarray(action_dict.get(aid, [0,0]), dtype=np.float32), -1, 1)
            actions[aid] = {'turn': float(a[0]), 'speed': float(a[1]),
                            'cmd_turn_rate': float(a[0]*15.0),
                            'cmd_speed': float(250.0 + a[1]*100.0)}

        terminated = False; truncated = False
        reason = "timeout"; kill_aid = None
        rewards = {aid: 0.0 for aid in self._agent_ids}
        term_agents = set()
        initial_dists = {i: ps.prev_dist for i, ps in enumerate(self.pursuers)}

        for _ in range(DECISION_STEPS):
            # Control pursuers
            for i, ps in enumerate(self.pursuers):
                aid = self._agent_ids[i]
                ac = actions[aid]; s = ps.aircraft.state
                ps.ref_hdg = (ps.ref_hdg + ac['cmd_turn_rate'] * dt) % 360.0
                fc_tgt = FlightControlTargets(heading_deg=ps.ref_hdg, altitude_m=ps.ref_alt_m, speed_mps=ac['cmd_speed'])
                thr, elev, ail, rud = ps.fc.compute(s, fc_tgt, dt)
                ps.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)

            # Control targets
            for ts in self.targets:
                s = ts.aircraft.state
                fc_tgt = FlightControlTargets(heading_deg=ts.ref_hdg, altitude_m=3000.0, speed_mps=kts_to_mps(310))
                thr, elev, ail, rud = ts.fc.compute(s, fc_tgt, dt)
                ts.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)

            # Physics
            for ps in self.pursuers:
                ps.aircraft.run(); ps.aircraft.position_ned[0:2] += ps.aircraft.velocity_ned[0:2] * dt
                ps.aircraft.position_ned[2] = ps.aircraft.state["alt_m"]
            for ts in self.targets:
                ts.aircraft.run(); ts.aircraft.position_ned[0:2] += ts.aircraft.velocity_ned[0:2] * dt
                ts.aircraft.position_ned[2] = ts.aircraft.state["alt_m"]
            self._step_counter += 1

            # NaN guard
            for ps in self.pursuers:
                if any(not np.isfinite(float(ps.aircraft.state[k])) for k in ["n_z_g","airspeed_mps","alt_m"]):
                    terminated = True; reason = "jsbsim_nan"; break
            if terminated: break

            # Per-pursuer reward + checks
            t_pos = self.targets[0].aircraft.position_ned
            any_success = False
            for i, ps in enumerate(self.pursuers):
                aid = self._agent_ids[i]; a_pos = ps.aircraft.position_ned
                cur_dist = float(np.linalg.norm(a_pos - t_pos))
                delta = ps.prev_dist - cur_dist

                # Success
                if cur_dist < 200.0 and ps.episode_start_dist > 400.0:
                    any_success = True; kill_aid = aid

                # Progress
                rewards[aid] += 1.5 * delta * 0.5
                if cur_dist < 500.0: rewards[aid] += 1.5 * delta * 5.0

                # ATA (distance-gated)
                a_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
                t_fwd = compute_forward_vector(self.targets[0].aircraft.rpy_rad)
                _, los_dir, _ = compute_los(a_pos, t_pos)
                geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
                dist_factor = max(0.0, 1.0 - cur_dist / 3000.0)
                rewards[aid] += 8.0 * max(geo["cos_ata"], -0.2) * dt * dist_factor

                # Terminal pull (ATA-gated)
                if 200.0 <= cur_dist <= 500.0:
                    ata_gate = max(0.0, float(geo["cos_ata"])**3)
                    rewards[aid] += (500.0 - cur_dist) / 300.0 * 50.0 * dt * ata_gate

                # ATA degradation penalty
                if cur_dist < 1000.0:
                    ata_deg = float(np.degrees(np.arccos(np.clip(geo["cos_ata"], -1, 1))))
                    if ata_deg > 20.0: rewards[aid] -= 1.0 * dt

                # Low-speed warning
                spd = float(ps.aircraft.state["airspeed_mps"])
                if spd < 130.0: rewards[aid] -= (130.0-spd)/130.0 * dt

                # Baseline bleed
                rewards[aid] -= 1.0 * dt

                # Proximity milestones
                for thresh, bonus in [(800,25),(500,50),(300,100)]:
                    if cur_dist < thresh and thresh not in ps.proximity_awarded:
                        rewards[aid] += bonus; ps.proximity_awarded.add(thresh)

                # Lost target
                if cur_dist > 10000.0:
                    rewards[aid] += REWARD_LOST_TARGET
                    terminated = True; reason = "lost_target"; break

                # Ground/crash
                alt = a_pos[2]
                if alt < 10.0: rewards[aid] += REWARD_CRASH; terminated = True; reason = "ground_crash"; break
                if alt > 12000.0: terminated = True; reason = "out_of_bounds"; break

                ps.prev_dist = cur_dist

            if terminated: break

            # Success
            if any_success:
                for aid in self._agent_ids:
                    rewards[aid] += REWARD_SUCCESS
                terminated = True; reason = "success"; break

            # Formation collision
            if self.N >= 2:
                p0 = self.pursuers[0].aircraft.position_ned
                p1 = self.pursuers[1].aircraft.position_ned
                if float(np.linalg.norm(p0 - p1)) < 50.0:
                    rewards["pursuer_0"] -= 3000; rewards["pursuer_1"] -= 3000
                    terminated = True; reason = "formation_collision"; break

        # Timeout
        if not terminated and not truncated:
            if self._step_counter / CTRL_FREQ >= MAX_EPISODE_TIME:
                truncated = True; reason = "timeout"
                for aid in self._agent_ids: rewards[aid] += REWARD_TIMEOUT

        # RLlib format: __all__ key for shared termination
        obs = self._get_obs()
        term = {aid: terminated or truncated for aid in self._agent_ids}
        term["__all__"] = terminated or truncated
        info = {"reason": reason, "kill_agent": kill_aid}

        return obs, rewards, term, term, info

    # ── Observation ──────────────────────────────────────────────────────

    def _get_obs(self):
        obs = {}
        target_pos = self.targets[0].aircraft.position_ned
        target_vel = self.targets[0].aircraft.velocity_ned

        # Global state: all aircraft pos(3) + vel(3) + heading(1)
        global_vec = []
        for ps in self.pursuers:
            p = ps.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ps.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ps.aircraft.state["yaw_deg"]) / 180.0])
            global_vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        for ts in self.targets:
            p = ts.aircraft.position_ned / np.array([MAX_DIST, MAX_DIST, MAX_HEIGHT])
            v = ts.aircraft.velocity_ned / MAX_VEL
            h = np.array([float(ts.aircraft.state["yaw_deg"]) / 180.0])
            global_vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
        global_state = np.array(global_vec, dtype=np.float32)

        for i, ps in enumerate(self.pursuers):
            aid = self._agent_ids[i]
            local = self._build_local_obs(i, ps, target_pos, target_vel)
            obs[aid] = {"obs": local.astype(np.float32), "global_state": global_state}

        return obs

    def _build_local_obs(self, idx, ps, target_pos, target_vel):
        """33-dim per-pursuer observation (same as FormationEnv)."""
        a_pos = ps.aircraft.position_ned; a_rpy = ps.aircraft.rpy_rad
        a_vel = ps.aircraft.velocity_ned
        rel_w = target_pos - a_pos
        ch, sh = np.cos(a_rpy[2]), np.sin(a_rpy[2])
        rel_body = np.array([rel_w[0]*ch+rel_w[1]*sh, -rel_w[0]*sh+rel_w[1]*ch, -rel_w[2]])
        vel_body = np.array([a_vel[0]*ch+a_vel[1]*sh, -a_vel[0]*sh+a_vel[1]*ch, a_vel[2]])
        t_vel_body = np.array([target_vel[0]*ch+target_vel[1]*sh, -target_vel[0]*sh+target_vel[1]*ch, target_vel[2]])
        ang_vel = self._ang_vel(ps.aircraft.rpy_rad, ps.prev_rpy)
        ps.prev_rpy = ps.aircraft.rpy_rad.copy()
        a_fwd = compute_forward_vector(a_rpy)
        t_fwd = compute_forward_vector(self.targets[0].aircraft.rpy_rad)
        _, los_dir, _ = compute_los(a_pos, target_pos)
        geo = compute_tactical_angles(a_fwd, t_fwd, los_dir)
        spd = float(ps.aircraft.state["airspeed_mps"]); alpha = float(ps.aircraft.state["alpha_deg"])
        r_h = target_pos[:2] - a_pos[:2]; dh = float(np.linalg.norm(r_h))
        lambda_dot = float(np.cross(r_h, target_vel[:2]-a_vel[:2]))/(dh*dh) if dh>1 else 0.0
        bearing = float(np.degrees(np.arctan2(r_h[1], r_h[0])))%360.0
        hdg = float(ps.aircraft.state["yaw_deg"])%360.0
        berr = (bearing-hdg+180)%360-180

        base = np.array([
            rel_body[0]/MAX_DIST, rel_body[1]/MAX_DIST, rel_body[2]/MAX_DIST,
            vel_body[0]/MAX_VEL, vel_body[1]/MAX_VEL, vel_body[2]/MAX_VEL,
            a_rpy[0]/np.pi, a_rpy[1]/(np.pi/2), a_rpy[2]/np.pi,
            ang_vel[0]/MAX_ANG_VEL, ang_vel[1]/MAX_ANG_VEL, ang_vel[2]/MAX_ANG_VEL,
            a_pos[2]/MAX_HEIGHT,
            t_vel_body[0]/MAX_VEL, t_vel_body[1]/MAX_VEL, t_vel_body[2]/MAX_VEL,
            0.0, 0.0, 0.0,
            geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
            alpha/MAX_AOA, spd/MAX_VEL, 0.0,
            np.clip(lambda_dot/MAX_LOS_RATE,-1,1), np.clip(berr/180,-1,1),
        ], dtype=np.float32)

        # Mate obs
        if self.N >= 2:
            mate_idx = 1 if idx == 0 else 0
            mp = self.pursuers[mate_idx].aircraft.position_ned
            mv = self.pursuers[mate_idx].aircraft.velocity_ned
            mrw = mp - a_pos; mrv = mv - a_vel
            mate_body_pos = np.array([mrw[0]*ch+mrw[1]*sh, -mrw[0]*sh+mrw[1]*ch, -mrw[2]])
            mate_body_vel = np.array([mrv[0]*ch+mrv[1]*sh, -mrv[0]*sh+mrv[1]*ch, mrv[2]])
            mate = np.array([mate_body_pos[0]/MAX_DIST, mate_body_pos[1]/MAX_DIST,
                             mate_body_pos[2]/MAX_DIST, mate_body_vel[0]/MAX_VEL,
                             mate_body_vel[1]/MAX_VEL, mate_body_vel[2]/MAX_VEL], dtype=np.float32)
        else:
            mate = np.zeros(6, dtype=np.float32)

        return np.clip(np.concatenate([base, mate]), -1, 1)

    def _ang_vel(self, cur, prev):
        d = cur - prev; d = (d + np.pi) % (2*np.pi) - np.pi; return d / PHYSICS_DT


# ── Internal state dataclasses ────────────────────────────────────────────

@dataclass
class _Pursuer:
    aircraft: Aircraft; fc: FlightController; autopilot: BFMAutopilot
    ref_hdg: float = 0.0; ref_alt_m: float = 3000.0
    prev_dist: float = 0.0; prev_ata_deg: Optional[float] = None
    prev_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_airspeed: float = 180.0
    proximity_awarded: set = field(default_factory=set)
    closure_rates: deque = field(default_factory=lambda: deque(maxlen=ANTI_STALL_WINDOW))
    zone_death_counter: int = 0; loiter_time: float = 0.0
    episode_start_dist: float = 0.0

@dataclass
class _Target:
    aircraft: Aircraft; fc: FlightController; autopilot: BFMAutopilot
    ref_hdg: float = 0.0; ref_alt_m: float = 3000.0
