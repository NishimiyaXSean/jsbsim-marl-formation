"""1v1 air combat Gymnasium environment backed by JSBSim F-16 flight dynamics.

Supports:
- Continuous action space: [throttle, elevator, aileron, rudder] ∈ [-1, 1]^4
- Multi-agent RLlib interface (MultiAgentEnv)
- Curriculum learning (3 stages)
- Tacview ACMI export
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.dynamics.aircraft import Aircraft
from src.environment.observations import compute_obs, compute_global_state
from src.environment.rewards import (
    RewardConfig,
    reward_progress,
    reward_time_pressure,
    reward_z_advantage,
    reward_energy_loss,
    reward_ground_warning,
    reward_tracking,
    reward_closing_speed,
    reward_survival,
    reward_escape,
    reward_spoofing,
)
from src.environment.termination import (
    check_collision,
    check_cpa,
    check_ground_crash,
    check_out_of_bounds,
    check_timeout,
)
from src.environment.scenario import generate_spawn
from src.environment.curriculum import get_curriculum, CURRICULUM_STAGES
from src.utils.geometry import compute_forward_vector, compute_los, compute_tactical_angles, compute_closing_speed
from src.utils.units import m_to_ft, mps_to_kts, deg_to_rad


class AirCombatEnv(MultiAgentEnv):
    """1v1 air combat environment with JSBSim F-16 dynamics.

    Action space (4-dim continuous, per agent):
        [throttle, elevator, aileron, rudder]  all ∈ [-1, 1]

    Observation space (19-dim, per agent):
        See `observations.compute_obs` for feature breakdown.
    """

    metadata = {"render_modes": ["human", "tacview"], "name": "air_combat_1v1_v0"}

    def __init__(
        self,
        gui: bool = False,
        record_tacview: bool = False,
        reward_config: Optional[RewardConfig] = None,
        jsbsim_data_dir: Optional[str] = None,
    ):
        super().__init__()

        # Agent identities
        self.possible_agents = ["attacker_0", "evader_0"]
        self.agents = self.possible_agents[:]

        # Aircraft
        self.attacker = Aircraft(jsbsim_data_dir)
        self.evader = Aircraft(jsbsim_data_dir)
        self._aircraft_map = {"attacker_0": self.attacker, "evader_0": self.evader}

        # Simulation params
        self.CTRL_FREQ = 60.0       # control frequency (Hz)
        self.PHYSICS_DT = 1.0 / 60.0
        self.DECISION_DT = 0.5      # AI decides every 0.5s (30 physics steps per decision)
        self.EPISODE_LEN_SEC = 240.0

        # Combat params
        self.CPA_RADIUS = 300.0     # proximity fuze trigger (m)
        self.MAX_SPEED = 400.0      # for normalization (m/s)
        self.MAX_G = 9.0
        self.MIN_G = -3.0

        # Bounds
        self.ATTACKER_DEATH_FLOOR = 10.0     # altitude (m)
        self.EVADER_DEATH_FLOOR = 595.0
        self.CEILING = 4900.0

        # Action space: 4-dim continuous
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Observation space: Dict for MAPPO CTDE
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32),
                "global_state": gym.spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32),
            })
            for agent in self.possible_agents
        }

        # Reward config
        self.reward_cfg = reward_config or RewardConfig()

        # Curriculum
        self.curriculum_stage = 1
        self._update_curriculum_bounds()

        # Tacview
        self.record_tacview = record_tacview
        self._tacview_frames: List[dict] = []  # accumulated frames for export

        # Internal state
        self.step_counter = 0
        self.macro_step = 0
        self.prev_dist = 0.0
        self.last_cos_ata_attacker = 0.0
        self._ref_lla = (30.0, 120.0, 2500.0)

    # ── Curriculum ──────────────────────────────────────────────────────────

    def set_curriculum_stage(self, stage: int) -> None:
        self.curriculum_stage = stage
        self._update_curriculum_bounds()

    def _update_curriculum_bounds(self) -> None:
        cfg = get_curriculum(self.curriculum_stage)
        self.evader_speed_coeff = cfg.evader_speed_coeff
        self.evader_g_coeff = cfg.evader_g_coeff
        self.warning_radius = cfg.warning_radius

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.agents = self.possible_agents[:]
        rng = np.random.default_rng(seed)

        # Generate spawn configuration
        spawn = generate_spawn(self.curriculum_stage, rng)
        self._ref_lla = spawn["ref_lla"]

        # Reset attacker
        a = spawn["attacker"]
        self.attacker.reset(
            lat_deg=a["lat_deg"], lon_deg=a["lon_deg"],
            alt_ft=a["alt_ft"], heading_deg=a["heading_deg"],
            speed_kts=a["speed_kts"],
            trim=False,  # skip trim for RL — JSBSim FCS stabilizes quickly
        )

        # Reset evader (with curriculum speed limit)
        e = spawn["evader"]
        e_speed = e["speed_kts"] * self.evader_speed_coeff
        self.evader.reset(
            lat_deg=e["lat_deg"], lon_deg=e["lon_deg"],
            alt_ft=e["alt_ft"], heading_deg=e["heading_deg"],
            speed_kts=e_speed,
            trim=False,
        )

        # Store NED positions for state computation
        self.attacker.position_ned = spawn["attacker"]["ned"]
        self.evader.position_ned = spawn["evader"]["ned"]

        # Initialize counters
        self.step_counter = 0
        self.macro_step = 0

        # Initial distance
        a_pos = self.attacker.position_ned
        e_pos = self.evader.position_ned
        self.prev_dist = float(np.linalg.norm(a_pos - e_pos))

        # Initial tactical angle for trend tracking
        a_forward = compute_forward_vector(self.attacker.rpy_rad)
        _, los_dir, _ = compute_los(a_pos, e_pos)
        self.last_cos_ata_attacker = float(np.clip(np.dot(a_forward, los_dir), -1.0, 1.0))

        # Tacview reset
        if self.record_tacview:
            self._tacview_frames = []
            self._record_tacview_frame(0.0)

        # Build observations
        obs = {}
        for agent_id in self.agents:
            obs[agent_id] = self._get_obs_dict(agent_id)

        return obs, {a: {} for a in self.agents}

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, actions: dict):
        if not actions:
            self.agents = []
            return {}, {}, {"__all__": True}, {"__all__": True}, {}

        decision_steps = int(self.DECISION_DT * self.CTRL_FREQ)
        dt = self.PHYSICS_DT

        total_rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.agents}

        self.macro_step += 1

        for _ in range(decision_steps):
            # Apply controls
            for agent_id in self.agents:
                ac = self._aircraft_map[agent_id]
                act = actions.get(agent_id, np.zeros(4))
                ac.set_controls(
                    throttle=float(act[0]),
                    elevator=float(act[1]),
                    aileron=float(act[2]),
                    rudder=float(act[3]),
                )

            # Step both aircraft
            self.attacker.run()
            self.evader.run()
            self.step_counter += 1

            # Update NED positions (approximate: accumulate velocity * dt)
            # In a full implementation, use lla_to_ned conversion
            self.attacker.position_ned = self.attacker.position_ned + self.attacker.velocity_ned * dt
            self.evader.position_ned = self.evader.position_ned + self.evader.velocity_ned * dt

            a_pos = self.attacker.position_ned
            e_pos = self.evader.position_ned
            a_vel = self.attacker.velocity_ned
            e_vel = self.evader.velocity_ned
            a_rpy = self.attacker.rpy_rad
            e_rpy = self.evader.rpy_rad

            # Distance
            current_dist = float(np.linalg.norm(a_pos - e_pos))
            raw_delta = current_dist - self.prev_dist
            micro_delta_dist = np.clip(raw_delta, -20.0, 20.0)

            # Tactical geometry
            a_forward = compute_forward_vector(a_rpy)
            e_forward = compute_forward_vector(e_rpy)
            _, los_dir, _ = compute_los(a_pos, e_pos)
            geo = compute_tactical_angles(a_forward, e_forward, los_dir)

            # ── Attacker rewards ─────────────────────────────────────────
            if "attacker_0" in actions and not terminations["attacker_0"]:
                dz = a_pos[2] - e_pos[2]
                vel_norm = float(np.linalg.norm(a_vel))
                sideslip = abs(a_vel[1])
                rel_vel = a_vel - e_vel
                rel_vel_dir = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
                cos_collision = np.clip(np.dot(rel_vel_dir, los_dir), -1.0, 1.0)
                closing_speed = compute_closing_speed(a_vel, los_dir)

                total_rewards["attacker_0"] += (
                    reward_progress(micro_delta_dist, dt, self.reward_cfg)
                    + reward_time_pressure(self.step_counter, self.CTRL_FREQ, dt, self.reward_cfg)
                    + reward_z_advantage(dz, geo["cos_ata"], micro_delta_dist, dt, self.reward_cfg)
                    + reward_energy_loss(vel_norm, a_vel[2], dt, self.reward_cfg)
                    + reward_ground_warning(a_pos[2], a_vel[2], dt, self.reward_cfg)
                    + reward_tracking(
                        geo["cos_ata"], geo["cos_aa"], geo["cos_hca"],
                        cos_collision, sideslip, current_dist, dz, micro_delta_dist,
                        self.warning_radius, self.MAX_SPEED, dt, self.reward_cfg,
                    )
                    + reward_closing_speed(closing_speed, dz, current_dist, 400.0, dt, self.reward_cfg)
                )

            # ── Evader rewards ──────────────────────────────────────────
            if "evader_0" in actions and not terminations["evader_0"]:
                e_r = reward_survival(dt, self.reward_cfg)
                if current_dist <= self.warning_radius:
                    e_r += reward_escape(micro_delta_dist, self.reward_cfg)
                    e_r += reward_spoofing(geo["cos_ata"], dt, self.reward_cfg)
                total_rewards["evader_0"] += e_r

            self.prev_dist = current_dist

            # ── Termination checks ──────────────────────────────────────
            # Collision kill
            if check_collision(current_dist, self.macro_step):
                total_rewards["attacker_0"] += self.reward_cfg.kill_reward
                total_rewards["evader_0"] -= self.reward_cfg.kill_reward
                terminations["attacker_0"] = True
                terminations["evader_0"] = True
                infos["attacker_0"]["reason"] = "success"
                break

            # CPA kill
            if check_cpa(current_dist, self.prev_dist, self.CPA_RADIUS, self.macro_step):
                miss_dist = current_dist
                linear_ratio = np.clip((self.CPA_RADIUS - miss_dist) / (self.CPA_RADIUS - 50.0), 0.0, 1.0)
                terminal_r = self.reward_cfg.cpa_base + self.reward_cfg.cpa_extra_max * linear_ratio
                total_rewards["attacker_0"] += terminal_r
                total_rewards["evader_0"] -= terminal_r
                terminations["attacker_0"] = True
                terminations["evader_0"] = True
                infos["attacker_0"]["reason"] = "success"
                break

            # Ground crash / out of bounds
            crash_occurred = False
            for agent_id, ac in [("attacker_0", self.attacker), ("evader_0", self.evader)]:
                if agent_id not in actions or terminations[agent_id]:
                    continue
                death_floor = self.ATTACKER_DEATH_FLOOR if agent_id == "attacker_0" else self.EVADER_DEATH_FLOOR
                if check_ground_crash(ac.position_ned[2], death_floor):
                    penalty = 2000.0 if agent_id == "attacker_0" else 5000.0
                    total_rewards[agent_id] -= penalty
                    terminations[agent_id] = True
                    infos[agent_id]["reason"] = "ground_crash"
                    crash_occurred = True
                elif check_out_of_bounds(ac.position_ned[2], self.CEILING):
                    total_rewards[agent_id] -= 2000.0
                    terminations[agent_id] = True
                    infos[agent_id]["reason"] = "out_of_bounds"
                    crash_occurred = True

            if crash_occurred:
                break

        # ── Timeout ─────────────────────────────────────────────────────────
        current_time = self.step_counter / self.CTRL_FREQ
        if check_timeout(current_time, self.EPISODE_LEN_SEC):
            for agent_id in self.agents:
                truncations[agent_id] = True
            if not terminations.get("attacker_0", True):
                total_rewards["attacker_0"] -= self.reward_cfg.timeout_attacker_penalty
            if not terminations.get("evader_0", True):
                total_rewards["evader_0"] += self.reward_cfg.timeout_evader_bonus
            infos.setdefault("attacker_0", {})["reason"] = "timeout"

        # ── Build observations ──────────────────────────────────────────────
        obs = {}
        for agent_id in self.agents:
            obs[agent_id] = self._get_obs_dict(agent_id)

        # Zero-padded obs for terminated agents
        for agent_id in self.possible_agents:
            if terminations[agent_id] or truncations[agent_id]:
                if agent_id not in obs:
                    obs[agent_id] = {
                        "obs": np.zeros(19, dtype=np.float32),
                        "global_state": compute_global_state(
                            self.attacker.position_ned, np.zeros(4),
                            self.attacker.velocity_ned, np.zeros(3),
                            self.evader.position_ned, np.zeros(4),
                            self.evader.velocity_ned, np.zeros(3),
                        ),
                    }

        # Remove dead agents
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        # Global flags
        terminations["__all__"] = any(terminations.values())
        truncations["__all__"] = any(truncations.values())

        # Tacview frame
        if self.record_tacview:
            self._record_tacview_frame(current_time)

        return obs, total_rewards, terminations, truncations, infos

    # ── Observation helpers ─────────────────────────────────────────────────

    def _get_obs_dict(self, agent_id: str) -> dict:
        own_ac = self._aircraft_map[agent_id]
        enemy_ac = self.evader if agent_id == "attacker_0" else self.attacker

        local_obs = compute_obs(
            own_ac.position_ned, own_ac.rpy_rad, own_ac.velocity_ned,
            np.zeros(3),  # angular velocity approximation
            enemy_ac.position_ned, enemy_ac.velocity_ned, enemy_ac.rpy_rad,
        )

        global_obs = compute_global_state(
            self.attacker.position_ned, np.zeros(4),  # quaternion placeholder
            self.attacker.velocity_ned, np.zeros(3),
            self.evader.position_ned, np.zeros(4),
            self.evader.velocity_ned, np.zeros(3),
        )

        return {"obs": local_obs, "global_state": global_obs}

    # ── Tacview ─────────────────────────────────────────────────────────────

    def _record_tacview_frame(self, time_sec: float) -> None:
        """Accumulate a Tacview frame for later export."""
        a_state = self.attacker.state
        e_state = self.evader.state
        self._tacview_frames.append({
            "time": time_sec,
            "attacker": {
                "lon_deg": a_state["lon_deg"], "lat_deg": a_state["lat_deg"],
                "alt_ft": a_state["alt_ft"],
                "roll_deg": a_state["roll_deg"], "pitch_deg": a_state["pitch_deg"],
                "yaw_deg": a_state["yaw_deg"],
            },
            "evader": {
                "lon_deg": e_state["lon_deg"], "lat_deg": e_state["lat_deg"],
                "alt_ft": e_state["alt_ft"],
                "roll_deg": e_state["roll_deg"], "pitch_deg": e_state["pitch_deg"],
                "yaw_deg": e_state["yaw_deg"],
            },
        })

    def export_tacview(self, filepath: str) -> None:
        """Write accumulated frames to a Tacview ACMI file."""
        from src.logging.tacview_exporter import TacviewExporter
        exporter = TacviewExporter(filepath)
        exporter.write(self._tacview_frames)
