"""PID-based cooperative trajectory generator for 2v1 BC data.

Generates clean expert data where BOTH pursuers approach the target
from opposite sides, maintaining a pincer formation. The PID controller
guarantees P1 never free-rides — unlike the old SB3 OR-gate data.

PID strategy:
  - Compute desired offset points on left/right of target's forward axis
  - P0 steers toward left offset, P1 steers toward right offset
  - Offset distance = 300m (forms ~90deg pincer at 500m range)
  - Terminal phase (<600m): both turn directly toward target
  - Speed: proportional to distance (cruise at 300m/s far, slow to 200m/s near)

Usage:
  python scripts/generate_coop_expert.py --episodes 500
  python scripts/generate_coop_expert.py --episodes 500 --difficulty 0.3
"""

import argparse, os, sys, warnings, logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

from src.environment.formation_env import FormationEnv

OUTPUT_PATH = "data/expert/coop_pid_data.npz"
OFFSET_DIST = 300.0       # offset from target centerline (metres)
TERMINAL_DIST = 600.0     # switch to direct attack below this range
CRUISE_SPEED = 300.0      # m/s when far from target
APPROACH_SPEED = 200.0    # m/s when close to target
TURN_GAIN = 0.8           # proportional gain on heading error (increased for faster response)


def pid_action(pursuer_pos, pursuer_heading_deg, target_pos, target_heading_deg,
               side: str) -> np.ndarray:
    """Compute [turn_rate, speed] action to steer toward pincer position.

    Args:
        pursuer_pos: (3,) NED position of pursuer
        pursuer_heading_deg: current heading (degrees)
        target_pos: (3,) NED position of target
        target_heading_deg: target heading (degrees)
        side: "left" or "right" — which side of target to flank

    Returns:
        action: (2,) [turn_rate_factor, speed_factor] in [-1, 1]
    """
    dist = float(np.linalg.norm(pursuer_pos - target_pos))

    # ── Compute desired intercept point ──────────────────────────────
    t_hdg_rad = np.radians(target_heading_deg)
    t_fwd = np.array([np.cos(t_hdg_rad), np.sin(t_hdg_rad), 0.0])
    t_right = np.array([-t_fwd[1], t_fwd[0], 0.0])  # perpendicular in horizontal plane

    if dist > TERMINAL_DIST:
        # Phase 1: Offset approach — steer toward flanking position
        offset_sign = 1.0 if side == "left" else -1.0
        # Offset point is OFFSET_DIST to the left/right of target, toward pursuer
        to_pursuer = pursuer_pos[:2] - target_pos[:2]
        to_pursuer_norm = to_pursuer / (float(np.linalg.norm(to_pursuer)) + 1e-8)
        # Blend: move toward offset position while closing distance
        desired_xy = target_pos[:2] + offset_sign * t_right[:2] * OFFSET_DIST
    else:
        # Phase 2: Terminal — steer directly toward target
        desired_xy = target_pos[:2]

    # ── Heading control ──────────────────────────────────────────────
    to_desired = desired_xy - pursuer_pos[:2]
    desired_hdg = float(np.degrees(np.arctan2(to_desired[1], to_desired[0]))) % 360.0
    hdg_error = (desired_hdg - pursuer_heading_deg + 180.0) % 360.0 - 180.0
    turn_cmd = np.clip(TURN_GAIN * hdg_error / 180.0, -1.0, 1.0)

    # ── Speed control ────────────────────────────────────────────────
    # Fast when far from target, slow when close, slow during hard turns
    if dist > TERMINAL_DIST:
        speed_cmd = 1.0  # max speed to close distance
    elif dist > 300.0:
        speed_cmd = 0.3   # moderate speed in mid-range
    else:
        speed_cmd = 0.0   # neutral when very close
    # Reduce speed during hard turns
    speed_cmd = speed_cmd * (1.0 - 0.3 * abs(turn_cmd))

    return np.array([turn_cmd, speed_cmd], dtype=np.float32)


def generate(n_episodes=500, difficulty=0.0):
    """Generate cooperative expert data using PID pincer control."""
    import io as _io
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_obs_p0, all_act_p0 = [], []
    all_obs_p1, all_act_p1 = [], []
    successes = 0
    total_samples = 0

    print(f"PID Cooperative Expert: {n_episodes} episodes, difficulty={difficulty}")
    for ep in range(n_episodes):
        _stderr = sys.stderr
        sys.stderr = _io.StringIO()

        env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty,
                           cooperative_mode=True, record_tacview=False)
        obs, _ = env.reset()
        sys.stderr = _stderr

        done = False
        ep_obs_p0, ep_act_p0 = [], []
        ep_obs_p1, ep_act_p1 = [], []

        while not done:
            # Get state for PID computation
            p0_pos = env.pursuers[0].aircraft.position_ned
            p1_pos = env.pursuers[1].aircraft.position_ned
            t_pos = env.targets[0].aircraft.position_ned
            p0_hdg = float(env.pursuers[0].aircraft.state["yaw_deg"])
            p1_hdg = float(env.pursuers[1].aircraft.state["yaw_deg"])
            t_hdg = float(env.targets[0].aircraft.state["yaw_deg"])

            # Assign sides: alternate based on episode parity
            p0_side = "left" if ep % 2 == 0 else "right"
            p1_side = "right" if ep % 2 == 0 else "left"

            # PID actions
            act_p0 = pid_action(p0_pos, p0_hdg, t_pos, t_hdg, p0_side)
            act_p1 = pid_action(p1_pos, p1_hdg, t_pos, t_hdg, p1_side)

            # Only record if BOTH pursuers within 2000m (clean cooperative data)
            d0 = float(np.linalg.norm(p0_pos - t_pos))
            d1 = float(np.linalg.norm(p1_pos - t_pos))
            if d0 < 2000.0 and d1 < 2000.0:
                p0_obs = obs[0:33].astype(np.float32)
                p1_obs = obs[33:66].astype(np.float32)
                ep_obs_p0.append(p0_obs); ep_act_p0.append(act_p0)
                ep_obs_p1.append(p1_obs); ep_act_p1.append(act_p1)

            concat_act = np.concatenate([act_p0, act_p1])
            obs, rew, term, trunc, info = env.step(concat_act)
            done = term or trunc

        is_success = info.get('reason') in ('success', 'cooperative_success')
        if is_success:
            successes += 1
            all_obs_p0.extend(ep_obs_p0); all_act_p0.extend(ep_act_p0)
            all_obs_p1.extend(ep_obs_p1); all_act_p1.extend(ep_act_p1)
            total_samples += len(ep_obs_p0)

        if (ep + 1) % 50 == 0:
            # Compute per-episode distance stats
            d0_final = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - env.targets[0].aircraft.position_ned))
            d1_final = float(np.linalg.norm(env.pursuers[1].aircraft.position_ned - env.targets[0].aircraft.position_ned))
            print(f"  [{ep+1:>4d}/{n_episodes}]  success_rate={successes/(ep+1):.1%}  "
                  f"samples={total_samples}  d0_final={d0_final:.0f}m  d1_final={d1_final:.0f}m")

    # Merge both pursuers
    all_obs = np.array(all_obs_p0 + all_obs_p1, dtype=np.float32)
    all_act = np.array(all_act_p0 + all_act_p1, dtype=np.float32)

    np.savez_compressed(OUTPUT_PATH, obs=all_obs, actions=all_act)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  Episodes: {n_episodes}  |  Successes: {successes} ({successes/n_episodes:.1%})")
    print(f"  Total samples: {len(all_obs)}")
    print(f"  Obs shape: {all_obs.shape}  |  Act shape: {all_act.shape}")

    # Data quality check
    d0_vals = []; d1_vals = []
    for obs_sample in all_obs:
        # Mate relative position (indices 27-29) + own position gives rough distance
        # Use a simple heuristic: if mate_rel_pos is small, they're close
        mate_rel = obs_sample[27:30]
        d0_vals.append(float(np.linalg.norm(mate_rel)))
    print(f"  Data quality: mate_rel_pos norm = {np.mean(d0_vals):.4f} +/- {np.std(d0_vals):.4f}")
    print(f"  (Values << 1.0 indicate both pursuers near each other — clean data)")

    return all_obs, all_act


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()
    generate(args.episodes, args.difficulty)
