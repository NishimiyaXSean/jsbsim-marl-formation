"""Render ACMI for 1v1 shoot training checkpoints (Stage 1 + Stage 2)."""
import os, sys, warnings, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask

# Register the canonical env name + legacy aliases so old checkpoints render.
ENV = "jsbsim_shoot_1v1"
ACTION_DIMS = [3, 16, 1, 2]  # speed, heading sector, altitude, fire


def _decode_action(raw):
    """Convert RLlib compute_single_action output to a per-dim action vector."""
    if isinstance(raw, (int, np.integer)):
        flat = int(raw)
    else:
        arr = np.asarray(raw).reshape(-1)
        if arr.size == len(ACTION_DIMS):
            return arr.astype(np.int64)
        flat = int(arr[0])
    out = np.zeros(len(ACTION_DIMS), dtype=np.int64)
    for i, d in enumerate(ACTION_DIMS):
        out[i] = flat % d
        flat //= d
    return out


for _name in [ENV, "jsbsim_shoot_v101", "jsbsim_shoot_1v1_v1"]:
    register_env(_name, lambda c: BaseEnv(task=SingleCombatShootTask(c)))

def render_checkpoint(ckpt_path, difficulty, seed, acmi_path, label):
    """Load RLlib checkpoint and render one episode to ACMI."""
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level="ERROR")

    algo = PPO.from_checkpoint(os.path.abspath(ckpt_path))
    policy = algo.get_policy("default_policy")

    pol_obs_dim = int(policy.observation_space.shape[0])
    env = BaseEnv(task=SingleCombatShootTask({
        "difficulty_level": difficulty,
        "obs_include_closure": pol_obs_dim >= 38,
    }))
    obs, _ = env.reset(seed=seed)

    # Force tail-chase geometry for visual clarity
    from src.dynamics.flight_controller import FlightControlTargets
    from src.environment.formation_task import PHYSICS_DT
    p0 = env.pursuers[0]; t0 = env.targets[0]
    t_hdg, t_spd, t_alt = 0.0, 230.0, 3000.0
    p_spd = 280.0
    t0.aircraft.reset(lat_deg=30.02, lon_deg=120.0, alt_ft=int(t_alt*3.28084),
                      heading_deg=t_hdg, speed_kts=int(t_spd/0.5144), trim=False)
    t0.aircraft.position_ned = np.array([2224.0, 0.0, t_alt])
    t0.ref_hdg, t0.ref_alt_m = t_hdg, t_alt
    p_ned_north = 2224.0 - 4000.0
    p_lat = 30.0 + p_ned_north / 111320.0
    p0.aircraft.reset(lat_deg=p_lat, lon_deg=120.0, alt_ft=int(t_alt*3.28084),
                      heading_deg=t_hdg, speed_kts=int(p_spd/0.5144), trim=False)
    p0.aircraft.position_ned = np.array([p_ned_north, 0.0, t_alt])
    p0.ref_hdg, p0.ref_alt_m = t_hdg, t_alt
    p0._cmd_speed = p_spd
    for _ in range(int(1.0 * 60)):
        for ac, hdg, alt, spd in [(p0, t_hdg, t_alt, p_spd), (t0, t_hdg, t_alt, t_spd)]:
            s = ac.aircraft.state
            tgt = FlightControlTargets(heading_deg=hdg, altitude_m=alt, speed_mps=spd)
            thr, elev, ail, rud = ac.fc.compute(s, tgt, PHYSICS_DT)
            ac.aircraft.set_controls(throttle=thr, elevator=elev, aileron=ail, rudder=rud)
            ac.aircraft.run()
            ac.aircraft.position_ned[0:2] += ac.aircraft.velocity_ned[0:2] * PHYSICS_DT
            ac.aircraft.position_ned[2] = s["alt_m"]

    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()

    total_r = 0; fires = 0; hits = 0; min_d = 99999
    for step in range(500):
        o = obs["p0"]
        act_raw = policy.compute_single_action(o, explore=False)[0]
        act = _decode_action(act_raw)
        if act[3] == 1: fires += 1
        obs, rews, terms, truncs, info = env.step({"p0": act})
        env.log_acmi_step()
        total_r += rews.get("p0", 0)
        d = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
        if d < min_d: min_d = d
        if not t0.is_alive: hits += 1  # count hit events
        if step < 5 or step % 20 == 0 or act[3] == 1:
            bd = env.task._reward_breakdown
            lr = bd.get("LaunchSuccess", {}).get("p0", 0)
            print(f"  [{label}] step {step:3d}: dist={d:.0f}m  rew={total_r:+.0f}  "
                  f"{'FIRE!' if act[3]==1 else ''}  launch_r={lr:+.0f}")
        if terms.get("__all__") or truncs.get("__all__"):
            break

    reason = info.get("p0", {}).get("termination_reason", "timeout")
    print(f"[{label}] Done: steps={step+1} rew={total_r:+.0f} min_d={min_d:.0f}m "
          f"fires={fires} {reason}")
    env.close()
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/shoot_training")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    configs = [
        ("stage1", "marl_runs/shoot_stage1_s42/checkpoints/best", 0.0, "Stage1 (直飞靶)"),
        ("stage2", "marl_runs/shoot_stage2_s42/checkpoints/best", 0.3, "Stage2 (规避靶)"),
    ]

    for tag, ckpt, diff, label in configs:
        acmi = os.path.join(args.outdir, f"shoot_{tag}_s{args.seed}.acmi")
        print(f"\n=== {label} ===")
        render_checkpoint(ckpt, diff, args.seed, acmi, tag)


if __name__ == "__main__":
    main()
