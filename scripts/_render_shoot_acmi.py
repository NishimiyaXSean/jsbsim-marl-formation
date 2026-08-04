"""Render Stage1 + Stage2 ACMI with tail-chase geometry."""
import os, sys, warnings, logging
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask

ENV = "jsbsim_shoot_1v1"
OUTDIR = "results/shoot_training"


def render_one(tag, ckpt, difficulty, seed, acmi_path):
    register_env(ENV, lambda c: BaseEnv(task=SingleCombatShootTask(c)))
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level="ERROR")
    algo = PPO.from_checkpoint(os.path.abspath(ckpt))
    policy = algo.get_policy("default_policy")

    env = BaseEnv(task=SingleCombatShootTask({"difficulty_level": difficulty}))
    # Use default reset geometry (training geometry, ~1km range)
    obs, _ = env.reset(seed=seed)
    env.task.reset(env)  # re-init task state after reset
    p0 = env.pursuers[0]; t0 = env.targets[0]

    env.enable_acmi_logging(acmi_path)
    env.log_acmi_step()

    s_p0 = p0.aircraft.state; s_t0 = t0.aircraft.state
    print(f"[{tag}] P0=({s_p0['lat_deg']:.4f},{s_p0['lon_deg']:.4f}) T0=({s_t0['lat_deg']:.4f},{s_t0['lon_deg']:.4f})")
    dist = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
    print(f"[{tag}] Initial distance: {dist:.0f}m")
    total_r, fires, min_d = 0, 0, 99999

    for step in range(500):
        a = policy.compute_single_action(obs["p0"], explore=False)[0]
        if isinstance(a, (int, np.integer)):
            a = np.array([a % 3, (a // 3) % 5, (a // 15) % 3, (a // 45) % 2], dtype=np.int64)
        else:
            a = np.asarray(a, dtype=np.int64).flatten()
        if a[3] == 1:
            fires += 1
        obs, rews, terms, truncs, info = env.step({"p0": a})
        env.log_acmi_step()
        total_r += rews.get("p0", 0)
        d = float(np.linalg.norm(p0.aircraft.position_ned - t0.aircraft.position_ned))
        if d < min_d:
            min_d = d
        if step < 5 or step % 30 == 0 or a[3] == 1:
            bd = env.task._reward_breakdown
            lr = bd.get("LaunchSuccess", {}).get("p0", 0)
            fire_mark = "FIRE!" if a[3] == 1 else ""
            print(f"  step {step:3d}: dist={d:.0f}m rew={total_r:+.0f} launch={lr:+.0f} {fire_mark}")
        if terms.get("__all__") or truncs.get("__all__"):
            break

    reason = info.get("p0", {}).get("termination_reason", "timeout")
    print(f"[{tag}] steps={step + 1} rew={total_r:+.0f} min_d={min_d:.0f}m fires={fires} {reason}")
    env.close()
    ray.shutdown()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    configs = [
        ("stage1", "marl_runs/shoot_v3_stage1_s42/checkpoints/best", 0.0),
        ("stage2", "marl_runs/shoot_v3_stage2_s42/checkpoints/best", 0.3),
    ]
    for tag, ckpt, diff in configs:
        acmi = os.path.join(OUTDIR, f"shoot_{tag}_s42.acmi")
        render_one(tag, ckpt, diff, 42, acmi)
    print(f"\nACMI files saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
