"""Evaluate 1v1 shoot checkpoints — launch timing & engagement statistics.

Usage:
  python scripts/eval_shoot_1v1.py \
      --checkpoint marl_runs/shoot_v11_4_A/checkpoints/best \
      --episodes 100 --difficulty 0.0,0.3 --outdir results/shoot_eval

Reports per difficulty:
  - launch window quality (bad/good/premium ratio)
  - launch geometry (range/ATA/AA/closure at the fire moment)
  - hit rate, miss distance, kills per episode
  - WEZ(geometry) first-entry -> first-fire latency
  - termination reason distribution
  - JSON summary for cross-version comparison

Auto-detects legacy (36-dim, no closure features) vs extended (38-dim)
checkpoints from the restored policy's observation space.
"""
import os, sys, warnings, logging, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['JSBSIM_DEBUG'] = '0'
warnings.filterwarnings('ignore')
for n in ['jsbsim', 'gymnasium']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.models.shoot_mask_model import ShootMaskModel
from src.utils.geometry import compute_forward_vector

ENV_NAME = "jsbsim_shoot_1v1"
ENV_ALIASES = ["jsbsim_shoot_v101", "jsbsim_shoot_1v1_v1"]
ACTION_DIMS = [3, 5, 1, 2]  # speed, heading, altitude, fire
MAX_STEPS = 1500


def env_creator(config):
    return BaseEnv(task=SingleCombatShootTask(config))


def decode_action(raw):
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


def launch_geometry(env):
    """Range / ATA / AA / closure at the moment of fire."""
    ps = env.pursuers[0]
    tgt = env.targets[0]
    p_pos = ps.aircraft.position_ned
    t_pos = tgt.aircraft.position_ned
    los = t_pos - p_pos
    dist = float(np.linalg.norm(los))
    los_dir = los / max(dist, 1e-6)
    p_fwd = compute_forward_vector(ps.aircraft.rpy_rad)
    t_fwd = compute_forward_vector(tgt.aircraft.rpy_rad)
    ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(p_fwd, los_dir))))))
    aa_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t_fwd, los_dir))))))
    closure = float(np.dot(tgt.aircraft.velocity_ned - ps.aircraft.velocity_ned, los_dir))
    return {"range_m": dist, "ata_deg": ata_deg, "aa_deg": aa_deg, "closure_mps": closure}


def classify_launch(g):
    """Same window classes as the task's QualityBonus (v10.3+).

    closure = d(range)/dt; NEGATIVE = closing (good window).
    """
    if g["closure_mps"] < 0 and g["ata_deg"] < 10.0 and 2000.0 < g["range_m"] < 4000.0:
        return "premium"
    if g["closure_mps"] < 0:
        return "good"
    return "bad"


def summarize(name, vals):
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def evaluate_checkpoint(ckpt_path, difficulties, episodes, seed, outdir):
    ray.init(ignore_reinit_error=True, num_cpus=1, logging_level="ERROR")
    ModelCatalog.register_custom_model("shoot_mask_model", ShootMaskModel)
    for _name in [ENV_NAME] + ENV_ALIASES:
        register_env(_name, lambda c: env_creator(c))

    algo = PPO.from_checkpoint(os.path.abspath(ckpt_path))
    policy = algo.get_policy("default_policy")
    pol_obs_dim = int(policy.observation_space.shape[0])
    include_closure = pol_obs_dim >= 38
    print(f"[eval] policy obs dim={pol_obs_dim} -> obs_include_closure={include_closure}")

    tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))) or "run"
    all_summaries = {}

    for diff in difficulties:
        n_ep = 0
        ep_rewards = []
        ep_steps = []
        term_reasons = {}
        launches_total = 0
        launch_geo = {"range_m": [], "ata_deg": [], "aa_deg": [], "closure_mps": []}
        quality = {"bad": 0, "good": 0, "premium": 0}
        hits_total = 0
        kills = 0
        miss_dist_hits = []
        misses_total = 0
        wez_fire_latency = []

        for ep in range(episodes):
            env = BaseEnv(task=SingleCombatShootTask({
                "difficulty_level": diff,
                "obs_include_closure": include_closure,
            }))
            obs, _ = env.reset(seed=seed + ep)
            ep_r = 0.0
            steps = 0
            reason = "timeout"
            wez_first = None
            fire_first = None
            ep_launches = 0
            ep_hits = 0

            for step in range(MAX_STEPS):
                act = decode_action(policy.compute_single_action(obs["p0"], explore=False)[0])
                obs, rews, terms, truncs, info = env.step({"p0": act})
                ep_r += rews.get("p0", 0.0)
                ep_hits += env.task._hit_this_step.get("p0", 0)
                if wez_first is None and env.task._is_valid_launch_envelope(env.pursuers[0], env.targets[0]):
                    wez_first = step
                if env.task._has_launched_this_step.get("p0", False):
                    if fire_first is None:
                        fire_first = step
                    g = launch_geometry(env)
                    launch_geo["range_m"].append(g["range_m"])
                    launch_geo["ata_deg"].append(g["ata_deg"])
                    launch_geo["aa_deg"].append(g["aa_deg"])
                    launch_geo["closure_mps"].append(g["closure_mps"])
                    quality[classify_launch(g)] += 1
                    ep_launches += 1
                steps = step + 1
                if terms.get("__all__") or truncs.get("__all__"):
                    reason = info.get("p0", {}).get("termination_reason", "unknown")
                    break
                if not np.isfinite(obs["p0"]).all():
                    reason = "jsbsim_nan"
                    break

            for m in env.pursuers[0].launch_missiles:
                if not m.is_done:
                    continue
                if m.is_success:
                    miss_dist_hits.append(m.miss_distance)
                else:
                    misses_total += 1

            launches_total += ep_launches
            hits_total += ep_hits
            if reason == "target_killed":
                kills += 1
            term_reasons[reason] = term_reasons.get(reason, 0) + 1
            ep_rewards.append(ep_r)
            ep_steps.append(steps)
            if wez_first is not None and fire_first is not None:
                wez_fire_latency.append(fire_first - wez_first)
            n_ep += 1
            env.close()

        summary = {
            "difficulty": diff,
            "episodes": n_ep,
            "episode_rewards": [float(r) for r in ep_rewards],
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            "termination_reasons": term_reasons,
            "launches": launches_total,
            "launches_per_episode": launches_total / max(n_ep, 1),
            "launch_geometry": {k: summarize(k, v) for k, v in launch_geo.items()},
            "launch_quality": quality,
            "hit_rate": hits_total / max(launches_total, 1),
            "hits": hits_total,
            "misses": misses_total,
            "hit_miss_distance_m": summarize("miss_distance", miss_dist_hits),
            "kills": kills,
            "kill_rate": kills / max(n_ep, 1),
            "wez_to_fire_latency_steps": summarize("latency", wez_fire_latency),
        }
        all_summaries[str(diff)] = summary

        print(f"\n===== difficulty={diff} ({n_ep} episodes) =====")
        print(f"  mean reward: {summary['mean_reward']:+.1f}   mean steps: {summary['mean_steps']:.0f}")
        print(f"  term reasons: {term_reasons}")
        print(f"  launches: {launches_total} ({summary['launches_per_episode']:.1f}/ep)  quality: {quality}")
        print(f"  hit rate: {summary['hit_rate']*100:.1f}% ({hits_total}/{launches_total})  "
              f"kills: {kills}/{n_ep}  misses: {misses_total}")
        if miss_dist_hits:
            print(f"  hit miss distance: mean={np.mean(miss_dist_hits):.0f}m "
                  f"median={np.median(miss_dist_hits):.0f}m")
        if launch_geo["range_m"]:
            print(f"  launch range: mean={np.mean(launch_geo['range_m']):.0f}m "
                  f"median={np.median(launch_geo['range_m']):.0f}m")
            print(f"  launch ATA: mean={np.mean(launch_geo['ata_deg']):.1f}°  "
                  f"AA: mean={np.mean(launch_geo['aa_deg']):.1f}°  "
                  f"closure: mean={np.mean(launch_geo['closure_mps']):+.0f}m/s")
        if wez_fire_latency:
            print(f"  WEZ->fire latency: mean={np.mean(wez_fire_latency):.1f} steps "
                  f"({np.mean(wez_fire_latency)*0.2:.1f}s)")

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"eval_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": os.path.abspath(ckpt_path), "seed": seed,
                   "results": all_summaries},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[json] {out_path}")
    compare_to_baseline(out_path)
    ray.shutdown()


def compare_to_baseline(out_path, baseline_path="results/shoot_eval/BASELINE_v19.json"):
    """Print a delta table of this run vs the v19 baseline (if available)."""
    if not os.path.exists(baseline_path):
        print("\n[baseline] no baseline file at %s — skipped" % baseline_path)
        return
    if not os.path.exists(out_path):
        return
    cur = json.load(open(out_path, encoding="utf-8")).get("results", {})
    base = json.load(open(baseline_path, encoding="utf-8")).get("results", {})

    def _lp(res):
        return res.get("launches_per_episode", res.get("launches_per_ep", 0.0))

    def _lost(res):
        n = res.get("episodes", 100)
        reasons = res.get("termination_reasons", {})
        if reasons:
            return reasons.get("lost_target", 0) / max(n, 1)
        pct = res.get("lost_target_pct")
        if pct is not None:
            return pct / 100.0
        return 0.0

    def _mean(res):
        return res.get("mean_reward", 0.0)

    print("\n===== vs v19 baseline =====")
    for diff in sorted(set(list(cur.keys()) + list(base.keys()))):
        c, b = cur.get(diff), base.get(diff)
        if not c or not b:
            continue
        print(f"  difficulty={diff}")
        rows = [
            ("launches/ep", _lp(c), _lp(b), 2),
            ("hit rate", c.get("hit_rate", 0.0), b.get("hit_rate", 0.0), 1),
            ("kill rate", c.get("kill_rate", 0.0), b.get("kill_rate", 0.0), 1),
            ("lost_target %", _lost(c), _lost(b), 1),
            ("mean reward", _mean(c), _mean(b), 0),
        ]
        for label, cv, bv, kind in rows:
            d = cv - bv
            if kind == 0:
                print(f"    {label:<14s} {cv:+9.0f}  vs base {bv:+9.0f}  (delta {d:+8.0f})")
            elif kind == 1:
                print(f"    {label:<14s} {cv*100:5.1f}%  vs base {bv*100:5.1f}%  (delta {d*100:+5.1f}pp)")
            else:
                print(f"    {label:<14s} {cv:5.2f}  vs base {bv:5.2f}  (delta {d:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1v1 shoot checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to an RLlib checkpoint (e.g. marl_runs/.../checkpoints/best)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default="0.0",
                        help="comma-separated difficulties, e.g. 0.0,0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/shoot_eval")
    args = parser.parse_args()

    difficulties = [float(x) for x in args.difficulty.split(",") if x.strip()]
    if not difficulties:
        parser.error("--difficulty must contain at least one value")
    evaluate_checkpoint(args.checkpoint, difficulties, args.episodes, args.seed, args.outdir)


if __name__ == "__main__":
    main()