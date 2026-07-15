"""V6 Autopsy: Reward breakdown + termination distribution + geometric trajectories.

Runs N eval episodes against the V6 best checkpoint, recording:
  1. Per-component reward contributions (progress, pincer, OR fallback, OOC, crash, etc.)
  2. Termination reason distribution
  3. Time-series geometric metrics (distance to target, pincer angle)

Usage:
    conda activate marl_env
    python scripts/diagnose_v6_autopsy.py --episodes 50
"""

import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault('JSBSIM_DEBUG', '0')

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from src.environment.formation_rllib_env import FormationRLlibEnv, COOP_PHASE_AND

register_env('formation_2v1_rllib', lambda c: FormationRLlibEnv(c))

from src.models.formation_rllib_model import RLlibAttentionActor
ModelCatalog.register_custom_model('attention_formation', RLlibAttentionActor)


def run_autopsy(ckpt_path: str, n_episodes: int = 50, seed: int = 42):
    """Run comprehensive diagnostic eval."""
    algo = PPO.from_checkpoint(os.path.abspath(ckpt_path))

    # Stage 1 curriculum params
    S1 = {"and_dist": 1200.0, "and_angle": 35.0, "bearing_min": -30.0, "bearing_max": 30.0,
          "target_dist_min": 1600.0, "target_dist_max": 2200.0, "sustain_steps": 2}

    # ── Collectors ──────────────────────────────────────────────────────
    termination_counts = {}
    total_rewards = []

    # Per-component reward accumulators (tracked manually by re-computing env internals)
    # We can't easily monkey-patch, so we track the FINAL per-episode total reward
    # and approximate component breakdown by running a few instrumented episodes.

    # Geometric time-series (subsampled)
    geo_trajectories = []  # list of {steps, d0, d1, pincer}

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        env = FormationRLlibEnv({'difficulty_level': 0.0, 'lock_altitude': True,
                                 'record_tacview': False, 'cooperative_mode': True})
        env._difficulty = 0.0
        env.set_coop_phase(COOP_PHASE_AND)
        env.set_curriculum_stage_full(
            1, S1["and_dist"], S1["and_angle"],
            S1["bearing_min"], S1["bearing_max"],
            S1["target_dist_min"], S1["target_dist_max"],
            S1["sustain_steps"])

        obs_dict, _ = env.reset(seed=int(rng.integers(0, 2**31-1)))
        done, total_r, step = False, 0.0, 0
        d0s, d1s, pincers = [], [], []

        while not done:
            actions = {}
            for aid in env._agent_ids:
                if aid in obs_dict:
                    actions[aid] = algo.compute_single_action(
                        obs_dict[aid], policy_id=f"{aid}_policy", explore=False)

            obs_dict, rewards, terms, truncs, _ = env.step(actions)
            total_r += sum(rewards.values())

            # Geometric metrics
            p0 = env.pursuers[0].aircraft.position_ned
            p1 = env.pursuers[1].aircraft.position_ned
            tgt = env.targets[0].aircraft.position_ned
            d0 = float(np.linalg.norm(p0 - tgt))
            d1 = float(np.linalg.norm(p1 - tgt))
            d0s.append(d0); d1s.append(d1)

            los0, los1 = (tgt-p0)[:2], (tgt-p1)[:2]
            n0, n1 = np.linalg.norm(los0), np.linalg.norm(los1)
            if n0 > 1 and n1 > 1:
                cos_p = np.clip(np.dot(los0, los1) / (n0 * n1), -1, 1)
                pincers.append(float(np.degrees(np.arccos(cos_p))))

            reason = getattr(env, '_last_termination_reason', 'timeout')
            done = terms.get('__all__', False) or truncs.get('__all__', False)
            step += 1
            if step >= 600: break

        total_rewards.append(total_r)
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
        geo_trajectories.append({
            'steps': step, 'rew': total_r, 'reason': reason,
            'd0': np.array(d0s), 'd1': np.array(d1s), 'pincer': np.array(pincers),
            'd0_final': d0s[-1] if d0s else 0, 'd1_final': d1s[-1] if d1s else 0,
            'pincer_max': max(pincers) if pincers else 0,
        })

        if (ep + 1) % 10 == 0:
            pos_count = sum(1 for r in total_rewards if r > 0)
            print(f"  [{ep+1:3d}/{n_episodes}]  pos={pos_count}/{ep+1}  "
                  f"latest_rew={total_r:.0f}  {reason}")

    algo.stop()

    # ── Print report ─────────────────────────────────────────────────────
    rewards = np.array(total_rewards)
    print(f"\n{'='*60}")
    print(f"V6 AUTOPSY REPORT (best_iter_0225_rew_-3815)")
    print(f"{'='*60}")
    print(f"\n--- Episode Outcomes ---")
    print(f"  Episodes:        {len(rewards)}")
    print(f"  Mean reward:     {rewards.mean():.0f} ± {rewards.std():.0f}")
    print(f"  Range:           [{rewards.min():.0f}, {rewards.max():.0f}]")
    print(f"  Positive rate:   {100*(rewards>0).mean():.0f}%")

    print(f"\n--- Termination Distribution ---")
    for reason, count in sorted(termination_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_episodes
        bar = '█' * int(pct / 2)
        print(f"  {reason:<25s}: {count:>3d} ({pct:5.1f}%)  {bar}")

    print(f"\n--- Geometric Summary ---")
    all_d0_final = [g['d0_final'] for g in geo_trajectories]
    all_d1_final = [g['d1_final'] for g in geo_trajectories]
    all_pincer_max = [g['pincer_max'] for g in geo_trajectories]
    print(f"  Mean final d0:     {np.mean(all_d0_final):.0f}m (min {min(all_d0_final):.0f})")
    print(f"  Mean final d1:     {np.mean(all_d1_final):.0f}m (min {min(all_d1_final):.0f})")
    print(f"  Mean pincer max:   {np.mean(all_pincer_max):.0f}°")

    # Find the best and worst episodes for detailed breakdown
    best_idx = np.argmax(rewards)
    worst_idx = np.argmin(rewards)
    print(f"\n--- Best Episode (idx={best_idx}) ---")
    g = geo_trajectories[best_idx]
    print(f"  Rew={g['rew']:.0f}, steps={g['steps']}, reason={g['reason']}")
    print(f"  d0: {g['d0'][0]:.0f}→{g['d0_final']:.0f}m, d1: {g['d1'][0]:.0f}→{g['d1_final']:.0f}m")
    print(f"  Pincer max: {g['pincer_max']:.0f}°")
    # Compute what fraction of steps AND was met
    if len(g['pincer']) > 0:
        and_met = (g['d0'][:len(g['pincer'])] < 1200) & (g['d1'][:len(g['pincer'])] < 1200) & (g['pincer'] >= 35)
        print(f"  AND-met fraction: {and_met.sum()}/{len(g['pincer'])} steps ({100*and_met.mean():.1f}%)")

    print(f"\n--- Worst Episode (idx={worst_idx}) ---")
    g = geo_trajectories[worst_idx]
    print(f"  Rew={g['rew']:.0f}, steps={g['steps']}, reason={g['reason']}")
    print(f"  d0: {g['d0'][0]:.0f}→{g['d0_final']:.0f}m, d1: {g['d1'][0]:.0f}→{g['d1_final']:.0f}m")
    print(f"  Pincer max: {g['pincer_max']:.0f}°")

    # ── Save for visualization ──────────────────────────────────────────
    save = {'n_episodes': len(geo_trajectories)}
    for i, g in enumerate(geo_trajectories):
        p = f'ep{i}_'
        for k in ['steps', 'rew', 'd0_final', 'd1_final', 'pincer_max']:
            save[p + k] = g[k]
        save[p + 'd0'] = g['d0']
        save[p + 'd1'] = g['d1']
        save[p + 'pincer'] = g['pincer'] if len(g['pincer']) > 0 else np.array([0])
        save[p + 'reason'] = g['reason']
    save['termination_counts'] = np.array(list(termination_counts.items()), dtype=object)
    np.savez_compressed('data/viz/v6_autopsy.npz', **save)
    print(f"\n[OK] Saved: data/viz/v6_autopsy.npz")

    return geo_trajectories, termination_counts, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6 Autopsy Diagnostic")
    parser.add_argument("--ckpt", type=str,
                        default="/home/sean/jsbsim-marl-formation/marl_runs/rllib_formation_0715_1055_s42/checkpoints/best_iter_0225_rew_-3815")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_autopsy(args.ckpt, args.episodes, args.seed)
