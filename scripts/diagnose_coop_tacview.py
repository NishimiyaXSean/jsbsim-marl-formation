"""Diagnostic Tacview export for cooperative 2v1 model.

Runs the best checkpoint, records per-step AND-gate diagnostics
(distances, pincer angle, sustain counter), and exports Tacview.

Usage:
  python scripts/diagnose_coop_tacview.py
  python scripts/diagnose_coop_tacview.py --episodes 3 --output results/coop_diag/
"""

import argparse, os, sys, warnings, logging
import numpy as np
import torch
from torch.distributions import Independent, Normal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

from src.environment.formation_env import FormationEnv
from src.models.attention_actor import AttentionFormationActor, AttentionCritic
from scripts.train_attention_actor import build_global_tokens

CKPT_PATH = "marl_runs/attn_2v1_coop_0702_1241_s42/best_policy.pth"


def run_diagnostic(n_episodes=3, difficulty=0.0, output_dir="results/coop_diag"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load model
    actor = AttentionFormationActor(mate_scale=1.0).to(device)
    critic = AttentionCritic().to(device)
    ck = torch.load(CKPT_PATH, map_location='cpu')
    actor.load_state_dict(ck['actor'])
    critic.load_state_dict(ck['critic'])
    actor.eval(); critic.eval()
    print(f"Loaded: {CKPT_PATH}")
    print(f"  Training stage: {ck.get('stage', '?')}  mate_scale={ck.get('mate_scale', '?')}")

    for ep in range(n_episodes):
        # Suppress JSBSIM stderr
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty,
                           cooperative_mode=True, record_tacview=True)
        # Set AND-gate to final strict values (matching end-of-training)
        env.set_coop_curriculum(1.0)
        print(f"\nEpisode {ep+1}: AND-gate d<{env._coop_success_dist:.0f}m, a>{env._coop_success_angle:.0f}deg")

        obs, _ = env.reset()
        sys.stderr = _stderr

        done = False; step = 0
        ep_data = {
            'times': [], 'd0': [], 'd1': [], 'pincer_angle': [],
            'sustain': [], 'both_kill': [], 'act_p0': [], 'act_p1': [],
            'spacing': [],
        }

        while not done:
            # Per-pursuer action via shared Actor
            actions = []
            for i in range(2):
                o_t = torch.as_tensor(obs[i*33:(i+1)*33], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    loc, scale = actor(o_t)
                actions.append(loc.squeeze(0).numpy())

            concat_act = np.concatenate(actions)
            obs, rew, term, trunc, info = env.step(concat_act)
            done = term or trunc

            # Compute diagnostics
            p0 = env.pursuers[0].aircraft.position_ned
            p1 = env.pursuers[1].aircraft.position_ned
            t = env.targets[0].aircraft.position_ned
            d0 = float(np.linalg.norm(p0 - t))
            d1 = float(np.linalg.norm(p1 - t))
            spacing = float(np.linalg.norm(p0 - p1))

            los0 = t[:2] - p0[:2]; los1 = t[:2] - p1[:2]
            n0 = float(np.linalg.norm(los0)); n1 = float(np.linalg.norm(los1))
            if n0 > 1 and n1 > 1:
                cos_a = np.clip(float(np.dot(los0, los1)) / (n0 * n1), -1, 1)
                pincer = float(np.degrees(np.arccos(cos_a)))
            else:
                pincer = 0.0

            both_kill = (d0 < env._coop_success_dist and d1 < env._coop_success_dist
                         and pincer >= env._coop_success_angle)

            ep_data['times'].append(step * 0.5)
            ep_data['d0'].append(d0); ep_data['d1'].append(d1)
            ep_data['pincer_angle'].append(pincer)
            ep_data['sustain'].append(env._coop_sustain_counter)
            ep_data['both_kill'].append(both_kill)
            ep_data['act_p0'].append(actions[0].copy())
            ep_data['act_p1'].append(actions[1].copy())
            ep_data['spacing'].append(spacing)

            step += 1

        reason = info.get('reason', 'timeout')
        success = reason == 'cooperative_success'

        # Export Tacview
        acmi_path = os.path.join(output_dir, f"coop_ep{ep+1}_{reason}.txt.acmi")
        env.export_tacview(acmi_path)

        # Print diagnostic summary
        kill_steps = sum(ep_data['both_kill'])
        max_sustain = max(ep_data['sustain'])
        print(f"  Result: {reason}  |  steps={step}  |  success={success}")
        print(f"  Kill zone: {kill_steps}/{step} steps ({100*kill_steps/step:.0f}%)  |  max sustain: {max_sustain}")
        print(f"  d0: {np.mean(ep_data['d0']):.0f} +/- {np.std(ep_data['d0']):.0f}m  "
              f"d1: {np.mean(ep_data['d1']):.0f} +/- {np.std(ep_data['d1']):.0f}m")
        print(f"  pincer: {np.mean(ep_data['pincer_angle']):.1f} +/- {np.std(ep_data['pincer_angle']):.1f} deg  "
              f"(max: {max(ep_data['pincer_angle']):.1f})")
        print(f"  spacing: {np.mean(ep_data['spacing']):.0f} +/- {np.std(ep_data['spacing']):.0f}m")
        print(f"  act_p0: turn={np.mean([a[0] for a in ep_data['act_p0']]):.2f} +/- {np.std([a[0] for a in ep_data['act_p0']]):.2f}  "
              f"spd={np.mean([a[1] for a in ep_data['act_p0']]):.2f}")
        print(f"  act_p1: turn={np.mean([a[0] for a in ep_data['act_p1']]):.2f} +/- {np.std([a[0] for a in ep_data['act_p1']]):.2f}  "
              f"spd={np.mean([a[1] for a in ep_data['act_p1']]):.2f}")

        # Kill-zone burst analysis: find consecutive stretches inside kill zone
        bursts = []
        in_burst = False; burst_start = 0
        for i, bk in enumerate(ep_data['both_kill']):
            if bk and not in_burst:
                in_burst = True; burst_start = i
            elif not bk and in_burst:
                in_burst = False
                duration = (i - burst_start) * 0.5
                if duration >= 0.05:  # at least 1 micro-step
                    bursts.append({'start_s': burst_start*0.5, 'dur_s': duration,
                                   'peak_pincer': max(ep_data['pincer_angle'][burst_start:i])})
        if in_burst:
            duration = (len(ep_data['both_kill']) - burst_start) * 0.5
            bursts.append({'start_s': burst_start*0.5, 'dur_s': duration,
                           'peak_pincer': max(ep_data['pincer_angle'][burst_start:])})

        print(f"  Kill-zone bursts: {len(bursts)}")
        for b in bursts:
            flag = "COOP_SUCCESS" if b['dur_s'] >= 0.1 else "brief"
            print(f"    t={b['start_s']:.1f}s  dur={b['dur_s']:.2f}s  peak_pincer={b['peak_pincer']:.0f}deg  [{flag}]")

        print(f"  Tacview: {acmi_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="results/coop_diag")
    args = parser.parse_args()
    run_diagnostic(args.episodes, args.difficulty, args.output)
