"""Evasive target maneuver diagnostic for dual-actor cooperative model.

Runs the best dual-actor checkpoint against targets with evasive patterns:
  - straight: baseline straight-and-level
  - spiral:  3D climbing/descending spiral (constant turn + altitude oscillation)
  - lissajous: horizontal snake curve (Lissajous trajectory)
  - weave:   random heading reversals (aggressive jinking)

Exports Tacview + per-step pincer diagnostics for each pattern.

Usage:
  python scripts/diagnose_dual_evasion.py
  python scripts/diagnose_dual_evasion.py --pattern all --episodes 2
"""

import argparse, os, sys, warnings, logging
import numpy as np
import torch
from torch.distributions import Independent, Normal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

from src.environment.formation_env import FormationEnv, COOP_PHASE_AND
from src.models.attention_actor import AttentionFormationActor, AttentionCritic
from scripts.train_attention_actor import build_global_tokens

CKPT = "benchmarks/dual_actor_coop_best.pth"
OUTPUT = "results/evasion_diag"

PATTERNS = {
    "straight": "Straight & level (baseline)",
    "spiral": "3D climbing spiral (turn 3deg/s + alt +/-200m @ 0.1Hz)",
    "lissajous": "Lissajous snake (X=800sin(0.03t), Y=800sin(0.05t))",
    "weave": "Aggressive weave (heading reversal every 8s, +/-60deg)",
}


def load_dual_model(device="cpu"):
    actors = [AttentionFormationActor(mate_scale=1.0).to(device) for _ in range(2)]
    critic = AttentionCritic().to(device)
    ck = torch.load(CKPT, map_location='cpu')
    for i, a in enumerate(actors):
        a.load_state_dict(ck[f'actor_p{i}'])
    critic.load_state_dict(ck['critic'])
    for a in actors: a.eval()
    critic.eval()
    return actors, critic


def apply_target_evasion(env, step_idx, pattern, base_hdg, base_alt=3000.0):
    """Modify target ref_hdg/ref_alt for evasive behavior. Call before env.step()."""
    t = step_idx * 0.5  # decision interval = 0.5s

    if pattern == "straight":
        env.targets[0].ref_hdg = base_hdg
        env.targets[0].ref_alt_m = base_alt

    elif pattern == "spiral":
        turn_rate = 3.0  # deg/s
        env.targets[0].ref_hdg = (base_hdg + turn_rate * t) % 360.0
        env.targets[0].ref_alt_m = base_alt + 200.0 * np.sin(0.1 * 2 * np.pi * t)

    elif pattern == "lissajous":
        # Lissajous heading: derivative of the curve
        dx = 800 * 0.03 * np.cos(0.03 * t)
        dy = 800 * 0.05 * np.cos(0.05 * t)
        desired_hdg = float(np.degrees(np.arctan2(dy, dx))) % 360.0
        env.targets[0].ref_hdg = desired_hdg
        env.targets[0].ref_alt_m = base_alt

    elif pattern == "weave":
        period = 16  # full cycle in steps (8s)
        phase = (step_idx % period) / period
        if phase < 0.5:
            env.targets[0].ref_hdg = (base_hdg + 60.0) % 360.0
        else:
            env.targets[0].ref_hdg = (base_hdg - 60.0) % 360.0
        env.targets[0].ref_alt_m = base_alt


def run_episode(actors, critic, device, pattern, ep_idx, difficulty=0.0):
    _stderr = sys.stderr; sys.stderr = open(os.devnull, 'w')
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty,
                       cooperative_mode=True, record_tacview=True)
    env.set_coop_phase(COOP_PHASE_AND)
    obs, _ = env.reset()
    sys.stderr = _stderr

    base_hdg = float(env.targets[0].aircraft.state["yaw_deg"])
    base_alt = 3000.0
    done = False; step = 0
    d0s, d1s, p_angs, sustains, kills = [], [], [], [], []

    while not done and step < 240:  # max 120 seconds
        # Apply target evasion
        apply_target_evasion(env, step, pattern, base_hdg, base_alt)

        # Dual-actor actions
        acts = []
        for i in range(2):
            o_t = torch.as_tensor(obs[i*33:(i+1)*33], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                loc, _ = actors[i](o_t)
            acts.append(loc.squeeze(0).numpy())

        obs, rew, term, trunc, info = env.step(np.concatenate(acts))
        done = term or trunc

        p0 = env.pursuers[0].aircraft.position_ned
        p1 = env.pursuers[1].aircraft.position_ned
        t = env.targets[0].aircraft.position_ned
        d0 = float(np.linalg.norm(p0 - t)); d1 = float(np.linalg.norm(p1 - t))
        los0 = t[:2] - p0[:2]; los1 = t[:2] - p1[:2]
        n0 = float(np.linalg.norm(los0)); n1 = float(np.linalg.norm(los1))
        pa = float(np.degrees(np.arccos(np.clip(np.dot(los0, los1) / (n0 * n1), -1, 1)))) if n0 > 1 and n1 > 1 else 0
        d0s.append(d0); d1s.append(d1); p_angs.append(pa)
        sustains.append(env._coop_sustain_counter)
        kills.append(d0 < 800 and d1 < 800 and pa >= 30)
        step += 1

    reason = info.get('reason', 'timeout')
    kill_pct = 100 * sum(kills) / max(step, 1)
    max_sus = max(sustains)

    os.makedirs(OUTPUT, exist_ok=True)
    acmi = os.path.join(OUTPUT, f"{pattern}_ep{ep_idx}_{reason}.txt.acmi")
    env.export_tacview(acmi)

    print(f"  [{pattern}] ep{ep_idx}: {reason:20s} | d0={np.mean(d0s):.0f}m d1={np.mean(d1s):.0f}m | "
          f"pincer={np.mean(p_angs):.0f}deg | kill_zone={kill_pct:.0f}% | max_sus={max_sus} | {acmi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="all",
                        help="Target pattern: straight, spiral, lissajous, weave, or all")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--difficulty", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cpu")
    actors, critic = load_dual_model(device)
    print(f"Loaded: {CKPT}")
    print(f"Patterns: {list(PATTERNS.keys()) if args.pattern == 'all' else [args.pattern]}\n")

    patterns = list(PATTERNS.keys()) if args.pattern == "all" else [args.pattern]
    for pat in patterns:
        print(f"--- {pat}: {PATTERNS[pat]} ---")
        for ep in range(args.episodes):
            run_episode(actors, critic, device, pat, ep + 1, args.difficulty)
        print()

    print(f"Tacview files: {OUTPUT}/")


if __name__ == "__main__":
    main()
