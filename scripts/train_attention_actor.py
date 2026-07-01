"""Cold-start training for Self-Attention FormationActor (MAPPO CTDE).

Training modes:
  1v1:        Single-pursuer warmup, mate_scale=0 (no wingman features)
  2v1:        Direct 2v1 cold-start, mate_scale=1 (full attention from scratch)
  curriculum: 1v1 pre-train → 2v1 fine-tune with mate_scale annealing

Key innovation: The Attention Actor learns to dynamically allocate attention
between Self, Target, and Mate token groups WITHOUT relying on tiled SB3 weights.
This forces the network to discover coordination patterns organically.

Usage:
  conda activate jsbsim_rl
  python scripts/train_attention_actor.py --mode curriculum --steps 500000
  python scripts/train_attention_actor.py --mode 1v1 --steps 200000
  python scripts/train_attention_actor.py --mode 2v1 --steps 300000
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.environment.formation_env import FormationEnv
from src.models.attention_actor import AttentionFormationActor, AttentionCritic

# ═══════════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════

GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.01; MAX_GRAD_NORM = 0.5
ACTOR_LR = 3e-4; CRITIC_LR = 1e-3
MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096
REWARD_SCALE = 100.0
OBS_PER_AGENT = 33; GLOBAL_DIM = 21; ACT_DIM = 2; N_PURSUERS = 2

# Curriculum
MATE_SCALE_RAMP_START = 0.0   # start with mate ignored
MATE_SCALE_RAMP_END = 200_000  # steps over which to ramp mate_scale to 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout + GAE
# ═══════════════════════════════════════════════════════════════════════════

def build_global_state(env):
    """21-dim: 2 pursuers + 1 target: pos(3)+vel(3)+heading(1) each."""
    MAX_D, MAX_H, MAX_V = 10000.0, 5000.0, 400.0
    vec = []
    for ps in env.pursuers:
        p = ps.aircraft.position_ned / np.array([MAX_D, MAX_D, MAX_H])
        v = ps.aircraft.velocity_ned / MAX_V
        h = np.array([float(ps.aircraft.state["yaw_deg"]) / 180.0])
        vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
    for ts in env.targets:
        p = ts.aircraft.position_ned / np.array([MAX_D, MAX_D, MAX_H])
        v = ts.aircraft.velocity_ned / MAX_V
        h = np.array([float(ts.aircraft.state["yaw_deg"]) / 180.0])
        vec.extend(np.clip(np.concatenate([p, v, h]), -1, 1))
    return np.array(vec, dtype=np.float32)


@torch.no_grad()
def collect_rollout(env, actor, critic, device, n_steps=ROLLOUT_STEPS, n_pursuers=2):
    """Collect rollout data for N pursuers."""
    buf = {f'p{i}': {'obs':[],'act':[],'rew':[],'val':[],'logp':[],
                      'done':[],'term':[],'gs':[]} for i in range(n_pursuers)}
    obs, _ = env.reset()
    ep_rew = 0.0; ep_count = 0

    for _ in range(n_steps):
        actions = {}
        gs = build_global_state(env)
        gs_t = torch.as_tensor(gs, dtype=torch.float32).unsqueeze(0).to(device)
        val = critic(gs_t).item()

        for i in range(n_pursuers):
            start = i * OBS_PER_AGENT
            o_t = torch.as_tensor(obs[start:start+OBS_PER_AGENT], dtype=torch.float32).unsqueeze(0).to(device)
            loc, scale = actor(o_t)
            dist = Independent(Normal(loc, scale), 1)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)

            k = f'p{i}'
            buf[k]['obs'].append(obs[start:start+OBS_PER_AGENT].copy())
            buf[k]['act'].append(act.cpu().squeeze(0).numpy())
            buf[k]['val'].append(val)
            buf[k]['logp'].append(logp.item())
            buf[k]['done'].append(0.0)
            buf[k]['term'].append(1.0)
            buf[k]['gs'].append(gs.copy())
            actions[f'p{i}'] = act.cpu().squeeze(0).numpy()

        concat_act = np.concatenate([actions[f'p{i}'] for i in range(n_pursuers)])
        next_obs, rew, term, trunc, info = env.step(concat_act)
        done = term or trunc; ep_rew += rew

        # Per-pursuer distances
        p_dists = []
        t_pos = env.targets[0].aircraft.position_ned
        for i in range(n_pursuers):
            p_dists.append(float(np.linalg.norm(env.pursuers[i].aircraft.position_ned - t_pos)))

        # Rubber band penalty (2v1 only)
        rubber_band = 0.0
        if n_pursuers >= 2:
            spacing = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned -
                                            env.pursuers[1].aircraft.position_ned))
            if spacing > 1500.0:
                rubber_band = -0.01 * (spacing - 1500.0) / REWARD_SCALE

        for i in range(n_pursuers):
            k = f'p{i}'
            base_r = rew / REWARD_SCALE / n_pursuers

            # Individual progress bonus
            if hasattr(env.pursuers[i], 'prev_dist'):
                delta = env.pursuers[i].prev_dist - p_dists[i]
                base_r += 0.5 * delta / REWARD_SCALE
            env.pursuers[i].prev_dist = p_dists[i]

            # Kill bonus gating
            if info.get('reason') == 'success':
                kd = p_dists[i]
                if kd < 1500.0:
                    base_r += (3000.0 if i == info.get('kill_agent_idx', 0) else 2000.0) / REWARD_SCALE

            if n_pursuers >= 2:
                base_r += rubber_band / n_pursuers

            buf[k]['rew'].append(base_r)
            buf[k]['done'][-1] = float(done)
            is_terminal = term and not trunc
            buf[k]['term'][-1] = 0.0 if is_terminal else 1.0

        if done:
            ep_count += 1
            next_obs, _ = env.reset()
        obs = next_obs

    # GAE per agent
    gs_final = build_global_state(env)
    final_val = critic(torch.as_tensor(gs_final, dtype=torch.float32).unsqueeze(0).to(device)).item()

    all_data = []
    for i in range(n_pursuers):
        k = f'p{i}'; n = len(buf[k]['rew'])
        adv = np.zeros(n, dtype=np.float32); ret = np.zeros(n, dtype=np.float32); gae = 0.0
        for t in reversed(range(n)):
            nv = (final_val if t == n-1 else buf[k]['val'][t+1]) * buf[k]['term'][t]
            delta = buf[k]['rew'][t] + GAMMA * nv * (1 - buf[k]['done'][t]) - buf[k]['val'][t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - buf[k]['done'][t]) * gae
            adv[t] = gae; ret[t] = adv[t] + buf[k]['val'][t]
        a_t = torch.tensor(adv, dtype=torch.float32)
        a_t = (a_t - a_t.mean()) / (a_t.std() + 1e-8)
        all_data.append({
            'obs': torch.tensor(np.array(buf[k]['obs']), dtype=torch.float32),
            'act': torch.tensor(np.array(buf[k]['act']), dtype=torch.float32),
            'logp': torch.tensor(buf[k]['logp'], dtype=torch.float32),
            'val': torch.tensor(buf[k]['val'], dtype=torch.float32),
            'adv': a_t, 'ret': torch.tensor(ret, dtype=torch.float32),
            'old_val': torch.tensor(buf[k]['val'], dtype=torch.float32),
            'gs': torch.tensor(np.array(buf[k]['gs']), dtype=torch.float32),
        })
    return all_data, ep_rew / max(ep_count, 1), ep_count


# ═══════════════════════════════════════════════════════════════════════════
#  PPO Update
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(actor, critic, actor_opt, critic_opt, all_data, device, epochs=PPO_EPOCHS):
    actor.train(); critic.train()
    all_vals, all_rets = [], []

    for _ in range(epochs):
        for data in all_data:
            n = len(data['obs'])
            idx_all = torch.randperm(n)
            for start in range(0, n, MINI_BATCH_SIZE):
                idx = idx_all[start:start+MINI_BATCH_SIZE]
                if len(idx) == 0: continue

                b_obs = data['obs'][idx].to(device); b_act = data['act'][idx].to(device)
                b_adv = data['adv'][idx].to(device); b_ret = data['ret'][idx].to(device)
                b_logp = data['logp'][idx].to(device); b_oldv = data['old_val'][idx].to(device)
                b_gs = data['gs'][idx].to(device)

                mb_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                loc, scale = actor(b_obs)
                dist = Independent(Normal(loc, scale), 1)
                new_logp = dist.log_prob(b_act).sum(-1)
                ratio = torch.exp(new_logp - b_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (only for cold-start; zero for hot-start)
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - ENT_COEF * entropy

                val_pred = critic(b_gs)
                v_clip = b_oldv + torch.clamp(val_pred - b_oldv, -CLIP_EPS, CLIP_EPS)
                v_l1 = (val_pred - b_ret)**2; v_l2 = (v_clip - b_ret)**2
                critic_loss = 0.5 * torch.max(v_l1, v_l2).mean()

                actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM); actor_opt.step()
                critic_opt.zero_grad(); (VF_COEF * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM); critic_opt.step()
                all_vals.append(val_pred.detach()); all_rets.append(b_ret)

    ev = 1.0 - torch.var(torch.cat(all_rets) - torch.cat(all_vals)) / (torch.var(torch.cat(all_rets)) + 1e-8) if all_vals else 0.0
    return float(ev)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_stats(actor, device):
    """Sample attention weights to monitor learning progress."""
    actor.eval()
    with torch.no_grad():
        dummy = torch.randn(16, 33).to(device)
        (_, _), info = actor(dummy, return_attention=True)
        attn = info['attn_weights']  # [16, 3, 3]
        # attn[i, j, k]: token i attends to token k
        # Row 0 = Self token, Row 1 = Target token, Row 2 = Mate token
        self_to_mate = attn[:, 0, 2].mean().item()   # Self attends to Mate
        self_to_target = attn[:, 0, 1].mean().item()  # Self attends to Target
        mate_to_self = attn[:, 2, 0].mean().item()    # Mate attends to Self
        pool = info['pool_weights'].squeeze(1)  # [16, 3]
        pool_self = pool[:, 0].mean().item()
        pool_target = pool[:, 1].mean().item()
        pool_mate = pool[:, 2].mean().item()
    actor.train()
    return {
        'attn_self2mate': self_to_mate, 'attn_self2target': self_to_target,
        'attn_mate2self': mate_to_self,
        'pool_self': pool_self, 'pool_target': pool_target, 'pool_mate': pool_mate,
    }


def train(mode="curriculum", total_steps=500000, difficulty=0.0, seed=42,
          load_ckpt=None, log_attn_every=50):
    """Main training loop.

    Args:
        mode: "1v1", "2v1", or "curriculum" (1v1 pre-train → 2v1 fine-tune)
        total_steps: Total environment steps
        difficulty: Initial difficulty level
        seed: Random seed
        load_ckpt: Resume from checkpoint .pth
        log_attn_every: Log attention statistics every N epochs
    """
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"attn_{mode}_{ts}_s{seed}"
    log_dir = f"./marl_runs/{run_name}"; os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Self-Attention MAPPO Cold-Start Training")
    print(f"  Mode:      {mode}")
    print(f"  Steps:     {total_steps:,}")
    print(f"  Difficulty: {difficulty:.2f}")
    print(f"  Device:    {device}")
    print(f"  Log dir:   {log_dir}")
    print(f"{'='*60}\n")

    # ── Build networks ─────────────────────────────────────────────────
    actor = AttentionFormationActor(mate_scale=0.0 if mode == "1v1" else MATE_SCALE_RAMP_START).to(device)
    critic = AttentionCritic().to(device)

    if load_ckpt:
        ck = torch.load(load_ckpt, map_location='cpu')
        actor.load_state_dict(ck['actor']); critic.load_state_dict(ck['critic'])
        print(f'Loaded checkpoint: {load_ckpt}')

    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-5)

    # ── Environment ────────────────────────────────────────────────────
    if mode == "1v1":
        # 1v1: single pursuer, no mate features needed
        env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)
        n_pursuers = 1
        actor.mate_scale = 0.0  # mate features are zero anyway
    else:
        env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)
        n_pursuers = 2

    # ── Training loop ──────────────────────────────────────────────────
    total = 0; epoch = 0; rew_win = deque(maxlen=10)
    best_rew = -float('inf')

    while total < total_steps:
        # Mate-scale curriculum (ramp mate_scale from start → 1.0)
        if mode in ("2v1", "curriculum") and n_pursuers >= 2:
            if total < MATE_SCALE_RAMP_END:
                frac = total / MATE_SCALE_RAMP_END
                # Smooth ramp with cosine schedule
                mate_scale = MATE_SCALE_RAMP_START + (1.0 - MATE_SCALE_RAMP_START) * (1 - np.cos(np.pi * frac)) / 2
            else:
                mate_scale = 1.0
            actor.mate_scale = mate_scale

        # LR annealing
        if total > 0:
            frac = 1.0 - total / total_steps
            actor_opt.param_groups[0]["lr"] = ACTOR_LR * max(frac, 0.1)
            critic_opt.param_groups[0]["lr"] = CRITIC_LR * max(frac, 0.1)

        all_data, avg_rew, n_ep = collect_rollout(env, actor, critic, device, n_pursuers=n_pursuers)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        ev = ppo_update(actor, critic, actor_opt, critic_opt, all_data, device)

        # Logging
        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            parts = [
                f"[{run_name}] step={total:>7d}",
                f"rew={avg_rew:8.1f}",
                f"avg10={avg10:8.1f}",
                f"ev={ev:5.3f}",
                f"lr_a={actor_opt.param_groups[0]['lr']:.1e}",
                f"eps={n_ep}",
            ]
            if n_pursuers >= 2:
                parts.append(f"mate_s={actor.mate_scale:.2f}")
            print("  ".join(parts))

        # Attention statistics (periodic)
        if epoch % log_attn_every == 0 and n_pursuers >= 2:
            attn_stats = compute_attention_stats(actor, device)
            print(f"  [Attn] self2target={attn_stats['attn_self2target']:.3f}  "
                  f"self2mate={attn_stats['attn_self2mate']:.3f}  "
                  f"mate2self={attn_stats['attn_mate2self']:.3f}  |  "
                  f"pool(S/T/M)={attn_stats['pool_self']:.3f}/{attn_stats['pool_target']:.3f}/{attn_stats['pool_mate']:.3f}")

        # Save best
        avg10 = np.mean(rew_win) if rew_win else avg_rew
        if avg10 > best_rew and len(rew_win) >= 5:
            best_rew = avg10
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "epoch": epoch, "total_steps": total,
                "mode": mode, "mate_scale": actor.mate_scale,
            }, os.path.join(log_dir, "best_policy.pth"))

        epoch += 1

    # ── Save final ─────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "final_policy.pth")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "epoch": epoch, "total_steps": total,
        "mode": mode, "mate_scale": actor.mate_scale,
    }, final_path)
    print(f"\n[Done] {final_path}")
    print(f"  Best avg10 rew: {best_rew:.1f}")
    return log_dir


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-start Self-Attention MAPPO Training")
    parser.add_argument("--mode", type=str, default="curriculum",
                        choices=["1v1", "2v1", "curriculum"],
                        help="Training mode (default: curriculum)")
    parser.add_argument("--steps", type=int, default=500000,
                        help="Total environment steps")
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Initial difficulty level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load", type=str, default=None,
                        help="Resume from checkpoint .pth")
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    train(args.mode, args.steps, args.difficulty, args.seed, args.load)
