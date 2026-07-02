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
from torch.utils.tensorboard import SummaryWriter

from src.environment.formation_env import FormationEnv
from src.models.attention_actor import AttentionFormationActor, AttentionCritic

# ═══════════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════

GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.005; ENT_COEF_MIN = 0.001; ENT_COEF_MAX = 0.02; MAX_GRAD_NORM = 0.5
ACTOR_LR_WARMUP = 0.0; ACTOR_LR_FINE = 1e-5
CRITIC_LR_WARMUP = 1e-3; CRITIC_LR_FINE = 5e-4
MINI_BATCH_SIZE = 128; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096  # more frequent EV checks
REWARD_SCALE = 100.0
OBS_PER_AGENT = 33; TOKEN_DIM = 7; ACT_DIM = 2; N_PURSUERS = 2
EV_UNFREEZE_THRESHOLD = 0.3
ENTROPY_COLLAPSE_THRESH = 0.5  # if action entropy drops below this, boost ENT_COEF
KL_TARGET = 0.015
TOKEN_ORDER_SELF = 0; TOKEN_ORDER_MATE = 1; TOKEN_ORDER_TARGET = 2


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout + GAE
# ═══════════════════════════════════════════════════════════════════════════

def build_global_tokens(env, pursuer_idx: int):
    """Build per-pursuer tokenized global state: (3, 7) = [Self, Mate, Target].

    Each token: pos(3) + vel(3) + heading(1), normalized to [-1, 1].
    Token order is view-relative — P0 sees [P0, P1, Target], P1 sees [P1, P0, Target].
    This view-relative tokenization lets the Critic output values aligned with
    the Actor's perspective, dramatically reducing fitting difficulty.
    """
    MAX_D, MAX_H, MAX_V = 10000.0, 5000.0, 400.0

    def entity_features(aircraft):
        p = aircraft.position_ned / np.array([MAX_D, MAX_D, MAX_H])
        v = aircraft.velocity_ned / MAX_V
        h = np.array([float(aircraft.state["yaw_deg"]) / 180.0])
        return np.clip(np.concatenate([p, v, h]), -1, 1).astype(np.float32)

    n_pursuers = len(env.pursuers)
    n_targets = len(env.targets)

    # Self token
    self_feat = entity_features(env.pursuers[pursuer_idx].aircraft)

    # Mate token: the OTHER pursuer (or self if only 1 pursuer)
    if n_pursuers >= 2:
        mate_idx = 1 if pursuer_idx == 0 else 0
        mate_feat = entity_features(env.pursuers[mate_idx].aircraft)
    else:
        mate_feat = np.zeros(TOKEN_DIM, dtype=np.float32)

    # Target token
    target_feat = entity_features(env.targets[0].aircraft)

    tokens = np.stack([self_feat, mate_feat, target_feat], axis=0)  # (3, 7)
    return tokens


@torch.no_grad()
def collect_rollout(env, actor, critic, device, n_steps=ROLLOUT_STEPS, n_pursuers=2):
    """Collect rollout data for N pursuers with per-pursuer tokenized global state."""
    buf = {f'p{i}': {'obs':[],'act':[],'rew':[],'val':[],'logp':[],
                      'done':[],'term':[],'gs':[]} for i in range(n_pursuers)}
    obs, _ = env.reset()
    ep_rew = 0.0; ep_count = 0

    for _ in range(n_steps):
        actions = {}
        # Per-pursuer view-relative tokenized global state + value
        gs_tokens = {}; vals = {}
        for i in range(n_pursuers):
            gs_i = build_global_tokens(env, i)  # (3, 7) view-relative
            gs_tokens[i] = gs_i
            gs_t = torch.as_tensor(gs_i, dtype=torch.float32).unsqueeze(0).to(device)
            vals[i] = critic(gs_t).item()

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
            buf[k]['val'].append(vals[i])
            buf[k]['logp'].append(logp.item())
            buf[k]['done'].append(0.0)
            buf[k]['term'].append(1.0)
            buf[k]['gs'].append(gs_tokens[i].copy())
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

    # GAE per agent (per-pursuer tokenized final values)
    final_vals = {}
    for i in range(n_pursuers):
        gs_final_i = build_global_tokens(env, i)
        gs_t = torch.as_tensor(gs_final_i, dtype=torch.float32).unsqueeze(0).to(device)
        final_vals[i] = critic(gs_t).item()

    all_data = []
    for i in range(n_pursuers):
        k = f'p{i}'; n = len(buf[k]['rew'])
        adv = np.zeros(n, dtype=np.float32); ret = np.zeros(n, dtype=np.float32); gae = 0.0
        fv = final_vals[i]
        for t in reversed(range(n)):
            nv = (fv if t == n-1 else buf[k]['val'][t+1]) * buf[k]['term'][t]
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
            'gs': torch.tensor(np.array(buf[k]['gs']), dtype=torch.float32),  # (n, 3, 7)
        })
    return all_data, ep_rew / max(ep_count, 1), ep_count


# ═══════════════════════════════════════════════════════════════════════════
#  PPO Update
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(actor, critic, actor_opt, critic_opt, all_data, device,
               epochs=PPO_EPOCHS, kl_target=KL_TARGET, ent_coef=ENT_COEF):
    actor.train(); critic.train()
    all_vals, all_rets = [], []
    kl_skips = 0; total_minibatches = 0

    for epoch_idx in range(epochs):
        for data in all_data:
            n = len(data['obs'])
            idx_all = torch.randperm(n)
            for start in range(0, n, MINI_BATCH_SIZE):
                idx = idx_all[start:start+MINI_BATCH_SIZE]
                if len(idx) == 0: continue
                total_minibatches += 1

                b_obs = data['obs'][idx].to(device); b_act = data['act'][idx].to(device)
                b_adv = data['adv'][idx].to(device); b_ret = data['ret'][idx].to(device)
                b_logp = data['logp'][idx].to(device); b_oldv = data['old_val'][idx].to(device)
                b_gs = data['gs'][idx].to(device)

                mb_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                loc, scale = actor(b_obs)
                dist = Independent(Normal(loc, scale), 1)
                new_logp = dist.log_prob(b_act).sum(-1)
                ratio = torch.exp(new_logp - b_logp)

                # KL early stopping (skip minibatch if KL too high)
                if epoch_idx > 0:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                        if approx_kl > kl_target:
                            kl_skips += 1
                            continue

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                if ent_coef > 0:
                    entropy = dist.entropy().mean()
                    actor_loss = actor_loss - ent_coef * entropy

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
    return float(ev), kl_skips


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def compute_attention_stats(actor, critic, device, n_pursuers=2):
    """Sample attention weights + entropy for both Actor and Critic."""
    actor.eval(); critic.eval()
    with torch.no_grad():
        dummy_obs = torch.randn(16, 33).to(device)
        (_, _), info = actor(dummy_obs, return_attention=True)
        attn = info['attn_weights']  # [16, 3, 3]
        self_to_mate = attn[:, 0, 2].mean().item()
        self_to_target = attn[:, 0, 1].mean().item()
        mate_to_self = attn[:, 2, 0].mean().item()
        pool = info['pool_weights'].squeeze(1)
        pool_self = pool[:, 0].mean().item()
        pool_target = pool[:, 1].mean().item()
        pool_mate = pool[:, 2].mean().item()
        eps = 1e-8
        mha_entropy = -(attn * (attn + eps).log()).sum(-1).mean().item()
        pool_entropy = -(pool * (pool + eps).log()).sum(-1).mean().item()

        # Critic attention
        dummy_gs = torch.randn(16, 3, 7).to(device)
        _, crit_attn = critic(dummy_gs, return_attention=True)
        crit_mha = crit_attn['mha_weights']   # [16, 3, 3]
        crit_pool = crit_attn['pool_weights']  # [16, 1, 3]
        crit_mha_ent = -(crit_mha * (crit_mha + eps).log()).sum(-1).mean().item()
        crit_pool_ent = -(crit_pool * (crit_pool + eps).log()).sum(-1).mean().item()
        # Critic mate attention: column 1 = Mate token
        crit_mate_mha = crit_mha[:, :, 1].mean().item()
        crit_mate_pool = crit_pool[:, 0, 1].mean().item()
    actor.train(); critic.train()
    return {
        'attn_self2mate': self_to_mate, 'attn_self2target': self_to_target,
        'attn_mate2self': mate_to_self,
        'pool_self': pool_self, 'pool_target': pool_target, 'pool_mate': pool_mate,
        'mha_entropy': mha_entropy, 'pool_entropy': pool_entropy,
        'critic_mha_ent': crit_mha_ent, 'critic_pool_ent': crit_pool_ent,
        'critic_mate_mha': crit_mate_mha, 'critic_mate_pool': crit_mate_pool,
    }


def train(mode="2v1", total_steps=200000, difficulty=0.0, seed=42,
          load_ckpt=None, load_bc=None, log_attn_every=50,
          critic_warmup_steps=100_000, actor_frozen_lr=0.0,
          critic_warmup_lr=5e-4, actor_fine_lr=1e-5, critic_fine_lr=1e-4,
          ev_unfreeze_threshold=EV_UNFREEZE_THRESHOLD, kl_target=KL_TARGET,
          ent_coef=ENT_COEF, use_mate_ramp=False, cooperative=False):
    """Two-stage MAPPO fine-tuning for Attention Actor with EV gating + KL protection.

    Stage 1 (Critic Warmup): Actor frozen at BC weights, Critic learns V(s).
      - EV-gated: only unfreeze when EV >= ev_unfreeze_threshold.
    Stage 2 (Joint Fine-tune): Actor unfrozen with tiny LR, KL early stopping.

    Args:
        use_mate_ramp: If True, use cosine ramp 0→1 for mate_scale.
        cooperative: If True, use Phase 5 cooperative 2v1 (pincer + AND-gate + asymmetric).
    """
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    stage_label = "coop" if cooperative else ("bc2v1" if (load_bc and not use_mate_ramp) else ("bc_finetune" if load_bc else "cold"))
    run_name = f"attn_{mode}_{stage_label}_{ts}_s{seed}"
    log_dir = f"./marl_runs/{run_name}"; os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard"); os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Self-Attention MAPPO Training (EV-gated + KL protection)")
    print(f"  Mode:      {mode}  |  Steps: {total_steps:,}  |  BC: {load_bc is not None}")
    print(f"  Cooperative: {cooperative}  |  EV gate: {ev_unfreeze_threshold}  |  KL: {kl_target}")
    print(f"  Mate ramp: {use_mate_ramp}  |  Difficulty: {difficulty:.2f}")
    print(f"  Log dir:   {log_dir}")
    print(f"  TensorBoard: {tb_dir}")
    print(f"{'='*60}\n")

    # ── Environment ────────────────────────────────────────────────────
    if mode == "1v1":
        env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)
        n_pursuers = 1
    else:
        env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty,
                           cooperative_mode=cooperative)
        n_pursuers = 2

    # ── Build networks ─────────────────────────────────────────────────
    init_mate_scale = 0.0 if (mode == "1v1" or use_mate_ramp) else 1.0
    actor = AttentionFormationActor(mate_scale=init_mate_scale).to(device)
    critic = AttentionCritic().to(device)  # tokenized: (3, 7) → scalar
    print(f"  Tokenized global: 3 entities x 7 features (view-relative per-pursuer)")
    print(f"  Actor: {sum(p.numel() for p in actor.parameters()):,} params")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")

    if load_bc:
        bc = torch.load(load_bc, map_location='cpu')
        actor.load_state_dict(bc['actor_state'])
        vl = bc.get('val_loss', None)
        vl_str = f"val_loss={vl:.6f}" if isinstance(vl, float) else ""
        bc_mate = bc.get('mate_scale', '?')
        print(f"  BC Actor loaded: {load_bc}  {vl_str}  mate_scale={bc_mate}")
        # If BC was trained with mate_scale=1.0, keep it at 1.0
        if isinstance(bc_mate, (int, float)) and bc_mate == 1.0:
            actor.mate_scale = 1.0

    if load_ckpt:
        ck = torch.load(load_ckpt, map_location='cpu')
        actor.load_state_dict(ck['actor']); critic.load_state_dict(ck['critic'])
        print(f'  MAPPO checkpoint loaded: {load_ckpt}')

    # ── Optimisers ─────────────────────────────────────────────────────
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR_WARMUP, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_warmup_lr, eps=1e-5)

    # ── Training loop ──────────────────────────────────────────────────
    total = 0; epoch = 0; rew_win = deque(maxlen=10)
    best_rew = -float('inf')
    stage = "WARMUP"; last_ev = 0.0
    warmup_done = False

    if not load_bc and not load_ckpt:
        critic_warmup_steps = 0; stage = "FINE_TUNE"; warmup_done = True

    while total < total_steps:
        # ── EV-gated stage transition ──────────────────────────────────
        if total >= critic_warmup_steps and not warmup_done:
            if last_ev >= ev_unfreeze_threshold:
                warmup_done = True; stage = "FINE_TUNE"
                actor_opt.param_groups[0]["lr"] = actor_fine_lr
                critic_opt.param_groups[0]["lr"] = critic_fine_lr
                print(f"\n[MAPPO] === Actor unfrozen (EV={last_ev:.3f} >= {ev_unfreeze_threshold}) ===\n")
            else:
                print(f"\n[MAPPO] === Step {total}: EV={last_ev:.3f} < {ev_unfreeze_threshold}, extending warmup ===\n")

        # ── Actor frozen during warmup ─────────────────────────────────
        if not warmup_done:
            actor_opt.param_groups[0]["lr"] = actor_frozen_lr
            critic_opt.param_groups[0]["lr"] = critic_warmup_lr
            actor.eval()
        else:
            actor.train()

        # ── Mate-scale ──────────────────────────────────────────────
        if use_mate_ramp and n_pursuers >= 2:
            if total < MATE_SCALE_RAMP_END:
                frac = total / MATE_SCALE_RAMP_END
                mate_scale = MATE_SCALE_RAMP_START + (1.0 - MATE_SCALE_RAMP_START) * (1 - np.cos(np.pi * frac)) / 2
            else:
                mate_scale = 1.0
            actor.mate_scale = mate_scale

        # ── Cooperative curriculum ramp (AND-gate easing) ────────────
        if cooperative:
            progress = total / total_steps
            env.set_coop_curriculum(progress)

        # ── Entropy guard: boost ent_coef if action entropy collapses ─
        if warmup_done and total > critic_warmup_steps:
            with torch.no_grad():
                sample_obs = all_data[0]['obs'][:64].to(device)
                loc, scale = actor(sample_obs)
                dist = Independent(Normal(loc, scale), 1)
                action_entropy = dist.entropy().mean().item()
            if action_entropy < ENTROPY_COLLAPSE_THRESH:
                ent_coef = min(ent_coef * 1.5, ENT_COEF_MAX)
                if epoch % 10 == 0:
                    print(f"  [Entropy] action_ent={action_entropy:.3f} < {ENTROPY_COLLAPSE_THRESH}, "
                          f"ent_coef boosted to {ent_coef:.4f}", flush=True)
            elif action_entropy > ENTROPY_COLLAPSE_THRESH * 2.0:
                ent_coef = max(ent_coef * 0.95, ENT_COEF_MIN)

        # ── LR annealing ───────────────────────────────────────────────
        if warmup_done and total > critic_warmup_steps:
            frac = 1.0 - (total - critic_warmup_steps) / max(total_steps - critic_warmup_steps, 1)
            actor_opt.param_groups[0]["lr"] = actor_fine_lr * max(frac, 0.1)
            critic_opt.param_groups[0]["lr"] = critic_fine_lr * max(frac, 0.1)

        all_data, avg_rew, n_ep = collect_rollout(env, actor, critic, device, n_pursuers=n_pursuers)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        ev, kl_skips = ppo_update(actor, critic, actor_opt, critic_opt, all_data, device,
                                   ent_coef=ent_coef, kl_target=kl_target)
        last_ev = ev

        # ── Logging ────────────────────────────────────────────────────
        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            parts = [
                f"[{run_name}] step={total:>7d}",
                f"rew={avg_rew:8.1f}",
                f"avg10={avg10:8.1f}",
                f"ev={ev:5.3f}",
                f"lr_a={actor_opt.param_groups[0]['lr']:.1e}",
                f"eps={n_ep}",
                f"[{stage[:4]}]",
            ]
            if n_pursuers >= 2:
                parts.append(f"mate_s={actor.mate_scale:.2f}")
            if cooperative:
                parts.append(f"coop_d={env._coop_success_dist:.0f}")
                parts.append(f"coop_a={env._coop_success_angle:.0f}")
            if kl_skips > 0:
                parts.append(f"kl_skip={kl_skips}")
            print("  ".join(parts), flush=True)

            # TensorBoard scalars
            writer.add_scalar("Train/Reward", avg_rew, total)
            writer.add_scalar("Train/Avg10Reward", avg10, total)
            writer.add_scalar("Train/EV", ev, total)
            writer.add_scalar("Train/ActorLR", actor_opt.param_groups[0]['lr'], total)
            writer.add_scalar("Train/CriticLR", critic_opt.param_groups[0]['lr'], total)
            writer.add_scalar("Train/Episodes", n_ep, total)
            writer.add_scalar("Train/KL_Skips", kl_skips, total)
            if n_pursuers >= 2:
                writer.add_scalar("Train/MateScale", actor.mate_scale, total)

        # ── Attention statistics (Actor + Critic) ──────────────────────
        if epoch % log_attn_every == 0 and n_pursuers >= 2:
            attn_stats = compute_attention_stats(actor, critic, device)
            mha_e = attn_stats['mha_entropy']; pool_e = attn_stats['pool_entropy']
            collapse_flags = []
            if mha_e < 0.3: collapse_flags.append(f"MHA_COLLAPSE(H={mha_e:.2f})")
            if pool_e < 0.3: collapse_flags.append(f"POOL_COLLAPSE(H={pool_e:.2f})")
            if mha_e > 1.05: collapse_flags.append(f"MHA_UNIFORM(H={mha_e:.2f})")
            status = " | ".join(collapse_flags) if collapse_flags else "HEALTHY"
            print(f"  [Actor Attn] S2T={attn_stats['attn_self2target']:.3f}  "
                  f"S2M={attn_stats['attn_self2mate']:.3f}  "
                  f"H_mha={mha_e:.3f} H_pool={pool_e:.3f}  |  pool={attn_stats['pool_self']:.2f}/{attn_stats['pool_target']:.2f}/{attn_stats['pool_mate']:.2f}  [{status}]", flush=True)
            print(f"  [Crit Attn] mate_mha={attn_stats['critic_mate_mha']:.3f}  "
                  f"mate_pool={attn_stats['critic_mate_pool']:.3f}  |  "
                  f"H_mha={attn_stats['critic_mha_ent']:.3f} H_pool={attn_stats['critic_pool_ent']:.3f}", flush=True)

            # TensorBoard attention
            for k, v in attn_stats.items():
                writer.add_scalar(f"Attn/{k}", v, total)

        # ── Save best ──────────────────────────────────────────────────
        avg10 = np.mean(rew_win) if rew_win else avg_rew
        if avg10 > best_rew and len(rew_win) >= 5:
            best_rew = avg10
            torch.save({
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "epoch": epoch, "total_steps": total,
                "mode": mode, "mate_scale": actor.mate_scale, "stage": stage,
            }, os.path.join(log_dir, "best_policy.pth"))

        epoch += 1

    # ── Save final ─────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "final_policy.pth")
    torch.save({
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "epoch": epoch, "total_steps": total,
        "mode": mode, "mate_scale": actor.mate_scale, "stage": stage,
    }, final_path)
    writer.close()
    print(f"\n[Done] {final_path}")
    print(f"  Best avg10 rew: {best_rew:.1f}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
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
                        help="Resume from MAPPO checkpoint .pth (actor + critic)")
    parser.add_argument("--load-bc", type=str, default=None,
                        help="Load BC-pretrained Actor weights (.pth from train_attention_bc.py)")
    parser.add_argument("--warmup", type=int, default=50_000,
                        help="Critic warmup steps with Actor frozen (default: 50000)")
    parser.add_argument("--critic-warmup-lr", type=float, default=5e-4,
                        help="Critic LR during warmup (default: 5e-4)")
    parser.add_argument("--actor-fine-lr", type=float, default=1e-5,
                        help="Actor LR after unfreeze (default: 1e-5)")
    parser.add_argument("--critic-fine-lr", type=float, default=1e-4,
                        help="Critic LR after warmup (default: 1e-4)")
    parser.add_argument("--cooperative", action="store_true",
                        help="Use Phase 5 cooperative 2v1 (pincer + AND-gate + asymmetric)")
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    # Suppress JSBSIM C-level stderr (massive aircraft config dump)
    _stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        train(args.mode, args.steps, args.difficulty, args.seed, args.load,
              load_bc=args.load_bc, critic_warmup_steps=args.warmup,
              actor_frozen_lr=0.0, critic_warmup_lr=args.critic_warmup_lr,
              actor_fine_lr=args.actor_fine_lr, critic_fine_lr=args.critic_fine_lr,
              cooperative=args.cooperative)
    finally:
        sys.stderr = _stderr_backup
