"""Dual-Actor MAPPO CTDE — independent P0/P1 networks to break symmetry.

Each pursuer gets its own AttentionActor + optimizer. Shared Critic.
BC pretrained weights loaded into both actors, then PPO fine-tuning
with separate surrogate losses and gradient steps.

Usage:
  python scripts/train_dual_actor.py --mode 2v1 --steps 500000 \
    --load-bc data/expert/attention_bc_2v1_filtered_pretrained.pth --cooperative
"""

import argparse, datetime, os, sys, warnings, logging
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from src.environment.formation_env import FormationEnv, COOP_PHASE_OR, COOP_PHASE_AND
from src.models.attention_actor import AttentionFormationActor, AttentionCritic
from scripts.train_attention_actor import build_global_tokens

# ═══════════════════════════════════════════════════════════════════════════
GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.005; ENT_COEF_MIN = 0.001; ENT_COEF_MAX = 0.02
MAX_GRAD_NORM = 0.5
ACTOR_LR_WARMUP = 0.0; ACTOR_LR_FINE = 1e-5
CRITIC_LR_WARMUP = 1e-3; CRITIC_LR_FINE = 5e-4
MINI_BATCH_SIZE = 128; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096
REWARD_SCALE = 100.0
OBS_PER_AGENT = 33; TOKEN_DIM = 7; N_PURSUERS = 2
EV_UNFREEZE_THRESHOLD = 0.3; KL_TARGET = 0.015
ENTROPY_COLLAPSE_THRESH = 0.5


@torch.no_grad()
def collect_rollout(env, actors, critic, device, n_pursuers=2):
    buf = {f'p{i}': {'obs':[],'act':[],'rew':[],'val':[],'logp':[],
                      'done':[],'term':[],'gs':[]} for i in range(n_pursuers)}
    obs, _ = env.reset()
    ep_rew = 0.0; ep_count = 0

    for _ in range(ROLLOUT_STEPS):
        actions = {}; gs_tokens = {}; vals = {}
        for i in range(n_pursuers):
            gs_i = build_global_tokens(env, i)
            gs_tokens[i] = gs_i
            vals[i] = critic(torch.as_tensor(gs_i, dtype=torch.float32).unsqueeze(0).to(device)).item()

        for i in range(n_pursuers):
            start = i * OBS_PER_AGENT
            o_t = torch.as_tensor(obs[start:start+OBS_PER_AGENT], dtype=torch.float32).unsqueeze(0).to(device)
            loc, scale = actors[i](o_t)
            dist = Independent(Normal(loc, scale), 1)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
            k = f'p{i}'
            buf[k]['obs'].append(obs[start:start+OBS_PER_AGENT].copy())
            buf[k]['act'].append(act.cpu().squeeze(0).numpy())
            buf[k]['val'].append(vals[i]); buf[k]['logp'].append(logp.item())
            buf[k]['done'].append(0.0); buf[k]['term'].append(1.0)
            buf[k]['gs'].append(gs_tokens[i].copy())
            actions[f'p{i}'] = act.cpu().squeeze(0).numpy()

        concat_act = np.concatenate([actions[f'p{i}'] for i in range(n_pursuers)])
        next_obs, rew, term, trunc, info = env.step(concat_act)
        done = term or trunc; ep_rew += rew
        p_dists = []; t_pos = env.targets[0].aircraft.position_ned
        for i in range(n_pursuers):
            p_dists.append(float(np.linalg.norm(env.pursuers[i].aircraft.position_ned - t_pos)))
        rubber_band = 0.0
        if n_pursuers >= 2:
            spacing = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned - env.pursuers[1].aircraft.position_ned))
            if spacing > 1500.0: rubber_band = -0.01 * (spacing - 1500.0) / REWARD_SCALE
        for i in range(n_pursuers):
            k = f'p{i}'; base_r = rew / REWARD_SCALE / n_pursuers
            if hasattr(env.pursuers[i], 'prev_dist'):
                base_r += 0.5 * (env.pursuers[i].prev_dist - p_dists[i]) / REWARD_SCALE
            env.pursuers[i].prev_dist = p_dists[i]
            if info.get('reason') in ('success','cooperative_success'):
                if p_dists[i] < 1500.0:
                    base_r += (3000.0 if i == info.get('kill_agent_idx', 0) else 2000.0) / REWARD_SCALE
            if n_pursuers >= 2: base_r += rubber_band / n_pursuers
            buf[k]['rew'].append(base_r)
            buf[k]['done'][-1] = float(done)
            buf[k]['term'][-1] = 0.0 if (term and not trunc) else 1.0
        if done: ep_count += 1; next_obs, _ = env.reset()
        obs = next_obs

    final_vals = {}
    for i in range(n_pursuers):
        gs_f = build_global_tokens(env, i)
        final_vals[i] = critic(torch.as_tensor(gs_f, dtype=torch.float32).unsqueeze(0).to(device)).item()

    all_data = []
    for i in range(n_pursuers):
        k = f'p{i}'; n = len(buf[k]['rew'])
        adv = np.zeros(n, dtype=np.float32); ret = np.zeros(n, dtype=np.float32); gae = 0.0
        for t in reversed(range(n)):
            nv = (final_vals[i] if t == n-1 else buf[k]['val'][t+1]) * buf[k]['term'][t]
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


def ppo_update(actors, critic, actor_opts, critic_opt, all_data, device,
               kl_target=KL_TARGET, ent_coef=ENT_COEF):
    """Per-pursuer decoupled PPO: separate surrogate loss + gradient step for each actor."""
    for a in actors: a.train()
    critic.train()
    all_vals, all_rets = [], []
    kl_skips = 0

    for epoch_idx in range(PPO_EPOCHS):
        for i, data in enumerate(all_data):
            actor = actors[i]; actor_opt = actor_opts[i]
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

                if epoch_idx > 0:
                    with torch.no_grad():
                        if ((ratio - 1) - ratio.log()).mean().item() > kl_target:
                            kl_skips += 1; continue

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                if ent_coef > 0: actor_loss = actor_loss - ent_coef * dist.entropy().mean()

                val_pred = critic(b_gs)
                v_clip = b_oldv + torch.clamp(val_pred - b_oldv, -CLIP_EPS, CLIP_EPS)
                critic_loss = 0.5 * torch.max((val_pred - b_ret)**2, (v_clip - b_ret)**2).mean()

                actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM); actor_opt.step()
                critic_opt.zero_grad(); (VF_COEF * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM); critic_opt.step()
                all_vals.append(val_pred.detach()); all_rets.append(b_ret)

    ev = 1.0 - torch.var(torch.cat(all_rets) - torch.cat(all_vals)) / (torch.var(torch.cat(all_rets)) + 1e-8) if all_vals else 0.0
    return float(ev), kl_skips


def train(mode="2v1", total_steps=500000, difficulty=0.0, seed=42,
          load_ckpt=None, load_bc=None, cooperative=False, critic_warmup_steps=100_000):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"dual_{mode}_coop_{ts}_s{seed}" if cooperative else f"dual_{mode}_{ts}_s{seed}"
    log_dir = f"./marl_runs/{run_name}"; os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard"); os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Dual-Actor MAPPO Training (Decoupled P0/P1)")
    print(f"  Mode: {mode}  |  Steps: {total_steps:,}  |  Cooperative: {cooperative}")
    print(f"  EV gate: {EV_UNFREEZE_THRESHOLD}  |  KL target: {KL_TARGET}")
    print(f"  Log: {log_dir}")
    print(f"{'='*60}\n")

    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty,
                       cooperative_mode=cooperative)
    n_pursuers = 2

    # ── Dual actors ────────────────────────────────────────────────────
    actors = [AttentionFormationActor(mate_scale=1.0).to(device) for _ in range(n_pursuers)]
    critic = AttentionCritic().to(device)
    print(f"  Actors: {n_pursuers} x {sum(p.numel() for p in actors[0].parameters()):,} params")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")

    if load_bc:
        bc = torch.load(load_bc, map_location='cpu')
        for a in actors: a.load_state_dict(bc['actor_state']); a.mate_scale = 1.0
        vl = bc.get('val_loss', '?')
        print(f"  BC loaded: {load_bc} (val_loss={vl}) → both actors")

    if load_ckpt:
        ck = torch.load(load_ckpt, map_location='cpu')
        for i, a in enumerate(actors):
            key = f'actor_p{i}' if f'actor_p{i}' in ck else 'actor'
            a.load_state_dict(ck[key])
        critic.load_state_dict(ck['critic'])
        print(f'  Checkpoint loaded: {load_ckpt}')

    actor_opts = [torch.optim.Adam(a.parameters(), lr=ACTOR_LR_WARMUP, eps=1e-5) for a in actors]
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR_WARMUP, eps=1e-5)

    # ── Training loop ──────────────────────────────────────────────────
    total = 0; epoch = 0; rew_win = deque(maxlen=10); best_rew = -float('inf')
    stage = "WARMUP"; last_ev = 0.0; warmup_done = False; ent_coef = ENT_COEF

    while total < total_steps:
        if total >= critic_warmup_steps and not warmup_done:
            if last_ev >= EV_UNFREEZE_THRESHOLD:
                warmup_done = True; stage = "FINE_TUNE"
                for opt in actor_opts: opt.param_groups[0]["lr"] = ACTOR_LR_FINE
                critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE
                print(f"\n[MAPPO] === Actor unfrozen (EV={last_ev:.3f}) ===\n")
            else:
                print(f"\n[MAPPO] Step {total}: EV={last_ev:.3f} < {EV_UNFREEZE_THRESHOLD}, extending warmup\n")

        if not warmup_done:
            for opt in actor_opts: opt.param_groups[0]["lr"] = ACTOR_LR_WARMUP
            for a in actors: a.eval()
        else:
            for a in actors: a.train()

        if cooperative:
            if total < 200_000: env.set_coop_phase(COOP_PHASE_OR)
            else: env.set_coop_phase(COOP_PHASE_AND)

        if warmup_done and total > critic_warmup_steps:
            frac = 1.0 - (total - critic_warmup_steps) / max(total_steps - critic_warmup_steps, 1)
            for opt in actor_opts: opt.param_groups[0]["lr"] = ACTOR_LR_FINE * max(frac, 0.1)
            critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE * max(frac, 0.1)

        # Entropy guard
        if warmup_done and total > critic_warmup_steps:
            with torch.no_grad():
                loc, scale = actors[0](all_data[0]['obs'][:64].to(device))
                action_entropy = Independent(Normal(loc, scale), 1).entropy().mean().item()
            if action_entropy < ENTROPY_COLLAPSE_THRESH:
                ent_coef = min(ent_coef * 1.5, ENT_COEF_MAX)
            elif action_entropy > ENTROPY_COLLAPSE_THRESH * 2.0:
                ent_coef = max(ent_coef * 0.95, ENT_COEF_MIN)

        all_data, avg_rew, n_ep = collect_rollout(env, actors, critic, device)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        ev, kl_skips = ppo_update(actors, critic, actor_opts, critic_opt, all_data, device, ent_coef=ent_coef)
        last_ev = ev

        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            phase = "[OR]" if (cooperative and total < 200_000) else ("[AND]" if cooperative else "")
            print(f"[{run_name}] step={total:>7d}  rew={avg_rew:8.1f}  avg10={avg10:8.1f}  "
                  f"ev={ev:5.3f}  lr_a={actor_opts[0].param_groups[0]['lr']:.1e}  "
                  f"eps={n_ep}  [{stage[:4]}]  {phase}  kl_skip={kl_skips}", flush=True)
            writer.add_scalar("Train/Reward", avg_rew, total)
            writer.add_scalar("Train/Avg10Reward", avg10, total)
            writer.add_scalar("Train/EV", ev, total)
            writer.add_scalar("Train/ActorLR", actor_opts[0].param_groups[0]['lr'], total)

        avg10 = np.mean(rew_win) if rew_win else avg_rew
        if avg10 > best_rew and len(rew_win) >= 5:
            best_rew = avg10
            ck = {"critic": critic.state_dict(), "epoch": epoch, "total_steps": total}
            for i, a in enumerate(actors): ck[f"actor_p{i}"] = a.state_dict()
            torch.save(ck, os.path.join(log_dir, "best_policy.pth"))

        epoch += 1

    writer.close()
    ck = {"critic": critic.state_dict(), "epoch": epoch, "total_steps": total}
    for i, a in enumerate(actors): ck[f"actor_p{i}"] = a.state_dict()
    final_path = os.path.join(log_dir, "final_policy.pth")
    torch.save(ck, final_path)
    print(f"\n[Done] {final_path}  |  Best avg10: {best_rew:.1f}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="2v1")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load", type=str, default=None, help="Resume MAPPO checkpoint")
    parser.add_argument("--load-bc", type=str, default=None, help="BC pretrained weights")
    parser.add_argument("--cooperative", action="store_true")
    parser.add_argument("--warmup", type=int, default=100000)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    _stderr = sys.stderr; sys.stderr = open(os.devnull, 'w')
    try:
        train(args.mode, args.steps, args.difficulty, args.seed,
              load_ckpt=args.load, load_bc=args.load_bc,
              cooperative=args.cooperative, critic_warmup_steps=args.warmup)
    finally:
        sys.stderr = _stderr
