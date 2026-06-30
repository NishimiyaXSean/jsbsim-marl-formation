"""2v1 MAPPO CTDE — shared Actor, centralized Critic, SB3 Phase 4.1 hot-start.

Two pursuers share a single Actor(33→2). Critic sees 21-dim global state.
Hot-started from SB3 Phase 4.1 weights via symmetric tiling.

Usage:
    python scripts/train_mappo_2v1.py
    python scripts/train_mappo_2v1.py --steps 200000 --sb3 marl_runs/formation_2v1_0629_1721_s42/formation_2v1_final.zip
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

# ── Hyperparameters ─────────────────────────────────────────────────────
GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.0; MAX_GRAD_NORM = 0.5
ACTOR_LR_WARMUP = 0.0; ACTOR_LR_FINE = 1e-5
CRITIC_LR_WARMUP = 5e-4; CRITIC_LR_FINE = 1e-4
WARMUP_STEPS = 150_000  # extended: let Critic fully converge
EV_UNFREEZE_THRESHOLD = 0.6  # only unfreeze Actor when Critic is confident
KL_TARGET = 0.015
MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096
REWARD_SCALE = 100.0
OBS_PER_AGENT = 33; GLOBAL_DIM = 21; ACT_DIM = 2; N_PURSUERS = 2

# ═══════════════════════════════════════════════════════════════════════════
#  Networks (same as 1v1, shared across pursuers)
# ═══════════════════════════════════════════════════════════════════════════

def ortho_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)

class Actor2v1(nn.Module):
    def __init__(self, obs_dim=33, act_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        ortho_init(self.mean, gain=0.01)

    def forward(self, obs):
        if isinstance(obs, np.ndarray): obs = torch.as_tensor(obs, dtype=torch.float32)
        loc = torch.tanh(self.mean(self.net(obs)))
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale

class Critic2v1(nn.Module):
    def __init__(self, global_dim=21, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(global_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.v_out = nn.Linear(hidden, 1)
        self.net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        ortho_init(self.v_out, gain=1.0)

    def forward(self, obs):
        if isinstance(obs, np.ndarray): obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        return self.v_out(self.net(obs)).squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════════
#  SB3 Weight Tiling (Phase 4.1 → 2v1 MAPPO)
# ═══════════════════════════════════════════════════════════════════════════

def load_sb3_2v1(actor, sb3_path):
    """Tile SB3 [256,66]→[256,33] and [4,256]→[2,256]."""
    import zipfile, io
    with zipfile.ZipFile(sb3_path, 'r') as zf:
        with zf.open('policy.pth') as f:
            src = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=True)

    w = src['mlp_extractor.policy_net.0.weight']  # [256, 66]
    actor.net[0].weight.data.copy_(w[:, :33])  # pursuer 0 features
    actor.net[0].bias.data.copy_(src['mlp_extractor.policy_net.0.bias'])
    actor.net[2].weight.data.copy_(src['mlp_extractor.policy_net.2.weight'])
    actor.net[2].bias.data.copy_(src['mlp_extractor.policy_net.2.bias'])

    aw = src['action_net.weight']  # [4, 256]
    ab = src['action_net.bias']    # [4]
    actor.mean.weight.data.copy_(aw[:2])   # pursuer 0 actions
    actor.mean.bias.data.copy_(ab[:2])
    print(f'  Loaded SB3 Phase 4.1 Actor (tiled from [256,66]/[4,256])')
    return actor

# ═══════════════════════════════════════════════════════════════════════════
#  Rollout + GAE (2-agent version)
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
def collect_rollout(env, actor, critic, device, n_steps=ROLLOUT_STEPS):
    buf = {f'p{i}': {'obs':[],'act':[],'rew':[],'val':[],'logp':[],
                      'done':[],'term':[],'gs':[]} for i in range(N_PURSUERS)}
    obs, _ = env.reset()
    ep_rew = 0.0; ep_count = 0

    for _ in range(n_steps):
        actions = {}
        gs = build_global_state(env)
        gs_t = torch.as_tensor(gs, dtype=torch.float32).unsqueeze(0).to(device)
        val = critic(gs_t).item()

        for i in range(N_PURSUERS):
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

        concat_act = np.concatenate([actions[f'p{i}'] for i in range(N_PURSUERS)])
        next_obs, rew, term, trunc, info = env.step(concat_act)
        done = term or trunc; ep_rew += rew

        # Per-pursuer distances + rubber band
        p_dists = []
        t_pos = env.targets[0].aircraft.position_ned
        for i in range(N_PURSUERS):
            p_dists.append(float(np.linalg.norm(env.pursuers[i].aircraft.position_ned - t_pos)))

        # Rubber band penalty: spacing > 1500m
        if N_PURSUERS >= 2:
            spacing = float(np.linalg.norm(env.pursuers[0].aircraft.position_ned -
                                            env.pursuers[1].aircraft.position_ned))
            rubber_band = 0.0
            if spacing > 1500.0:
                rubber_band = -0.01 * (spacing - 1500.0) / REWARD_SCALE

        for i in range(N_PURSUERS):
            k = f'p{i}'
            # Base: split env reward + individual progress (proportional to distance)
            base_r = rew / REWARD_SCALE / N_PURSUERS
            # Individual delta-distance (positive = closing)
            if hasattr(env.pursuers[i], 'prev_dist'):
                delta = env.pursuers[i].prev_dist - p_dists[i]
                base_r += 0.5 * delta / REWARD_SCALE  # individual progress bonus
            env.pursuers[i].prev_dist = p_dists[i]

            # Engagement zone gate: only nearby pursuers get kill bonus
            if info.get('reason') == 'success':
                kd = p_dists[i]
                if kd < 1500.0:
                    base_r += (3000.0 if i == info.get('kill_agent_idx', 0) else 2000.0) / REWARD_SCALE

            # Rubber band (shared penalty)
            if N_PURSUERS >= 2:
                base_r += rubber_band / N_PURSUERS

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
    for i in range(N_PURSUERS):
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

                if _ > 0:
                    with torch.no_grad():
                        if ((ratio - 1) - ratio.log()).mean().item() > KL_TARGET:
                            continue

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

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

def train(total_steps=200000, difficulty=0.0, seed=42, sb3_ckpt=None, load_ckpt=None):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = f"./marl_runs/mappo_2v1_{ts}_s{seed}"; os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAPPO 2v1] Steps={total_steps:,} diff={difficulty:.2f}")

    actor = Actor2v1().to(device); critic = Critic2v1().to(device)
    if sb3_ckpt: actor = load_sb3_2v1(actor, sb3_ckpt)
    if load_ckpt:
        ck = torch.load(load_ckpt, map_location='cpu')
        actor.load_state_dict(ck['actor']); critic.load_state_dict(ck['critic'])
        print(f'  Loaded MAPPO checkpoint: {load_ckpt}')
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR_WARMUP, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR_WARMUP, eps=1e-5)
    warmup_done = False

    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)
    total = 0; epoch = 0; rew_win = deque(maxlen=10)

    last_ev = 0.0
    while total < total_steps:
        if total >= WARMUP_STEPS and not warmup_done:
            # EV gating: only unfreeze if Critic is confident
            if last_ev >= EV_UNFREEZE_THRESHOLD:
                actor_opt.param_groups[0]["lr"] = ACTOR_LR_FINE
                critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE
                warmup_done = True
                print(f"\n[MAPPO] === Actor unfrozen (EV={last_ev:.3f} >= {EV_UNFREEZE_THRESHOLD}) ===\n")
            else:
                print(f"\n[MAPPO] === Step {total}: EV={last_ev:.3f} < {EV_UNFREEZE_THRESHOLD}, extending warmup ===\n")
        if warmup_done:
            frac = 1.0 - (total - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
            actor_opt.param_groups[0]["lr"] = ACTOR_LR_FINE * max(frac, 0.1)
            critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE * max(frac, 0.1)

        all_data, avg_rew, n_ep = collect_rollout(env, actor, critic, device)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        ev = ppo_update(actor, critic, actor_opt, critic_opt, all_data, device); last_ev = ev

        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            print(f"[MAPPO] step={total:>7d}  rew={avg_rew:8.1f}  "
                  f"avg10={avg10:8.1f}  ev={ev:5.3f}  lr_a={actor_opt.param_groups[0]['lr']:.1e}  eps={n_ep}")
        epoch += 1

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(log_dir, "final_policy.pth"))
    print(f"[MAPPO] Done. {log_dir}/final_policy.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sb3", type=str, default=None)
    parser.add_argument("--load", type=str, default=None, help="Resume from MAPPO checkpoint .pth")
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(args.steps, args.difficulty, args.seed, args.sb3, args.load)
