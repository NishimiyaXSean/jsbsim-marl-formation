"""1v1 MAPPO validation — pure PyTorch CTDE with GAE + PPO clip.

Four SB3-verified optimizations (2026-06-30):
  P1: Orthogonal init (gain=1.414 hidden, 0.01 actor out, 1.0 critic out)
  P2: Asymmetric LR (actor=1e-4, critic=5e-4, Adam eps=1e-5)
  P3: Per-minibatch advantage normalization
  P4: Linear LR + ent_coef annealing over training

Usage:
    python scripts/train_mappo_1v1.py
    python scripts/train_mappo_1v1.py --steps 100000 --difficulty 0.0
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
VF_COEF = 0.5; ENT_COEF_INIT = 0.02; MAX_GRAD_NORM = 0.5
ACTOR_LR = 1e-4; CRITIC_LR = 5e-4  # P2: asymmetric
MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096


# ═══════════════════════════════════════════════════════════════════════════
#  P1: Orthogonal initialization (SB3's secret sauce)
# ═══════════════════════════════════════════════════════════════════════════

def ortho_init(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Actor1v1(nn.Module):
    def __init__(self, obs_dim=33, act_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # P1: orthogonal init
        self.net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        ortho_init(self.mean, gain=0.01)  # actor output: tiny gain

    def forward(self, obs):
        if isinstance(obs, np.ndarray): obs = torch.as_tensor(obs, dtype=torch.float32)
        loc = torch.tanh(self.mean(self.net(obs)))
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale


class Critic1v1(nn.Module):
    def __init__(self, obs_dim=33, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.v_out = nn.Linear(hidden, 1)

        # P1: orthogonal init
        self.net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        ortho_init(self.v_out, gain=1.0)  # critic output: normal gain

    def forward(self, obs):
        if isinstance(obs, np.ndarray): obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        return self.v_out(self.net(obs)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout + GAE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_rollout(env, actor, critic, device, n_steps=ROLLOUT_STEPS):
    obs_list, act_list, rew_list, done_list = [], [], [], []
    val_list, logp_list, term_list = [], [], []

    obs, _ = env.reset()
    ep_rew = 0.0; ep_count = 0

    for _ in range(n_steps):
        o_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        loc, scale = actor(o_t)
        dist = Independent(Normal(loc, scale), 1)
        act = dist.sample()
        logp = dist.log_prob(act).sum(-1)
        val = critic(o_t)

        obs_list.append(obs.copy())
        act_list.append(act.cpu().squeeze(0).numpy())
        val_list.append(val.item())
        logp_list.append(logp.item())

        next_obs, rew, term, trunc, info = env.step(act.cpu().squeeze(0).numpy())
        done = term or trunc; ep_rew += rew

        rew_list.append(rew)
        done_list.append(float(done))
        is_terminal = term and not trunc
        term_list.append(0.0 if is_terminal else 1.0)

        if done:
            ep_count += 1
            next_obs, _ = env.reset()
        obs = next_obs

    o_final = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    final_val = critic(o_final).item()

    n = len(rew_list)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(n)):
        next_val = (final_val if t == n - 1 else val_list[t + 1]) * term_list[t]
        delta = rew_list[t] + GAMMA * next_val * (1.0 - done_list[t]) - val_list[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - done_list[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + val_list[t]

    # Global normalization (once, after GAE)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    data = {
        "obs": torch.tensor(np.array(obs_list), dtype=torch.float32),
        "act": torch.tensor(np.array(act_list), dtype=torch.float32),
        "logp": torch.tensor(logp_list, dtype=torch.float32),
        "val": torch.tensor(val_list, dtype=torch.float32),
        "adv": adv_tensor,
        "ret": torch.tensor(returns, dtype=torch.float32),
    }
    return data, ep_rew / max(ep_count, 1), ep_count


# ═══════════════════════════════════════════════════════════════════════════
#  PPO Update (with P2 async optimizers, P3 per-minibatch norm, P4 annealing)
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(actor, critic, actor_opt, critic_opt, data, device,
               ent_coef, epochs=PPO_EPOCHS):
    actor.train(); critic.train()
    n = len(data["obs"])
    total_loss = 0.0

    for _ in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, MINI_BATCH_SIZE):
            idx = indices[start:start + MINI_BATCH_SIZE]
            if len(idx) == 0: continue

            batch_obs = data["obs"][idx].to(device)
            batch_act = data["act"][idx].to(device)
            batch_adv = data["adv"][idx].to(device)
            batch_ret = data["ret"][idx].to(device)
            batch_logp = data["logp"][idx].to(device)

            # P3: per-minibatch advantage normalization
            mb_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

            loc, scale = actor(batch_obs)
            dist = Independent(Normal(loc, scale), 1)
            new_logp = dist.log_prob(batch_act).sum(-1)
            ratio = torch.exp(new_logp - batch_logp)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()

            val_pred = critic(batch_obs)
            critic_loss = VF_COEF * ((val_pred - batch_ret) ** 2).mean()

            actor_opt.zero_grad()
            (actor_loss - ent_coef * entropy).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            critic_opt.step()

            total_loss += (actor_loss + critic_loss).item()

    return total_loss


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop (with P4: linear annealing)
# ═══════════════════════════════════════════════════════════════════════════

def train(total_steps: int = 100000, difficulty: float = 0.0, seed: int = 42):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = f"./marl_runs/mappo_1v1_{ts}_s{seed}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MAPPO 1v1] Steps={total_steps:,} diff={difficulty:.2f}")
    print(f"  P1: Ortho init (gain=1.414/0.01/1.0)")
    print(f"  P2: Asym LR (actor={ACTOR_LR}, critic={CRITIC_LR})")
    print(f"  P3: Per-minibatch adv norm")
    print(f"  P4: Linear LR + ent_coef annealing")

    actor = Actor1v1().to(device); critic = Critic1v1().to(device)
    # P2: separate optimizers with Adam eps=1e-5
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-5)

    env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)

    total = 0; epoch = 0; best_rew = -float("inf"); rew_win = deque(maxlen=10)

    while total < total_steps:
        # P4: linear annealing
        frac = 1.0 - total / total_steps
        actor_opt.param_groups[0]["lr"] = ACTOR_LR * frac
        critic_opt.param_groups[0]["lr"] = CRITIC_LR * frac
        ent_coef = ENT_COEF_INIT * frac

        data, avg_rew, n_ep = collect_rollout(env, actor, critic, device, ROLLOUT_STEPS)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        ppo_update(actor, critic, actor_opt, critic_opt, data, device, ent_coef)

        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            with torch.no_grad():
                loc, scale = actor(torch.zeros(1, 33).to(device))
                ent = Independent(Normal(loc, scale), 1).entropy().sum(-1).mean().item()
            print(f"[MAPPO] step={total:>7d}  rew={avg_rew:8.1f}  "
                  f"avg10={avg10:8.1f}  ent={ent:.3f}  lr_a={actor_opt.param_groups[0]['lr']:.1e}  eps={n_ep}")
            if avg10 > best_rew:
                best_rew = avg10
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                           os.path.join(log_dir, "best_policy.pth"))
        epoch += 1

    final = os.path.join(log_dir, "final_policy.pth")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, final)
    print(f"[MAPPO] Done. Best avg10: {best_rew:.1f}. {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(total_steps=args.steps, difficulty=args.difficulty, seed=args.seed)
