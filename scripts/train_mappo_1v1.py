"""1v1 MAPPO validation — pure PyTorch CTDE with GAE + PPO clip.

Verifies the hand-rolled MAPPO training loop against the known SB3
baseline (~30% success from scratch at 100K steps on diff=0.0).

Bug fixes applied (2026-06-30):
  1. log_std init = 0.0 (std=1.0, not 0.6)
  2. GAE: timeout → bootstrap V(s_T); crash/kill → value=0
  3. Data alignment: obs/act/rew/next_obs strictly paired across episode boundaries
  4. 1v1 mode: Actor and Critic both see 33-dim local obs (no global state)

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
VF_COEF = 0.5; ENT_COEF = 0.02; MAX_GRAD_NORM = 0.5; LR = 3e-4
MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096
LOG_STD_INIT = 0.0  # Fix 1: std=1.0 (was -0.5)


# ═══════════════════════════════════════════════════════════════════════════
#  Networks (1v1: same 33-dim input for both Actor and Critic)
# ═══════════════════════════════════════════════════════════════════════════

class Actor1v1(nn.Module):
    def __init__(self, obs_dim=33, act_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * LOG_STD_INIT)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        feat = self.net(obs)
        loc = torch.tanh(self.mean(feat))
        scale = torch.exp(self.log_std).expand_as(loc)
        return loc, scale


class Critic1v1(nn.Module):
    def __init__(self, obs_dim=33, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Rollout + GAE (with Fix 2 + Fix 3)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_rollout(env, actor, critic, device, n_steps=ROLLOUT_STEPS):
    """Collect trajectory with correct obs/act/rew/next_obs alignment."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    val_list, logp_list, term_list = [], [], []  # term_list: True=bootstrap, False=zero

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
        done = term or trunc
        ep_rew += rew

        rew_list.append(rew)
        done_list.append(1.0 if done else 0.0)
        # Fix 2: timeout/truncation → bootstrap; crash/success → zero
        is_terminal = term and not trunc  # true termination (crash, kill)
        term_list.append(0.0 if is_terminal else 1.0)  # 1=bootstrap, 0=zero-value

        if done:
            ep_count += 1
            next_obs, _ = env.reset()
            # Fix 3: don't overwrite obs until AFTER storing this step's data
            # (next_obs for GAE comes from the NEXT iteration's obs_list)

        obs = next_obs  # Fix 3: s_{t+1} → next iteration's s_t

    # Final value for GAE bootstrap (last s_{t+1})
    o_final = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    final_val = critic(o_final).item()

    # ── GAE computation ──────────────────────────────────────────────
    n = len(rew_list)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_val = final_val * term_list[t]  # Fix 2: bootstrap only if not terminal
            next_adv = 0.0
        else:
            next_val = val_list[t + 1] * term_list[t] + 0.0 * (1.0 - term_list[t])
            next_adv = advantages[t + 1]

        delta = rew_list[t] + GAMMA * next_val * (1.0 - done_list[t]) - val_list[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - done_list[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + val_list[t]

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
#  PPO Update
# ═══════════════════════════════════════════════════════════════════════════

def ppo_update(actor, critic, optim, data, device, epochs=PPO_EPOCHS):
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

            loc, scale = actor(batch_obs)
            dist = Independent(Normal(loc, scale), 1)
            new_logp = dist.log_prob(batch_act).sum(-1)
            ratio = torch.exp(new_logp - batch_logp)

            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()

            val_pred = critic(batch_obs)
            critic_loss = VF_COEF * ((val_pred - batch_ret) ** 2).mean()

            loss = actor_loss - ENT_COEF * entropy + critic_loss

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
            optim.step()
            total_loss += loss.item()

    return total_loss


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(total_steps: int = 100000, difficulty: float = 0.0, seed: int = 42):
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = f"./marl_runs/mappo_1v1_{ts}_s{seed}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MAPPO 1v1] Steps={total_steps:,} diff={difficulty:.2f} seed={seed}")
    print(f"  Fixes: log_std=0, GAE bootstrap, data alignment, 1v1 global=local")

    actor = Actor1v1().to(device); critic = Critic1v1().to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)

    total_steps_done = 0; epoch = 0
    best_reward = -float("inf")
    rew_window = deque(maxlen=10)

    while total_steps_done < total_steps:
        data, avg_rew, n_ep = collect_rollout(env, actor, critic, device, ROLLOUT_STEPS)
        total_steps_done += ROLLOUT_STEPS
        rew_window.append(avg_rew)

        loss = ppo_update(actor, critic, optim, data, device)

        if epoch % 5 == 0:
            avg10 = np.mean(rew_window) if rew_window else avg_rew
            entropy = Independent(Normal(actor(torch.zeros(1, 33).to(device))[0],
                                         actor(torch.zeros(1, 33).to(device))[1]), 1).entropy().sum(-1).mean().item()
            print(f"[MAPPO] step={total_steps_done:>7d}  rew={avg_rew:8.1f}  "
                  f"avg10={avg10:8.1f}  entropy={entropy:.3f}  eps={n_ep}")

            if avg10 > best_reward:
                best_reward = avg10
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                           os.path.join(log_dir, "best_policy.pth"))

        epoch += 1

    final = os.path.join(log_dir, "final_policy.pth")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, final)
    print(f"[MAPPO] Done. Best avg10 reward: {best_reward:.1f}. Saved: {final}")


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
