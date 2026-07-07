"""Phase 5: Pure PyTorch MAPPO 2v1 CTDE training (no framework deps).

Uses the existing PettingZoo wrapper + Actor/Critic networks with a
hand-rolled MAPPO training loop.  This avoids Tianshou/Ray compatibility
issues while providing full CTDE control.

MAPPO with shared weights:
  - Both pursuers use the same Actor (33→2) and Critic (21→1)
  - GAE advantage estimation per agent
  - Clipped PPO objective with entropy bonus
  - Centralized Critic sees global state; Actor sees local obs

Usage:
    python scripts/train_formation_mappo_torch.py
    python scripts/train_formation_mappo_torch.py --epochs 200 --difficulty 0.0
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging, pickle
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.environment.formation_pettingzoo import FormationPettingZooEnv
from src.models.tianshou_networks import FormationActor, FormationCritic

# ── Hyperparameters ─────────────────────────────────────────────────────
GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.01; MAX_GRAD_NORM = 0.5
LR = 3e-4; MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10
ROLLOUT_STEPS = 2048; NUM_ENVS = 2  # parallel envs
AGENTS = ["pursuer_0", "pursuer_1"]


def make_env(difficulty=0.0):
    def _init():
        return FormationPettingZooEnv(num_pursuers=2, difficulty_level=difficulty,
                                       lock_altitude=True)
    return [_init() for _ in range(NUM_ENVS)]


def compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    advantages = []; gae = 0.0
    next_val = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        next_val = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def run_episode(envs, actor, critic, device, max_steps=ROLLOUT_STEPS):
    """Collect rollout data from NUM_ENVS parallel envs."""
    data = {aid: {"obs": [], "act": [], "rew": [], "val": [], "done": [],
                   "global": [], "log_prob": []} for aid in AGENTS}

    obs_list = [env.reset()[0] for env in envs]
    done_flags = [False] * NUM_ENVS
    total_rew = 0.0

    for step in range(max_steps // NUM_ENVS):
        if all(done_flags): break

        for env_idx, env in enumerate(envs):
            if done_flags[env_idx]:
                continue

            ob = obs_list[env_idx]

            for aid in AGENTS:
                o = torch.as_tensor(ob[aid], dtype=torch.float32).unsqueeze(0).to(device)
                gs = torch.as_tensor(
                    env._last_info[aid].get("global_state", np.zeros(21)),
                    dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    (loc, scale), _ = actor({"obs": o})
                    dist = Independent(Normal(loc, scale), 1)
                    act = dist.sample()
                    log_prob = dist.log_prob(act).sum(-1)
                    val = critic({"global_state": gs})

                data[aid]["obs"].append(o.cpu().squeeze(0))
                data[aid]["act"].append(act.cpu().squeeze(0))
                data[aid]["val"].append(val.item())
                data[aid]["log_prob"].append(log_prob.item())
                data[aid]["global"].append(gs.cpu().squeeze(0))

            # Step all agents at once
            actions = {aid: data[aid]["act"][-1].numpy() for aid in AGENTS}
            next_ob, rews, terms, truncs, infos = env.step(actions)

            done = terms[AGENTS[0]] or truncs[AGENTS[0]]
            for aid in AGENTS:
                data[aid]["rew"].append(rews[aid])
                data[aid]["done"].append(1.0 if done else 0.0)
            total_rew += sum(rews.values()) / len(AGENTS)

            obs_list[env_idx] = next_ob
            done_flags[env_idx] = done
            if done:
                obs_list[env_idx] = env.reset()[0]
                done_flags[env_idx] = False

    return data, total_rew / max(step, 1)


def ppo_update(actor, critic, optim, data, device):
    """Single PPO epoch over collected rollout data."""
    actor.train(); critic.train()
    total_loss = 0.0

    for aid in AGENTS:
        agent_data = data[aid]
        if len(agent_data["obs"]) == 0: continue

        obs = torch.stack(agent_data["obs"]).to(device)
        acts = torch.stack(agent_data["act"]).to(device)
        old_logp = torch.tensor(agent_data["log_prob"], dtype=torch.float32).to(device)
        rews = agent_data["rew"]
        vals = agent_data["val"]
        dones = agent_data["done"]
        gs = torch.stack(agent_data["global"]).to(device)

        # Compute GAE
        adv, ret = compute_gae(rews, vals, dones, GAMMA, GAE_LAMBDA)
        adv = adv.to(device); ret = ret.to(device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        indices = torch.randperm(n)

        for start in range(0, n, MINI_BATCH_SIZE):
            idx = indices[start:start + MINI_BATCH_SIZE]
            if len(idx) == 0: continue

            batch_obs = obs[idx]; batch_act = acts[idx]
            batch_adv = adv[idx]; batch_ret = ret[idx]
            batch_gs = gs[idx]

            # Actor
            (loc, scale), _ = actor({"obs": batch_obs})
            dist = Independent(Normal(loc, scale), 1)
            new_logp = dist.log_prob(batch_act).sum(-1)
            ratio = torch.exp(new_logp - old_logp[idx])

            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * dist.entropy().sum(-1).mean()

            # Critic
            val_pred = critic({"global_state": batch_gs})
            critic_loss = VF_COEF * ((val_pred - batch_ret) ** 2).mean()

            loss = actor_loss + critic_loss

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
            optim.step()

            total_loss += loss.item()

    return total_loss


def train(epochs: int = 200, difficulty: float = 0.0, seed: int = 42):
    ts_str = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = f"./marl_runs/formation_mappo_torch_{ts_str}_s{seed}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MAPPO Torch] Device: {device}  Epochs: {epochs}  Diff: {difficulty:.2f}")
    print(f"  Architecture: CTDE — shared Actor(33→2) + Critic(21→1)")
    print(f"  Rollout: {ROLLOUT_STEPS} steps × {NUM_ENVS} envs, PPO epochs={PPO_EPOCHS}")

    actor = FormationActor(obs_dim=33, act_dim=2).to(device)
    critic = FormationCritic(global_dim=21).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    envs = [FormationPettingZooEnv(num_pursuers=2, difficulty_level=difficulty) for _ in range(NUM_ENVS)]
    [env.reset() for env in envs]

    reward_history = []
    best_reward = -float("inf")

    for epoch in range(epochs):
        data, avg_rew = run_episode(envs, actor, critic, device)
        reward_history.append(avg_rew)

        for _ in range(PPO_EPOCHS):
            ppo_update(actor, critic, optim, data, device)

        if epoch % 10 == 0:
            recent = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            print(f"[MAPPO] epoch={epoch:4d}  rew={avg_rew:8.1f}  avg10={recent:8.1f}")

            if recent > best_reward:
                best_reward = recent
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                           os.path.join(log_dir, "best_policy.pth"))

    final = os.path.join(log_dir, "final_policy.pth")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, final)
    print(f"[MAPPO] Done. Best avg rew={best_reward:.1f}. Saved: {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(epochs=args.epochs, difficulty=args.difficulty, seed=args.seed)
