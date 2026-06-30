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
VF_COEF = 0.5; ENT_COEF_INIT = 0.0; MAX_GRAD_NORM = 0.5   # P5: ent=0, SB3 weights don't need exploration
ACTOR_LR_WARMUP = 0.0; ACTOR_LR_FINE = 1e-5  # P6: frozen warmup, then gentle
CRITIC_LR_WARMUP = 5e-4; CRITIC_LR_FINE = 1e-4
WARMUP_STEPS = 50_000  # Critic-only phase
KL_TARGET = 0.015  # P6: early stopping if KL exceeds this
MINI_BATCH_SIZE = 64; PPO_EPOCHS = 10; ROLLOUT_STEPS = 4096
REWARD_SCALE = 100.0  # P0: scale rewards to ~[-1,1] for Critic stability


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

        rew_list.append(rew / REWARD_SCALE)  # P0: scale to ~[-1,1]
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
        "old_val": torch.tensor(val_list, dtype=torch.float32),  # P1: Value Clipping
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
    all_vals = []; all_rets = []

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
            batch_old_val = data["old_val"][idx].to(device)

            # P3: per-minibatch advantage normalization
            mb_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

            # ── Actor ──────────────────────────────────────────────
            loc, scale = actor(batch_obs)
            dist = Independent(Normal(loc, scale), 1)
            new_logp = dist.log_prob(batch_act).sum(-1)
            ratio = torch.exp(new_logp - batch_logp)

            # P6: KL early stopping — protect SB3 weights from drift
            with torch.no_grad():
                approx_kl = ((ratio - 1) - ratio.log()).mean().item()
            if approx_kl > KL_TARGET:
                break  # skip remaining minibatches in this epoch

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(-1).mean()

            actor_opt.zero_grad()
            (actor_loss - ent_coef * entropy).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            actor_opt.step()

            # ── Critic with Value Clipping (P1) ───────────────────
            val_pred = critic(batch_obs)
            v_clipped = batch_old_val + torch.clamp(val_pred - batch_old_val,
                                                     -CLIP_EPS, CLIP_EPS)
            v_loss_1 = (val_pred - batch_ret) ** 2
            v_loss_2 = (v_clipped - batch_ret) ** 2
            critic_loss = 0.5 * torch.max(v_loss_1, v_loss_2).mean()

            critic_opt.zero_grad()
            (VF_COEF * critic_loss).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            critic_opt.step()

            total_loss += (actor_loss + critic_loss).item()
            all_vals.append(val_pred.detach())
            all_rets.append(batch_ret)

    # P0: Explained Variance
    all_v = torch.cat(all_vals)
    all_r = torch.cat(all_rets)
    ev = 1.0 - torch.var(all_r - all_v) / (torch.var(all_r) + 1e-8)
    return total_loss, float(ev)


# ═══════════════════════════════════════════════════════════════════════════
#  Training loop (with P4: linear annealing)
# ═══════════════════════════════════════════════════════════════════════════

def load_sb3_weights(actor, critic, sb3_path: str):
    """Hot-start MAPPO networks from Phase 3.6 SB3 checkpoint."""
    import zipfile, io
    from stable_baselines3 import PPO
    sb3 = PPO.load(sb3_path, device='cpu')
    src = sb3.policy.state_dict()

    # Actor: pad first layer [256,27] → [256,33] (extra 6 cols = mate features, always 0)
    w = src['mlp_extractor.policy_net.0.weight']  # [256, 27]
    actor.net[0].weight.data[:, :27] = w
    actor.net[0].bias.data.copy_(src['mlp_extractor.policy_net.0.bias'])
    actor.net[2].weight.data.copy_(src['mlp_extractor.policy_net.2.weight'])
    actor.net[2].bias.data.copy_(src['mlp_extractor.policy_net.2.bias'])
    actor.mean.weight.data.copy_(src['action_net.weight'])
    actor.mean.bias.data.copy_(src['action_net.bias'])
    # log_std: SB3 uses state-dependent (not loaded), keep our init

    # Critic: same padding
    w_v = src['mlp_extractor.value_net.0.weight']  # [256, 27]
    critic.net[0].weight.data[:, :27] = w_v
    critic.net[0].bias.data.copy_(src['mlp_extractor.value_net.0.bias'])
    critic.net[2].weight.data.copy_(src['mlp_extractor.value_net.2.weight'])
    critic.net[2].bias.data.copy_(src['mlp_extractor.value_net.2.bias'])
    critic.v_out.weight.data.copy_(src['value_net.weight'])
    critic.v_out.bias.data.copy_(src['value_net.bias'])

    print(f'  Loaded Phase 3.6 weights: {sb3_path}')
    return actor, critic


def train(total_steps: int = 200000, difficulty: float = 0.0, seed: int = 42,
          sb3_ckpt: str | None = None):
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
    if sb3_ckpt:
        actor, critic = load_sb3_weights(actor, critic, sb3_ckpt)
    # P2/P6: two-stage LR — warmup (Critic only), then fine-tune
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR_WARMUP, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR_WARMUP, eps=1e-5)
    warmup_done = False

    env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)

    total = 0; epoch = 0; best_rew = -float("inf"); rew_win = deque(maxlen=10)

    while total < total_steps:
        # P6: two-stage LR switch
        if total >= WARMUP_STEPS and not warmup_done:
            actor_opt.param_groups[0]["lr"] = ACTOR_LR_FINE
            critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE
            warmup_done = True
            print(f"\n[MAPPO] === Step {total}: Actor unfrozen (LR={ACTOR_LR_FINE}), "
                  f"Critic LR={CRITIC_LR_FINE} ===\n")

        # P4: linear annealing (only after warmup)
        if warmup_done:
            frac = 1.0 - (total - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
            actor_opt.param_groups[0]["lr"] = ACTOR_LR_FINE * max(frac, 0.1)
            critic_opt.param_groups[0]["lr"] = CRITIC_LR_FINE * max(frac, 0.1)
        ent_coef = ENT_COEF_INIT  # P5: no entropy needed with SB3 weights

        data, avg_rew, n_ep = collect_rollout(env, actor, critic, device, ROLLOUT_STEPS)
        total += ROLLOUT_STEPS; rew_win.append(avg_rew)
        _, ev = ppo_update(actor, critic, actor_opt, critic_opt, data, device, ent_coef)

        if epoch % 5 == 0:
            avg10 = np.mean(rew_win) if rew_win else avg_rew
            with torch.no_grad():
                loc, scale = actor(torch.zeros(1, 33).to(device))
                ent = Independent(Normal(loc, scale), 1).entropy().sum(-1).mean().item()
            print(f"[MAPPO] step={total:>7d}  rew={avg_rew:8.1f}  "
                  f"avg10={avg10:8.1f}  ev={ev:5.3f}  ent={ent:.3f}  "
                  f"lr_a={actor_opt.param_groups[0]['lr']:.1e}  eps={n_ep}")
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
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sb3", type=str, default=None,
                        help="Path to SB3 Phase 3.6 checkpoint for hot-start")
    args = parser.parse_args()
    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)
    train(total_steps=args.steps, difficulty=args.difficulty, seed=args.seed,
          sb3_ckpt=args.sb3)
