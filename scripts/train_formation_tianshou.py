"""Phase 5: Tianshou MAPPO 2v1 formation training (CTDE, shared weights).

Uses Tianshou 0.5.1 PPOPolicy + MultiAgentPolicyManager for CTDE.
Each pursuer has its own Box(2) action, 33-dim local obs.
Both share the same Actor/Critic weights (parameter sharing).

Usage:
    python scripts/train_formation_tianshou.py
    python scripts/train_formation_tianshou.py --epochs 200 --difficulty 0.0
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

import tianshou as ts
from tianshou.policy import PPOPolicy
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, PettingZooEnv

from src.environment.formation_pettingzoo import FormationPettingZooEnv
from src.models.tianshou_networks import FormationActor, FormationCritic

# ── Hyperparameters ─────────────────────────────────────────────────────
GAMMA = 0.99; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
VF_COEF = 0.5; ENT_COEF = 0.01; MAX_GRAD_NORM = 0.5; LR = 3e-4
BATCH_SIZE = 64; REPEAT_PER_COLLECT = 10; BUFFER_SIZE = 8192; NUM_ENVS = 2


def make_env(difficulty: float = 0.0):
    def _init():
        return FormationPettingZooEnv(
            num_pursuers=2, difficulty_level=difficulty,
            lock_altitude=True, record_tacview=False)
    return _init


def build_policies(device: torch.device):
    actor = FormationActor(obs_dim=33, act_dim=2, hidden=256).to(device)
    critic = FormationCritic(global_dim=21, hidden=256).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    def _dist(loc_scale):
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=_dist,
        discount_factor=GAMMA, gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM, vf_coef=VF_COEF,
        ent_coef=ENT_COEF, eps_clip=CLIP_EPS,
        deterministic_eval=True,
        action_space=ts.utils.spaces.Box(-1, 1, (2,)),
    )
    return policy, actor, critic


def train(epochs: int = 200, difficulty: float = 0.0, seed: int = 42):
    ts_str = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = f"./marl_runs/formation_tianshou_{ts_str}_s{seed}"
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tianshou] Device: {device}  Epochs: {epochs}  Diff: {difficulty:.2f}")

    # Build envs
    train_envs = SubprocVectorEnv([make_env(difficulty) for _ in range(NUM_ENVS)])
    # PettingZooEnv wrapper needed for MultiAgentPolicyManager
    # We use DummyVectorEnv for test since SubprocVectorEnv with PettingZoo is complex
    test_env = PettingZooEnv(FormationPettingZooEnv(num_pursuers=2, difficulty_level=difficulty))

    # Build policy
    policy, actor, critic = build_policies(device)

    # MultiAgentPolicyManager: same policy for both agents (shared weights)
    manager = MultiAgentPolicyManager([policy, policy], test_env)

    # Collectors
    train_collector = Collector(
        manager, train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(manager, DummyVectorEnv([lambda: test_env]))

    step_count = 0; best_reward = -float("inf")

    for epoch in range(epochs):
        result = train_collector.collect(n_step=8192 // NUM_ENVS, random=False)
        step_count += result["n/st"]

        losses = {}
        for _ in range(REPEAT_PER_COLLECT):
            for batch_idx in range(0, len(train_collector.buffer), BATCH_SIZE):
                batch, _ = train_collector.buffer.sample(BATCH_SIZE)
                if len(batch) == 0: continue
                loss_dict = policy.update(sample_size=0, batch_size=BATCH_SIZE, repeat=1)
                for k, v in (loss_dict or {}).items():
                    losses[k] = losses.get(k, 0) + float(v)

        if epoch % 10 == 0:
            test_result = test_collector.collect(n_episode=5)
            ep_rew = float(np.mean(test_result.get("rews", [0])))
            ep_len = float(np.mean(test_result.get("lens", [0])))
            loss_str = " ".join(f"{k}={v/REPEAT_PER_COLLECT:.3f}" for k,v in list(losses.items())[:3])
            print(f"[MAPPO] e={epoch:4d}  steps={step_count:>7d}  "
                  f"rew={ep_rew:8.1f}  len={ep_len:6.1f}  {loss_str}")
            if ep_rew > best_reward:
                best_reward = ep_rew
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                           os.path.join(log_dir, "best_policy.pth"))

        train_collector.reset_buffer()

    final = os.path.join(log_dir, "final_policy.pth")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, final)
    print(f"[MAPPO] Done. Best rew={best_reward:.1f}. Saved: {final}")


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
