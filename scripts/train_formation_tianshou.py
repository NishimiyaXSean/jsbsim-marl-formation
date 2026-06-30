"""Phase 5: Tianshou MAPPO 2v1 formation training (CTDE).

Four-step assembly:
  1. PettingZoo ParallelEnv wrapper → dict obs/act spaces
  2. Actor(33→2) + Critic(21→1) pure PyTorch networks
  3. MAPPOPolicy + MultiAgentPolicyManager
  4. Vectorized Collector + training loop

Usage:
    python scripts/train_formation_tianshou.py
    python scripts/train_formation_tianshou.py --epochs 200 --difficulty 0.0
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn

import tianshou as ts
from tianshou.policy import MAPPOPolicy
from tianshou.policy.multiagent import MultiAgentPolicyManager
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from src.environment.formation_pettingzoo import FormationPettingZooEnv
from src.models.tianshou_networks import FormationActor, FormationCritic


# ── Hyperparameters ─────────────────────────────────────────────────────
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3e-4
BATCH_SIZE = 64
REPEAT_PER_COLLECT = 10
BUFFER_SIZE = 8192
NUM_ENVS = 4  # parallel env runners


# ── Environment factory ─────────────────────────────────────────────────

def make_env(difficulty: float = 0.0):
    """Create a single PettingZoo-wrapped FormationEnv."""
    def _init():
        return FormationPettingZooEnv(
            num_pursuers=2, difficulty_level=difficulty,
            lock_altitude=True, record_tacview=False)
    return _init


# ── Preprocess: extract local obs for Actor, global state for Critic ────

def actor_preprocess(obs):
    """Actor sees only local observation."""
    if isinstance(obs, dict):
        return obs.get("obs", obs)
    return obs


def critic_preprocess(obs):
    """Critic sees global state."""
    if isinstance(obs, dict):
        return obs.get("global_state", obs)
    return obs


# ── Policy builder ──────────────────────────────────────────────────────

def build_policies(
    device: torch.device,
) -> Tuple[MAPPOPolicy, MultiAgentPolicyManager]:
    """Build shared MAPPO policy for both pursuers."""

    actor = FormationActor(obs_dim=33, act_dim=2, hidden=256).to(device)
    critic = FormationCritic(global_dim=21, hidden=256).to(device)

    # Wrap for Tianshou: Actor(Net, preprocess_fn), Critic(Net, preprocess_fn)
    actor_net = Net(
        actor, {"obs": 33, "global_state": 21},
        state_shape="hidden", hidden_sizes=[256, 256],
        activation=nn.Tanh, device=device)
    critic_net = Net(
        critic, {"obs": 33, "global_state": 21},
        state_shape="hidden", hidden_sizes=[256, 256],
        activation=nn.Tanh, device=device)

    # Shared optimizer
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR)

    policy = MAPPOPolicy(
        actor=actor_net,
        critic=critic_net,
        optim=optim,
        dist_fn=torch.distributions.Independent,
        discount_factor=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        eps_clip=CLIP_EPS,
        action_space=None,  # inferred from env
        deterministic_eval=True,
    )

    # Multi-agent: "pursuer_0" and "pursuer_1" share the same policy
    manager = MultiAgentPolicyManager(
        {"pursuer_0": policy, "pursuer_1": policy}, policy)
    return policy, manager


# ── Training loop ───────────────────────────────────────────────────────

def train(epochs: int = 200, difficulty: float = 0.0, seed: int = 42):
    ts_str = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.abspath(f"./marl_runs/formation_tianshou_{ts_str}_s{seed}")
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Tianshou MAPPO] Device: {device}")
    print(f"  Epochs: {epochs}, Difficulty: {difficulty:.2f}, Envs: {NUM_ENVS}")
    print(f"  CTDE: shared Actor(33→2) + Critic(21→1)")
    print(f"  Log: {log_dir}")

    # ── Environments ──────────────────────────────────────────────────
    train_envs = SubprocVectorEnv([make_env(difficulty) for _ in range(NUM_ENVS)])
    test_envs = DummyVectorEnv([make_env(difficulty) for _ in range(2)])

    # ── Policies ──────────────────────────────────────────────────────
    policy, manager = build_policies(device)

    # ── Collectors ────────────────────────────────────────────────────
    train_collector = Collector(
        manager, train_envs,
        VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(manager, test_envs, exploration_noise=False)

    # ── Training ──────────────────────────────────────────────────────
    step_count = 0
    best_reward = -float("inf")

    for epoch in range(epochs):
        # Collect
        collect_result = train_collector.collect(
            n_step=8192 // NUM_ENVS,
            random=False)
        step_count += collect_result["n/st"]

        # Update
        losses = policy.update(
            sample_size=0,  # use all data in buffer
            batch_size=BATCH_SIZE,
            repeat=REPEAT_PER_COLLECT,
        )

        # Logging
        if epoch % 5 == 0:
            test_result = test_collector.collect(n_episode=10)
            ep_rew = test_result.get("rews", 0)
            ep_len = test_result.get("lens", 0)

            if isinstance(ep_rew, np.ndarray):
                ep_rew = float(np.mean(ep_rew))
            if isinstance(ep_len, np.ndarray):
                ep_len = float(np.mean(ep_len))

            loss_str = f"loss={losses.get('loss', 0):.2f}" if isinstance(losses, dict) else ""
            print(f"[MAPPO] epoch={epoch:4d}  steps={step_count:>7d}  "
                  f"ep_rew={ep_rew:8.1f}  ep_len={ep_len:6.1f}  {loss_str}")

            if ep_rew > best_reward:
                best_reward = ep_rew
                torch.save(policy.state_dict(), os.path.join(log_dir, "best_policy.pth"))

        # Reset buffer after each collect-update cycle
        train_collector.reset_buffer()

    # ── Save ──────────────────────────────────────────────────────────
    final_path = os.path.join(log_dir, "final_policy.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"[MAPPO] Training complete. Best reward: {best_reward:.1f}")
    print(f"  Final policy: {final_path}")


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
