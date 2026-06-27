"""Behavioural Cloning pretraining for Phase 2 continuous pursuit.

Loads PN expert trajectories (obs, actions) from a .npz file and
supervises the PPO actor network to imitate them via MSE loss.
The pretrained weights become the starting point for PPO fine-tuning.

Usage:
    python scripts/train_bc_pretrain.py
    python scripts/train_bc_pretrain.py --data data/expert/pn_expert_500ep_38344steps_0627_1548.npz
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO

from src.environment.continuous_pursuit_env import ContinuousPursuitEnv
from src.environment.ablation_wrappers import LeadPursuitRewardWrapper

# ── BC hyperparameters ─────────────────────────────────────────────────
BC_EPOCHS = 50
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-4
BC_TRAIN_SPLIT = 0.95        # 95% train, 5% validation


def build_dummy_env():
    """Minimal env for PPO model construction (not used for rollout)."""
    env = ContinuousPursuitEnv(lock_altitude=True, difficulty_level=0.0)
    env = LeadPursuitRewardWrapper(env)
    return env


def load_expert_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (observations, actions) from .npz file."""
    data = np.load(filepath)
    obs = data["observations"]
    acts = data["actions"]
    print(f"Loaded expert data: {obs.shape[0]:,} steps, "
          f"obs={obs.shape[1]} dim, act={acts.shape[1]} dim")
    return obs, acts


def train_bc(model: PPO, obs: np.ndarray, acts: np.ndarray,
             epochs: int, batch_size: int, lr: float, train_split: float,
             device: torch.device):
    """BC pretrain the PPO actor with MSE loss on expert actions."""

    # ── Train/val split ──────────────────────────────────────────────
    n_train = int(len(obs) * train_split)
    indices = np.random.permutation(len(obs))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = TensorDataset(
        torch.tensor(obs[train_idx], dtype=torch.float32),
        torch.tensor(acts[train_idx], dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(obs[val_idx], dtype=torch.float32),
        torch.tensor(acts[val_idx], dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Optimiser ─────────────────────────────────────────────────────
    policy = model.policy
    policy.train()
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")

    print(f"\nBC pretraining: {epochs} epochs, {batch_size} batch, lr={lr}")
    print(f"  Train: {n_train:,} steps  Val: {len(obs) - n_train:,} steps")

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────
        policy.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            # SB3 continuous policy: forward → distribution → deterministic action
            distribution = policy.get_distribution(batch_obs)
            pred_mean = distribution.get_actions(deterministic=True)  # (batch, 2)

            loss = loss_fn(pred_mean, batch_act)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * len(batch_obs)

        train_loss /= n_train

        # ── Validate ──────────────────────────────────────────────────
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs = batch_obs.to(device)
                batch_act = batch_act.to(device)
                distribution = policy.get_distribution(batch_obs)
                pred_mean = distribution.get_actions(deterministic=True)
                val_loss += loss_fn(pred_mean, batch_act).item() * len(batch_obs)
        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"best_val={best_val_loss:.6f}")

    print(f"\nBC complete.  Best val MSE: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="BC pretrain PPO actor on PN expert trajectories")
    parser.add_argument("--data", type=str,
                        default="./data/expert/pn_expert_500ep_38344steps_0627_1548.npz",
                        help="Path to expert .npz file")
    parser.add_argument("--epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BC_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        # Try to find the latest .npz in data/expert/
        expert_dir = Path("./data/expert")
        npz_files = sorted(expert_dir.glob("pn_expert_*.npz"))
        if npz_files:
            data_path = npz_files[-1]
            print(f"Auto-selected latest dataset: {data_path}")
        else:
            raise FileNotFoundError(f"No expert data found at {args.data}")

    obs, acts = load_expert_data(str(data_path))

    # ── Build PPO model (dummy env, never stepped) ────────────────────
    dummy_env = build_dummy_env()
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPO(
        "MlpPolicy",
        dummy_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=args.seed,
        device=device,
    )

    # ── BC pretrain ───────────────────────────────────────────────────
    train_bc(model, obs, acts,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             train_split=BC_TRAIN_SPLIT, device=device)

    # ── Save ──────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_path = f"./data/expert/bc_pretrained_{timestamp}.zip"
    model.save(save_path)
    print(f"BC-pretrained model saved: {save_path}")

    # Quick sanity check: predict on a few samples
    model.policy.eval()
    with torch.no_grad():
        sample_obs = torch.tensor(obs[:5], dtype=torch.float32).to(device)
        distribution = model.policy.get_distribution(sample_obs)
        pred = distribution.get_actions(deterministic=True).cpu().numpy()
        true = acts[:5]
        print(f"\nSanity check (first 5 samples):")
        for i in range(5):
            print(f"  pred={pred[i]}  true={true[i]}  "
                  f"err={np.abs(pred[i] - true[i])}")


if __name__ == "__main__":
    main()
