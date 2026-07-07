"""Incremental BC: fine-tune existing PPO checkpoint on new expert data.

Loads a trained model and runs a few BC epochs on additional expert
trajectories to inject new prior knowledge without destroying existing
skills.  Designed for hot-starting from a Phase 3v3 checkpoint with
extreme-geometry BC data.

Usage:
    python scripts/train_bc_incremental.py \
        --base-model marl_runs/phase2_continuous_0627_2326_s42_bc/phase2_final.zip \
        --data data/expert_extreme/pn_expert_500ep_?????_????.npz \
        --epochs 20 --lr 5e-5
"""

from __future__ import annotations

import argparse, datetime, os, sys, warnings, logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description="Incremental BC fine-tuning")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-split", type=float, default=0.95)
    args = parser.parse_args()

    os.environ.setdefault("JSBSIM_DEBUG", "0")
    warnings.filterwarnings("ignore")
    logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = PPO.load(args.base_model, device=device)

    # Load expert data
    data = np.load(args.data)
    obs = data["observations"]
    acts = data["actions"]
    print(f"Expert data: {obs.shape[0]:,} steps, obs={obs.shape[1]}, act={acts.shape[1]}")

    # Train/val split
    n_train = int(len(obs) * args.train_split)
    indices = np.random.permutation(len(obs))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = TensorDataset(
        torch.tensor(obs[train_idx], dtype=torch.float32),
        torch.tensor(acts[train_idx], dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(obs[val_idx], dtype=torch.float32),
        torch.tensor(acts[val_idx], dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Train
    policy = model.policy
    policy.train()
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")

    print(f"Incremental BC: {args.epochs} epochs, lr={args.lr}")
    for epoch in range(args.epochs):
        policy.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
            distribution = policy.get_distribution(batch_obs)
            pred = distribution.get_actions(deterministic=True)
            loss = loss_fn(pred, batch_act)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(batch_obs)
        train_loss /= n_train

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
                distribution = policy.get_distribution(batch_obs)
                pred = distribution.get_actions(deterministic=True)
                val_loss += loss_fn(pred, batch_act).item() * len(batch_obs)
        val_loss /= len(val_idx)

        if val_loss < best_val:
            best_val = val_loss

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}")

    # Save
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_path = f"./data/expert/bc_hotstart_extreme_{timestamp}.zip"
    model.save(save_path)
    print(f"Saved: {save_path}")
    print(f"Best val MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
