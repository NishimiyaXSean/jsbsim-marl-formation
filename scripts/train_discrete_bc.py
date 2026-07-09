"""Discrete Behavioral Cloning — discretize continuous expert data & pretrain.

Route B from the Continuous→Discrete Migration Plan:
  1. Load existing continuous expert data (SB3 97.3% trajectories)
  2. Discretize actions via argmin projection onto MultiDiscrete([5,3]) grid
  3. Train Self-Attention Actor with CrossEntropyLoss on discrete labels
  4. Save weights compatible with RLlib BC loader format

Usage:
    conda activate marl_env
    python scripts/train_discrete_bc.py --data data/expert/attention_bc_2v1_filtered.npz \
        --epochs 30 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.attention_actor import AttentionFormationActor, SELF_DIM, TARGET_DIM, MATE_DIM
from src.environment.formation_rllib_env import TURN_RATES, SPEEDS, N_TURN, N_SPEED

# Physical action grid for discretization
TURN_GRID = np.array(TURN_RATES, dtype=np.float32)   # [5]
SPEED_GRID = np.array(SPEEDS, dtype=np.float32)       # [3]


# ═══════════════════════════════════════════════════════════════════════════════
#  Discretization: continuous [-1,1]² → discrete (turn_idx, speed_idx)
# ═══════════════════════════════════════════════════════════════════════════════

def discretize_actions(cont_actions: np.ndarray) -> np.ndarray:
    """Convert continuous SB3 actions [-1,1]² to MultiDiscrete labels.

    SB3 continuous action mapping (from formation_env.py):
      cmd_turn_rate = a[0] * 15.0   → [-15, +15] deg/s
      cmd_speed     = 250 + a[1]*100 → [150, 350] m/s

    Args:
        cont_actions: [N, 2] in [-1, 1] range

    Returns:
        discrete_labels: [N, 2] with turn_idx ∈ [0,4], speed_idx ∈ [0,2]
    """
    # Convert to physical units
    turn_physical = cont_actions[:, 0] * 15.0        # [N] in [-15, 15]
    speed_physical = 250.0 + cont_actions[:, 1] * 100.0  # [N] in [150, 350]

    # Argmin projection onto discrete grid
    turn_idx = np.argmin(np.abs(turn_physical[:, None] - TURN_GRID[None, :]), axis=1)
    speed_idx = np.argmin(np.abs(speed_physical[:, None] - SPEED_GRID[None, :]), axis=1)

    discrete = np.stack([turn_idx, speed_idx], axis=1).astype(np.int64)
    return discrete


def print_discretization_stats(cont_actions: np.ndarray, discrete: np.ndarray):
    """Print mapping statistics for verification."""
    turn_idx = discrete[:, 0]
    speed_idx = discrete[:, 1]

    print("Discretization statistics:")
    print(f"  Turn distribution:  ", end="")
    for i in range(N_TURN):
        print(f"bin{i}({TURN_RATES[i]:+.0f}°/s)={(turn_idx==i).sum()/len(turn_idx)*100:.1f}%  ", end="")
    print()
    print(f"  Speed distribution: ", end="")
    for i in range(N_SPEED):
        print(f"bin{i}({SPEEDS[i]:.0f}m/s)={(speed_idx==i).sum()/len(speed_idx)*100:.1f}%  ", end="")
    print()
    print(f"  Total samples: {len(cont_actions)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Discrete BC Model (AttentionFormationActor + categorical heads)
# ═══════════════════════════════════════════════════════════════════════════════

class DiscreteBCActor(nn.Module):
    """Self-Attention Actor for discrete BC pretraining.

    Mirrors RLlibAttentionActor's actor structure:
      AttentionFormationActor (feature extractor)
        → forward_features() → [B, 256]
        → turn_head: Linear(256, 5)
        → speed_head: Linear(256, 3)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_hidden: int = 256):
        super().__init__()
        self.backbone = AttentionFormationActor(
            obs_dim=33, act_dim=2, d_model=d_model,
            n_heads=n_heads, mlp_hidden=mlp_hidden,
        )
        self.turn_head = nn.Linear(mlp_hidden, N_TURN)
        self.speed_head = nn.Linear(mlp_hidden, N_SPEED)

        # Init heads
        nn.init.orthogonal_(self.turn_head.weight, gain=0.01)
        nn.init.constant_(self.turn_head.bias, 0.0)
        nn.init.orthogonal_(self.speed_head.weight, gain=0.01)
        nn.init.constant_(self.speed_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """Return (turn_logits, speed_logits)."""
        feat = self.backbone.forward_features(obs)  # [B, 256]
        return self.turn_head(feat), self.speed_head(feat)

    def get_rllib_state_dict(self) -> dict:
        """Export state dict in RLlib-compatible format.

        RLlib's load_bc_weights maps: bc_key → "actor.{bc_key}"
        So we save backbone params with "actor_state" and head params
        with "actor.turn_head.*" / "actor.speed_head.*" prefixes.
        """
        export = {}
        # Backbone params → "actor_state.xxx" (matches existing BC loader)
        backbone_sd = self.backbone.state_dict()
        export["actor_state"] = backbone_sd

        # Head params → "actor.turn_head.xxx" etc.
        # But RLlib maps them as: rllib_key = f"actor.{bc_key}"
        # So we need to match whatever the RLlibAttentionActor expects
        # For the turn/speed heads, they're top-level in RLlibAttentionActor,
        # so the mapping would be: "turn_head.weight" → "actor.turn_head.weight"
        # Actually looking at load_bc_weights: bc_key → "actor.{bc_key}"
        # So we save "turn_head.xxx" and it maps to "actor.turn_head.xxx" ✓
        export["turn_head.weight"] = self.turn_head.weight.data.clone()
        export["turn_head.bias"] = self.turn_head.bias.data.clone()
        export["speed_head.weight"] = self.speed_head.weight.data.clone()
        export["speed_head.bias"] = self.speed_head.bias.data.clone()

        export["val_loss"] = 0.0
        export["epoch"] = 0
        return export


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_bc(
    data_path: str,
    output_path: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    val_split: float = 0.1,
):
    # ── Load & discretize data ────────────────────────────────────────────
    print(f"Loading: {data_path}")
    raw = np.load(data_path)
    obs = raw["obs"].astype(np.float32)          # [N, 33]
    cont_act = raw["actions"].astype(np.float32)  # [N, 2]

    discrete_labels = discretize_actions(cont_act)
    print_discretization_stats(cont_act, discrete_labels)

    # ── Train/val split ───────────────────────────────────────────────────
    N = len(obs)
    idx = np.random.permutation(N)
    n_val = int(N * val_split)
    train_idx = idx[n_val:]
    val_idx = idx[:n_val]

    X_train = torch.tensor(obs[train_idx])
    Y_train = torch.tensor(discrete_labels[train_idx])
    X_val = torch.tensor(obs[val_idx])
    Y_val = torch.tensor(discrete_labels[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val),
                            batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DiscreteBCActor().to(device_t)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, device={device_t}")

    # Separate learning rates: heads need faster convergence than backbone
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": model.turn_head.parameters(), "lr": lr * 10},
        {"params": model.speed_head.parameters(), "lr": lr * 10},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            turn_logits, speed_logits = model(xb)
            loss = (ce_loss(turn_logits, yb[:, 0]) +
                    ce_loss(speed_logits, yb[:, 1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)

        train_loss = train_loss_sum / len(train_idx)
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_turn_acc = 0
        val_speed_acc = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device_t), yb.to(device_t)
                turn_logits, speed_logits = model(xb)
                loss = (ce_loss(turn_logits, yb[:, 0]) +
                        ce_loss(speed_logits, yb[:, 1]))
                val_loss_sum += loss.item() * len(xb)
                val_turn_acc += (turn_logits.argmax(1) == yb[:, 0]).sum().item()
                val_speed_acc += (speed_logits.argmax(1) == yb[:, 1]).sum().item()

        val_loss = val_loss_sum / len(val_idx)
        val_turn_acc /= len(val_idx)
        val_speed_acc /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_sd = model.get_rllib_state_dict()
            best_sd["val_loss"] = val_loss
            best_sd["epoch"] = epoch + 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"turn_acc={val_turn_acc*100:.1f}%  speed_acc={val_speed_acc*100:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(best_sd, output_path)
    print(f"\nSaved: {output_path}")
    print(f"  val_loss={best_val_loss:.6f}  "
          f"turn_acc={val_turn_acc*100:.1f}%  speed_acc={val_speed_acc*100:.1f}%")
    return best_sd


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discretize continuous expert data and train discrete BC actor")
    parser.add_argument("--data", type=str,
                        default="data/expert/attention_bc_2v1_filtered.npz",
                        help="Path to continuous expert .npz file")
    parser.add_argument("--output", type=str,
                        default="data/expert/discrete_attention_bc.pth",
                        help="Output path for BC model weights")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    train_bc(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
