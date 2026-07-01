"""2v1 Collaborative BC Pipeline for Attention Actor.

Step 1: Collect per-pursuer (obs_33, act_2) from SB3 Phase 4.1 (97.3% 2v1).
Step 2: BC-train Attention Actor with mate_scale=1.0 on coordinated data.

Key difference from 1v1 BC: mate features carry real coordination semantics.
The Attention Actor learns from day one that mate proximity demands attention.

Usage:
  python scripts/train_attention_bc_2v1.py --collect --episodes 500
  python scripts/train_attention_bc_2v1.py --train --epochs 80
  python scripts/train_attention_bc_2v1.py --collect --train
"""

from __future__ import annotations

import argparse, os, sys, warnings, logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("JSBSIM_DEBUG", "0")
warnings.filterwarnings("ignore")
logging.getLogger("jsbsim").setLevel(logging.CRITICAL)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.environment.formation_env import FormationEnv
from src.models.attention_actor import AttentionFormationActor

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH = "data/expert/attention_bc_2v1_data.npz"
MODEL_PATH = "data/expert/attention_bc_2v1_pretrained.pth"
SB3_2V1_PATH = "benchmarks/sb3_2v1_97p3/model.zip"


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: 2v1 Collaborative Expert Data Collection
# ═══════════════════════════════════════════════════════════════════════════

def _load_sb3_2v1():
    from stable_baselines3 import PPO
    try:
        return PPO.load(SB3_2V1_PATH, device='cpu')
    except (ValueError, RuntimeError):
        import zipfile, io
        temp_env = FormationEnv(num_pursuers=2, num_targets=1)
        model = PPO("MlpPolicy", temp_env, policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        ), device='cpu')
        with zipfile.ZipFile(SB3_2V1_PATH, 'r') as zf:
            with zf.open('policy.pth') as f:
                policy_state = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=True)
        model.policy.load_state_dict(policy_state, strict=False)
        return model


def collect_2v1_expert(n_episodes: int = 500, difficulty: float = 0.0,
                         min_success_ratio: float = 0.8):
    """Run SB3 Phase 4.1 on FormationEnv(2v1), collect per-pursuer (obs_33, act_2).

    Each env step produces 2 samples: (p0_obs, p0_act) + (p1_obs, p1_act).
    Only successful episodes are kept to build a pure coordinated-tactics database.
    """
    import io as _io
    _stderr = sys.stderr
    sys.stderr = _io.StringIO()

    model = _load_sb3_2v1()
    env = FormationEnv(num_pursuers=2, num_targets=1, difficulty_level=difficulty)

    all_obs_p0 = []; all_act_p0 = []
    all_obs_p1 = []; all_act_p1 = []
    successes = 0; total_samples = 0

    print(f"Collecting 2v1 collaborative expert data: {n_episodes} episodes...")
    ep = 0
    while ep < n_episodes:
        obs_66, _ = env.reset()
        done = False
        ep_obs_p0 = []; ep_act_p0 = []
        ep_obs_p1 = []; ep_act_p1 = []

        while not done:
            action_4, _ = model.predict(obs_66, deterministic=True)
            p0_obs = obs_66[0:33].astype(np.float32)
            p1_obs = obs_66[33:66].astype(np.float32)
            p0_act = action_4[0:2].astype(np.float32)
            p1_act = action_4[2:4].astype(np.float32)

            ep_obs_p0.append(p0_obs); ep_act_p0.append(p0_act)
            ep_obs_p1.append(p1_obs); ep_act_p1.append(p1_act)

            obs_66, rew, term, trunc, info = env.step(action_4)
            done = term or trunc

        is_success = info.get('reason') == 'success'
        if is_success:
            successes += 1
            all_obs_p0.extend(ep_obs_p0); all_act_p0.extend(ep_act_p0)
            all_obs_p1.extend(ep_obs_p1); all_act_p1.extend(ep_act_p1)
            total_samples += len(ep_obs_p0)

        ep += 1
        if ep % 50 == 0:
            sr = successes / ep if ep > 0 else 0
            print(f"  [{ep:>4d}/{n_episodes}]  success_rate={sr:.1%}  "
                  f"samples={total_samples}  kept_only_success")

    # Merge both pursuers into single dataset (both learn the same expert policy)
    all_obs = np.array(all_obs_p0 + all_obs_p1, dtype=np.float32)
    all_act = np.array(all_act_p0 + all_act_p1, dtype=np.float32)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    np.savez_compressed(DATA_PATH, obs=all_obs, actions=all_act)
    print(f"\nSaved: {DATA_PATH}")
    print(f"  Episodes attempted: {n_episodes}  |  Successes: {successes}")
    print(f"  Total samples (both pursuers): {len(all_obs)}")
    print(f"  Obs shape: {all_obs.shape}  |  Act shape: {all_act.shape}")

    sys.stderr = _stderr
    return all_obs, all_act


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: BC Pretraining with mate_scale=1.0
# ═══════════════════════════════════════════════════════════════════════════

def bc_pretrain_2v1(data_path: str = DATA_PATH, epochs: int = 80,
                    batch_size: int = 128, lr: float = 1e-3, device: str = "cpu"):
    """BC-train Attention Actor on 2v1 coordinated data, mate_scale=1.0."""
    print(f"\n{'='*60}")
    print(f"2v1 BC Pretraining — Attention Actor (mate_scale=1.0)")
    print(f"Data:     {data_path}")
    print(f"Epochs:   {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"{'='*60}")

    data = np.load(data_path)
    obs = torch.tensor(data['obs'], dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.float32)
    print(f"  Loaded: {len(obs)} samples  |  obs={list(obs.shape)}  |  act={list(actions.shape)}")

    n_val = max(int(len(obs) * 0.1), 256)
    idx = torch.randperm(len(obs))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    train_obs, train_act = obs[train_idx], actions[train_idx]
    val_obs, val_act = obs[val_idx], actions[val_idx]
    print(f"  Train: {len(train_obs)}  |  Val: {len(val_obs)}")

    train_loader = DataLoader(TensorDataset(train_obs, train_act),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_act),
                            batch_size=batch_size, shuffle=False)

    # mate_scale=1.0: Mate token fully active from day one
    actor = AttentionFormationActor(mate_scale=1.0).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        actor.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
            loc, _ = actor(batch_obs)
            loss = criterion(loc, batch_act)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item() * len(batch_obs)
        train_loss /= len(train_obs)
        scheduler.step()

        actor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
                loc, _ = actor(batch_obs)
                val_loss += criterion(loc, batch_act).item() * len(batch_obs)
        val_loss /= len(val_obs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'actor_state': actor.state_dict(), 'val_loss': val_loss,
                         'epoch': epoch, 'mate_scale': 1.0}, MODEL_PATH)

        if epoch % 10 == 0 or epoch == epochs - 1:
            # Check attention allocation with mate_scale=1.0
            actor.eval()
            with torch.no_grad():
                (_, _), attn = actor(val_obs[:64].to(device), return_attention=True)
                pool = attn['pool_weights'].squeeze(1)
                s2m = attn['attn_weights'][:, 0, 2].mean().item()
            actor.train()
            mark = "*" if val_loss == best_val_loss else " "
            print(f"  epoch {epoch:>3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"S2M_attn={s2m:.3f}  pool_mate={pool[:,2].mean().item():.3f}  {mark}")

    print(f"\n2v1 BC pretraining done. Best val_loss={best_val_loss:.6f}")
    print(f"Model saved: {MODEL_PATH}")
    return MODEL_PATH


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2v1 Collaborative BC Pipeline for Attention Actor")
    parser.add_argument("--collect", action="store_true", help="Step 1: collect 2v1 expert data")
    parser.add_argument("--train", action="store_true", help="Step 2: BC pretrain with mate_scale=1.0")
    parser.add_argument("--episodes", type=int, default=500, help="Expert episodes")
    parser.add_argument("--epochs", type=int, default=80, help="BC training epochs")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.collect and not args.train:
        args.collect = args.train = True

    if args.collect:
        collect_2v1_expert(n_episodes=args.episodes)

    if args.train:
        bc_pretrain_2v1(data_path=DATA_PATH, epochs=args.epochs,
                        batch_size=args.batch, lr=args.lr, device=args.device)
