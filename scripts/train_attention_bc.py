"""Step 1+2: Expert data collection + BC pretraining for Attention Actor.

Step 1 — Data Generation:
  Run SB3 Phase 3.6 (83% 1v1) on FormationEnv(1v1), extracting 27-dim subset
  from 33-dim obs to feed the model. Save (obs_33, action_2) pairs.

Step 2 — BC Pretraining:
  Train Attention Actor (mate_scale=0.0) via MSE on action_mean.
  log_std stays at 0.0 for MAPPO exploration headroom.

Usage:
  python scripts/train_attention_bc.py --collect --episodes 500
  python scripts/train_attention_bc.py --train --epochs 50 --batch 128
  python scripts/train_attention_bc.py --collect --train  # both
"""

from __future__ import annotations

import argparse, os, sys, warnings, logging
from collections import deque

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
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_DATA_PATH = "data/expert/attention_bc_data.npz"
BC_MODEL_PATH = "data/expert/attention_bc_pretrained.pth"
SB3_PHASE36_PATH = "marl_runs/phase2_continuous_0629_1447_s42_bc/phase2_final.zip"

# Mapping: FormationEnv 33-dim → ContinuousPursuitEnv 27-dim
# FormationEnv: 0-2:rel_pos, 3-5:own_vel, 6-8:rpy, 9-11:ang_vel, 12:height,
#               13-15:tgt_vel, 16-18:placeholder, 19-21:tac_geo, 22:alpha,
#               23:airspeed, 24:placeholder, 25:los_rate, 26:bearing_err,
#               27-32:mate(not in 27-dim)
# Phase 3.6:   0-2:rel_pos, 3-5:own_vel, 6-8:rpy, 9-11:ang_vel, 12:height,
#               13-15:tgt_vel, 16-18:tgt_ang_vel, 19-21:tac_geo, 22:alpha,
#               23:airspeed, 24:Ps, 25:los_rate, 26:bearing_err
INDICES_33_TO_27 = [
    0,1,2, 3,4,5, 6,7,8, 9,10,11, 12,        # 0-12: same
    13,14,15,                                   # 13-15: tgt_vel (same position)
    # 16-18 in 33-dim = placeholder → map to 16-18 in 27-dim = tgt_ang_vel
    # But FormationEnv doesn't have tgt_ang_vel, so we pass zeros
    # 16,17,18 → (handled in code)
    19,20,21,                                   # 19-21: tac_geo (same position)
    22,23,                                      # 22-23: alpha, airspeed (same)
    # 24 in 33-dim = placeholder → map to 24 in 27-dim = Ps (zero)
    25,26,                                      # 25-26: los_rate, bearing_err (same)
]
# The 33-dim indices map one-to-one to 27-dim indices for most features,
# except dims 16-18 (placeholder→zero) and dim 24 (placeholder→zero)


def obs_33_to_27(obs_33: np.ndarray) -> np.ndarray:
    """Convert FormationEnv 33-dim obs to Phase 3.6 27-dim format.

    Drops mate features (27-32), fills missing features (tgt_ang_vel, Ps) with zeros.
    """
    obs_27 = np.zeros(27, dtype=np.float32)
    # Copy matching features (same indices in both formats)
    for i in range(16):
        obs_27[i] = obs_33[i]
    # obs_27[16:19] = 0.0  (tgt_ang_vel — unavailable in FormationEnv)
    for i_33, i_27 in [(19,19),(20,20),(21,21),(22,22),(23,23)]:
        obs_27[i_27] = obs_33[i_33]
    # obs_27[24] = 0.0  (Ps — unavailable in FormationEnv)
    obs_27[25] = obs_33[25]
    obs_27[26] = obs_33[26]
    return np.clip(obs_27, -1, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Expert Data Collection
# ═══════════════════════════════════════════════════════════════════════════

def load_phase36_model():
    """Load SB3 Phase 3.6 model, handling optimizer version mismatch."""
    from stable_baselines3 import PPO
    try:
        return PPO.load(SB3_PHASE36_PATH, device='cpu')
    except (ValueError, RuntimeError):
        import zipfile, io
        from src.environment.continuous_pursuit_env import ContinuousPursuitEnv
        temp_env = ContinuousPursuitEnv()
        model = PPO("MlpPolicy", temp_env, policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
        ), device='cpu')
        with zipfile.ZipFile(SB3_PHASE36_PATH, 'r') as zf:
            with zf.open('policy.pth') as f:
                policy_state = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=True)
        model.policy.load_state_dict(policy_state, strict=False)
        return model


def collect_expert_data(n_episodes: int = 500, difficulty: float = 0.0):
    """Run Phase 3.6 model on FormationEnv(1v1), collect (obs_33, act_2) pairs."""
    import io as _io
    _stderr = sys.stderr
    sys.stderr = _io.StringIO()

    model = load_phase36_model()
    env = FormationEnv(num_pursuers=1, num_targets=1, difficulty_level=difficulty)

    all_obs = []
    all_actions = []
    total_steps = 0
    successes = 0

    print(f"Collecting expert data: {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs_33, _ = env.reset()
        done = False
        ep_steps = 0

        while not done:
            # Extract 27-dim subset for Phase 3.6 model
            obs_27 = obs_33_to_27(obs_33)
            action_2, _ = model.predict(obs_27, deterministic=True)

            # Save (obs_33, action_2) pair
            all_obs.append(obs_33.astype(np.float32))
            all_actions.append(action_2.astype(np.float32))

            # Step environment
            concat_act = action_2  # Box(2) for 1 pursuer
            obs_33, rew, term, trunc, info = env.step(concat_act)
            done = term or trunc
            ep_steps += 1

        total_steps += ep_steps
        if info.get('reason') == 'success':
            successes += 1

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1:>4d}/{n_episodes}]  "
                  f"success_rate={successes/(ep+1):.1%}  "
                  f"total_steps={total_steps}")

    # Save
    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.float32)
    np.savez_compressed(EXPERT_DATA_PATH, obs=obs_array, actions=act_array)
    print(f"\nSaved: {EXPERT_DATA_PATH}")
    print(f"  Episodes: {n_episodes}  |  Successes: {successes} ({successes/n_episodes:.1%})")
    print(f"  Total samples: {len(all_obs)}  |  Avg steps/ep: {total_steps/n_episodes:.1f}")
    print(f"  Obs shape: {obs_array.shape}  |  Act shape: {act_array.shape}")

    sys.stderr = _stderr
    return obs_array, act_array


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: BC Pretraining
# ═══════════════════════════════════════════════════════════════════════════

def bc_pretrain(data_path: str = EXPERT_DATA_PATH,
                epochs: int = 50,
                batch_size: int = 128,
                lr: float = 1e-3,
                device: str = "cpu"):
    """BC-train Attention Actor with mate_scale=0 on expert (obs_33, act_2) data."""
    print(f"\n{'='*60}")
    print(f"BC Pretraining — Attention Actor (mate_scale=0)")
    print(f"Data:     {data_path}")
    print(f"Epochs:   {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"{'='*60}")

    # Load data
    data = np.load(data_path)
    obs = torch.tensor(data['obs'], dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.float32)
    print(f"  Loaded: {len(obs)} samples  |  obs={list(obs.shape)}  |  act={list(actions.shape)}")

    # Split train/val
    n_val = max(int(len(obs) * 0.1), 128)
    idx = torch.randperm(len(obs))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    train_obs, train_act = obs[train_idx], actions[train_idx]
    val_obs, val_act = obs[val_idx], actions[val_idx]
    print(f"  Train: {len(train_obs)}  |  Val: {len(val_obs)}")

    train_loader = DataLoader(TensorDataset(train_obs, train_act),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_act),
                            batch_size=batch_size, shuffle=False)

    # Build Attention Actor with mate_scale=0 (force ignore mate token)
    actor = AttentionFormationActor(mate_scale=0.0).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        actor.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
            loc, _ = actor(batch_obs)  # only use action_mean, ignore scale
            loss = criterion(loc, batch_act)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item() * len(batch_obs)

        train_loss /= len(train_obs)
        scheduler.step()

        # Validate
        actor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
                loc, _ = actor(batch_obs)
                val_loss += criterion(loc, batch_act).item() * len(batch_obs)
        val_loss /= len(val_obs)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'actor_state': actor.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
            }, BC_MODEL_PATH)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"{'*' if val_loss == best_val_loss else ' '}")

    print(f"\nBC pretraining done. Best val_loss={best_val_loss:.6f}")
    print(f"Model saved: {BC_MODEL_PATH}")
    return BC_MODEL_PATH


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Actor: Expert Data + BC Pretraining")
    parser.add_argument("--collect", action="store_true", help="Step 1: collect expert data")
    parser.add_argument("--train", action="store_true", help="Step 2: BC pretrain")
    parser.add_argument("--episodes", type=int, default=500, help="Expert episodes to collect")
    parser.add_argument("--epochs", type=int, default=50, help="BC training epochs")
    parser.add_argument("--batch", type=int, default=128, help="BC batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="BC learning rate")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.collect and not args.train:
        args.collect = args.train = True  # default: both

    if args.collect:
        collect_expert_data(n_episodes=args.episodes)

    if args.train:
        bc_pretrain(data_path=EXPERT_DATA_PATH, epochs=args.epochs,
                     batch_size=args.batch, lr=args.lr, device=args.device)
