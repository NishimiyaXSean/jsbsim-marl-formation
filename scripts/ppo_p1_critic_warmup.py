"""PPO P1: independent critic warm-up on frozen-BC trajectories.

The BC actor (encoder + heading/speed/fire heads) is FULLY frozen: eval mode,
requires_grad_(False), and its parameters are NOT in the critic optimizer.
The critic has an independent encoder + value head (ShootFireOnlyModel),
initialized from the BC encoder weights but with fully separate parameters.

Reward spec (identical to P2 stage 1):
  damage : +R_HP per HP removed (default 250)
  kill   : +KILL at the terminal step (default 1000)
  bad    : -BAD on any bad-window launch (default 100)
  premium: no bonus; launch cost 0

Rollout scene mix: ID ~65%, dist2_3k ~20%, other OOD ~15%.

Stop conditions:
  1) value val-loss plateau (patience epochs)
  2) explained variance stable positive
  3) value ordering: mean V(kill eps) > mean V(no-kill eps)
  4) actor state-dict sha256 identical before/after
  5) actor logits identical on a fixed obs batch
  6) fixed seeds: per-step action sequences + outcomes identical

Usage:
  python scripts/ppo_p1_critic_warmup.py --rollout-episodes 400
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from gymnasium import spaces

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from src.models.shoot_mask_model import ShootFireOnlyModel
from scripts.train_shoot_bc import BCShootPolicy, ACTION_DIMS
from scripts.eval_bc_1v1 import policy_action
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch

MAX_STEPS = 1500

SCENES = [
    ("id", {}, 0.65),
    ("dist2_3k", {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0}, 0.20),
    ("alt_diff300", {"alt_diff_m": 300.0}, 0.05),
    ("closure_separating", {"p_speed_range": (150.0, 180.0),
                            "t_speed_range": (210.0, 240.0)}, 0.05),
    ("target_evasive", {"difficulty_level": 0.3}, 0.05),
]


def state_hash(sd):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().clone() for k, v in sd.items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def rollout_episode(bc_model, device, seed, scene_cfg, reward_cfg, gamma):
    """Run one frozen-BC episode; return (obs_steps, returns, outcome)."""
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(scene_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    rewards = []
    obs_steps = []
    launches = 0
    reason = "timeout"
    for step in range(MAX_STEPS):
        obs_steps.append(obs["p0"].astype(np.float32).copy())
        act = policy_action(bc_model, obs["p0"], device)
        r = 0.0
        hits_before = env.targets[0].hits_taken
        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits_now = env.targets[0].hits_taken
        if hits_now > hits_before:
            r += (hits_now - hits_before) * reward_cfg["damage"]
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if classify_launch(launch_geometry(env)) == "bad":
                r -= reward_cfg["bad"]
        rewards.append(r)
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            if reason == "target_killed":
                rewards[-1] += reward_cfg["kill"]
            break
        if not np.isfinite(obs["p0"]).all():
            reason = "jsbsim_nan"
            break
    env.close()
    ret = 0.0
    returns = np.zeros(len(rewards), dtype=np.float64)
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * ret
        returns[i] = ret
    outcome = {
        "kill": reason == "target_killed",
        "lost": reason == "lost_target",
        "launches": launches,
        "reason": reason,
    }
    return obs_steps, returns, outcome


def deterministic_model_action(model, obs, device):
    x = torch.tensor(obs[:30], dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.tensor(obs[30:], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model({"obs": torch.cat([x, m], dim=1)}, [], None)
    act = np.zeros(4, dtype=np.int64)
    off = 0
    for i, d in enumerate(ACTION_DIMS):
        lg = logits[0, off:off + d]
        act[i] = int(lg.argmax().item())
        off += d
    return act


def verify_sequences(model, device, seeds, scene_cfg):
    """Return per-seed (actions_list, outcome) using the model's actor."""
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(scene_cfg)
    out = {}
    for s in seeds:
        env = BaseEnv(task=SingleCombatShootTask(cfg))
        obs, _ = env.reset(seed=s)
        acts = []
        reason = "timeout"
        launches = 0
        for step in range(MAX_STEPS):
            act = deterministic_model_action(model, obs["p0"], device)
            acts.append(act.tolist())
            obs, rews, terms, truncs, info = env.step({"p0": act})
            if env.task._has_launched_this_step.get("p0", False):
                launches += 1
            if terms.get("__all__") or truncs.get("__all__"):
                reason = info.get("p0", {}).get("termination_reason", "unknown")
                break
        env.close()
        out[s] = {"acts": acts, "reason": reason,
                  "kill": reason == "target_killed",
                  "lost": reason == "lost_target", "launches": launches}
    return out


def fixed_logits(model, device, n=128):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, (n, 41)).astype(np.float32)
    x[:, 30:] = rng.integers(0, 2, (n, 11)).astype(np.float32)
    with torch.no_grad():
        logits, _ = model({"obs": torch.tensor(x, device=device)}, [], None)
    return logits.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-episodes", type=int, default=400)
    parser.add_argument("--verify-seeds", type=int, default=100)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--damage-reward", type=float, default=250.0)
    parser.add_argument("--kill-bonus", type=float, default=1000.0)
    parser.add_argument("--bad-penalty", type=float, default=100.0)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out-critic", default="data/expert/shoot_critic_p1.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.rollout_episodes = 20
        args.verify_seeds = 4
        args.max_epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location="cpu")
    bc_sd = ck["state_dict"]
    reward_cfg = {"damage": args.damage_reward, "kill": args.kill_bonus,
                  "bad": args.bad_penalty}

    bc_model = BCShootPolicy().to(device)
    bc_model.load_state_dict(bc_sd)
    bc_model.eval()

    obs_space = spaces.Box(-1e6, 1e6, (41,), np.float32)
    act_space = spaces.MultiDiscrete([3, 5, 1, 2])
    model = ShootFireOnlyModel(obs_space, act_space, 11, {}, "p1").to(device)
    for k, v in bc_sd.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)
    with torch.no_grad():
        for src, dst in zip(model.encoder.parameters(),
                            model.critic_encoder.parameters()):
            dst.copy_(src)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for head in model.action_heads:
        for p in head.parameters():
            p.requires_grad_(False)
    model.encoder.eval()
    for head in model.action_heads:
        head.eval()

    actor_params = set(model.encoder.parameters()) | \
        set(p for h in model.action_heads for p in h.parameters())
    critic_params = list(model.critic_encoder.parameters()) + \
        list(model.critic_value.parameters())
    assert not (set(critic_params) & actor_params), "critic/actor overlap"
    opt = torch.optim.Adam(critic_params, lr=args.critic_lr)
    print(f"[p1] critic params={sum(p.numel() for p in critic_params):,} "
          f"actor frozen={len(actor_params)} params")

    # actor invariance references BEFORE training
    verify_seeds = list(range(args.verify_seeds))
    seq_before = verify_sequences(model, device, verify_seeds, {})
    logits_before = fixed_logits(model, device)

    # rollout (frozen BC trajectories, mixed scenes)
    rng = np.random.default_rng(20260806)
    cum = np.cumsum([w for _, _, w in SCENES])
    episodes = []
    for idx in range(args.rollout_episodes):
        p = rng.random()
        si = int(np.searchsorted(cum, p))
        _, scene_cfg, _ = SCENES[si]
        obs_steps, returns, outcome = rollout_episode(
            bc_model, device, 10000 + idx, scene_cfg, reward_cfg, args.gamma)
        episodes.append({"scene": SCENES[si][0], "obs": obs_steps,
                         "returns": returns, "outcome": outcome})
        if (idx + 1) % 100 == 0:
            print(f"  rollout {idx+1}/{args.rollout_episodes}")
    n_tr = int(0.8 * len(episodes))
    tr_eps, va_eps = episodes[:n_tr], episodes[n_tr:]
    X_tr = np.concatenate([e["obs"] for e in tr_eps])[:, :30].astype(np.float32)
    y_tr = np.concatenate([e["returns"] for e in tr_eps]).astype(np.float32)
    X_va = np.concatenate([e["obs"] for e in va_eps])[:, :30].astype(np.float32)
    y_va = np.concatenate([e["returns"] for e in va_eps]).astype(np.float32)
    print(f"[p1] rollout eps={len(episodes)} train_steps={len(X_tr)} "
          f"val_steps={len(X_va)} scenes="
          f"{ {k: sum(1 for e in episodes if e['scene'] == k) for k, _, _ in SCENES} }")
    print(f"[p1] return stats: mean={y_tr.mean():.1f} std={y_tr.std():.1f} "
          f"kill-eps={sum(e['outcome']['kill'] for e in episodes)}")

    # critic training (independent params only)
    best_val = float("inf")
    best_sd = None
    patience_left = args.patience
    for epoch in range(args.max_epochs):
        model.critic_encoder.train()
        model.critic_value.train()
        idxs = np.random.default_rng(epoch).permutation(len(X_tr))
        bs = 512
        tot = 0.0
        for s in range(0, len(X_tr), bs):
            e = min(s + bs, len(X_tr))
            i = idxs[s:e]
            xb = torch.tensor(X_tr[i], device=device)
            yb = torch.tensor(y_tr[i], device=device)
            opt.zero_grad()
            pred = model.critic_value(model.critic_encoder(xb)).squeeze(1)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            opt.step()
            tot += loss.item() * len(i)
        tr_loss = tot / len(X_tr)
        with torch.no_grad():
            model.critic_encoder.eval()
            model.critic_value.eval()
            pred_v = model.critic_value(
                model.critic_encoder(torch.tensor(X_va, device=device))).squeeze(1)
            val_loss = float(torch.nn.functional.mse_loss(
                pred_v, torch.tensor(y_va, device=device)).item())
        if val_loss < best_val * 0.995:
            best_val = val_loss
            best_sd = {("critic_encoder." + k): v.detach().cpu().clone()
                       for k, v in model.critic_encoder.state_dict().items()}
            best_sd.update({("critic_value." + k): v.detach().cpu().clone()
                            for k, v in model.critic_value.state_dict().items()})
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch % 5 == 0 or patience_left <= 0:
            print(f"  epoch {epoch+1:3d} train={tr_loss:.3e} val={val_loss:.3e} "
                  f"patience={patience_left}")
        if patience_left <= 0:
            break
    assert best_sd is not None
    model.load_state_dict(
        {k: v for k, v in best_sd.items() if k.startswith("critic_")},
        strict=False)

    with torch.no_grad():
        model.critic_encoder.eval()
        model.critic_value.eval()
        pred_v = model.critic_value(
            model.critic_encoder(torch.tensor(X_va, device=device))).squeeze(1)
        residual = pred_v.cpu().numpy() - y_va
        explained = 1.0 - float(np.var(residual)) / max(float(np.var(y_va)), 1e-9)
        va_kill = [e for e in va_eps if e["outcome"]["kill"]]
        va_nokill = [e for e in va_eps if not e["outcome"]["kill"]]
        v_kill = float(np.mean([float(model.critic_value(model.critic_encoder(
            torch.tensor(e["obs"][0][:30], device=device).unsqueeze(0))).item())
            for e in va_kill])) if va_kill else 0.0
        v_nokill = float(np.mean([float(model.critic_value(model.critic_encoder(
            torch.tensor(e["obs"][0][:30], device=device).unsqueeze(0))).item())
            for e in va_nokill])) if va_nokill else 0.0

    # stop-condition checks
    checks = {}
    checks["val_loss"] = best_val
    checks["explained_variance"] = explained
    checks["value_order_kill_gt_nokill"] = bool(v_kill > v_nokill)
    checks["v_kill_mean"] = v_kill
    checks["v_nokill_mean"] = v_nokill
    actor_sd_now = {("encoder." + k): v.detach().cpu().clone()
                    for k, v in model.encoder.state_dict().items()}
    actor_sd_now.update({("action_heads." + k): v.detach().cpu().clone()
                         for k, v in model.action_heads.state_dict().items()})
    ref_actor_sd = {k: v for k, v in bc_sd.items()
                    if k.startswith("encoder") or k.startswith("action_heads")}
    checks["actor_hash_unchanged"] = bool(
        state_hash(actor_sd_now) == state_hash(ref_actor_sd))
    logits_now = fixed_logits(model, device)
    checks["actor_logits_abs_max"] = float(np.abs(logits_now - logits_before).max())
    checks["actor_logits_unchanged"] = checks["actor_logits_abs_max"] < 1e-9
    seq_after = verify_sequences(model, device, verify_seeds, {})
    seq_ok = True
    for s in verify_seeds:
        a, b = seq_before[s], seq_after[s]
        if a["acts"] != b["acts"] or a["reason"] != b["reason"] \
                or a["kill"] != b["kill"] or a["lost"] != b["lost"] \
                or a["launches"] != b["launches"]:
            seq_ok = False
            break
    checks["sequence_identity_100seeds"] = seq_ok
    checks["verify_seeds"] = args.verify_seeds

    print("=" * 66)
    print("[p1] stop-condition checks")
    print(f"  val_loss={best_val:.3e}  explained_variance={explained:.3f}")
    print(f"  value order kill>nokill: {v_kill:.1f} > {v_nokill:.1f} -> "
          f"{checks['value_order_kill_gt_nokill']}")
    print(f"  actor hash unchanged: {checks['actor_hash_unchanged']}")
    print(f"  actor logits unchanged (max diff): "
          f"{checks['actor_logits_abs_max']:.2e}")
    print(f"  {args.verify_seeds}-seed action-sequence identity: {seq_ok}")
    verdict = (explained > 0.0 and checks["value_order_kill_gt_nokill"]
               and checks["actor_hash_unchanged"]
               and checks["actor_logits_unchanged"] and seq_ok)
    print(f"[p1] VERDICT: {'PASS' if verdict else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out_critic) or ".", exist_ok=True)
    torch.save({
        "critic_encoder": {k: v for k, v in best_sd.items()
                           if k.startswith("critic_encoder")},
        "critic_value": {k: v for k, v in best_sd.items()
                         if k.startswith("critic_value")},
        "actor": actor_sd_now,
        "meta": {"critic_lr": args.critic_lr, "gamma": args.gamma,
                 "reward": reward_cfg, "rollout_episodes": args.rollout_episodes,
                 "best_val_loss": best_val, "verdict": verdict},
        "checks": checks,
    }, args.out_critic)
    print(f"[p1] saved critic: {args.out_critic}")
    os.makedirs("logs", exist_ok=True)
    with open("logs/p1_warmup.json", "w", encoding="utf-8") as f:
        json.dump({"checks": checks, "meta": {
            "critic_lr": args.critic_lr, "gamma": args.gamma,
            "reward": reward_cfg}}, f, indent=2)


if __name__ == "__main__":
    main()


