"""P2A: fire-only ASAP distillation (supervised, no PPO).

Freeze: BC encoder, heading/speed/alt heads. Train ONLY the fire head so that
the policy fires whenever the environment mask allows (ASAP rule):

  fire_allowed=true  -> target fire=1  (only these steps provide gradient)
  fire_allowed=false -> mask forces fire=0 at inference; no gradient needed

Config: init from the frozen BC checkpoint, lr 1e-5-1e-4, 1-5 epochs,
masked CE on the fire head, no fire anchor (lambda_fire=0).

Gates (same harness as P1):
  * heading/speed logits identical to the frozen BC (max diff 0)
  * heading/speed step sequences 100% identical on fixed seeds
  * fire-vs-ASAP agreement on allowed windows ~100%
  * lost=0, bad=0

Quick comparison eval: rule override (ASAP), distilled, and BC on the same
seeds (ID + dist2_3k), reporting kill/lost/launches/window utilization.

Usage:
  python scripts/distill_fire_asap.py --rollout-episodes 300
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy, ACTION_DIMS, MASK_NEG
from scripts.eval_bc_1v1 import policy_action
from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch

MAX_STEPS = 1500

SCENES = [
    ("id", {}, 0.70),
    ("dist2_3k", {"chase_dist_min": 2000.0, "chase_dist_max": 3000.0}, 0.30),
]


def state_hash(sd):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().clone() for k, v in sd.items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def collect_dataset(bc_model, device, episodes, base_seed=20000):
    """Frozen-BC rollouts -> (obs, allowed) steps with episode boundaries."""
    rng = np.random.default_rng(20260807)
    cum = np.cumsum([w for _, _, w in SCENES])
    obs_all, allowed_all, eid_all = [], [], []
    n_ep = 0
    for idx in range(episodes):
        p = rng.random()
        si = int(np.searchsorted(cum, p))
        _, scene_cfg, _ = SCENES[si]
        cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
        cfg.update(scene_cfg)
        env = BaseEnv(task=SingleCombatShootTask(cfg))
        obs, _ = env.reset(seed=base_seed + idx)
        for step in range(MAX_STEPS):
            mask = env.task.get_action_mask(env, "p0")
            obs_all.append(obs["p0"].astype(np.float32).copy())
            allowed_all.append(1.0 if mask[10] == 1.0 else 0.0)
            eid_all.append(n_ep)
            act = policy_action(bc_model, obs["p0"], device)
            obs, rews, terms, truncs, info = env.step({"p0": act})
            if terms.get("__all__") or truncs.get("__all__"):
                break
            if not np.isfinite(obs["p0"]).all():
                break
        env.close()
        n_ep += 1
        if (idx + 1) % 100 == 0:
            print(f"  collect {idx+1}/{episodes}")
    return (np.array(obs_all, dtype=np.float32),
            np.array(allowed_all, dtype=np.float32),
            np.array(eid_all, dtype=np.int64))


def eval_episode(model, device, seed, scene_cfg, mode="bc"):
    """mode: bc | distilled | rule (rule = BC hdg/spd + fire=allowed)."""
    cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
    cfg.update(scene_cfg)
    env = BaseEnv(task=SingleCombatShootTask(cfg))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    launches = hits = allowed_steps = 0
    bad = 0
    quality = {"bad": 0, "good": 0, "premium": 0}
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        if mask[10] == 1.0:
            allowed_steps += 1
        if mode == "rule":
            act = policy_action(model, obs["p0"], device)
            act[3] = 1 if mask[10] == 1.0 else 0
        else:
            act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        hits += env.task._hit_this_step.get("p0", 0)
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            q = classify_launch(launch_geometry(env))
            quality[q] += 1
            if q == "bad":
                bad += 1
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {"reason": reason, "kill": reason == "target_killed",
            "lost": reason == "lost_target", "launches": launches,
            "hits": hits, "allowed_steps": allowed_steps,
            "bad": bad, "quality": quality,
            "reach4": launches >= 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-seeds", type=int, default=60)
    parser.add_argument("--weights", default="data/expert/shoot_bc_round1_baseline.pth")
    parser.add_argument("--out-weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.rollout_episodes = 30
        args.epochs = 6
        args.eval_seeds = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ck = torch.load(args.weights, map_location="cpu")
    bc_sd = ck["state_dict"]
    ref_actor_keys = {k: v for k, v in bc_sd.items()
                      if k.startswith("encoder") or k.startswith("action_heads")}
    ref_hash = state_hash(ref_actor_keys)

    bc_model = BCShootPolicy().to(device)
    bc_model.load_state_dict(bc_sd)
    bc_model.eval()

    # frozen actor reference logits
    rng = np.random.default_rng(7)
    fx = rng.uniform(-1.0, 1.0, (128, 41)).astype(np.float32)
    fx[:, 30:] = rng.integers(0, 2, (128, 11)).astype(np.float32)
    with torch.no_grad():
        _feat_ref = bc_model.encoder(torch.tensor(fx[:, :30], device=device))
        ref_logits = [head(_feat_ref).detach()
                      for head in bc_model.action_heads]

    # ── collect dataset ────────────────────────────────────────────────────
    obs, allowed, eid = collect_dataset(bc_model, device, args.rollout_episodes)
    n_allowed = int(allowed.sum())
    print(f"[p2a] dataset: steps={len(obs)} episodes={eid.max()+1} "
          f"allowed={n_allowed} ({n_allowed/len(obs)*100:.2f}%)")
    eps = np.unique(eid)
    rng2 = np.random.default_rng(0)
    shuf = rng2.permutation(eps)
    n_va = int(0.2 * len(shuf))
    va_eps, tr_eps = set(shuf[:n_va]), set(shuf[n_va:])
    tr_m = np.isin(eid, list(tr_eps))
    va_m = np.isin(eid, list(va_eps))
    allowed_tr = allowed[tr_m] == 1.0
    allowed_va = allowed[va_m] == 1.0
    print(f"  train allowed={int(allowed_tr.sum())} val allowed={int(allowed_va.sum())}")

    # ── fire-head-only distillation ────────────────────────────────────────
    distilled = BCShootPolicy().to(device)
    distilled.load_state_dict(bc_sd)
    for name, p in distilled.named_parameters():
        p.requires_grad_(False)
    for p in distilled.action_heads[3].parameters():
        p.requires_grad_(True)
    distilled.eval()
    opt = torch.optim.Adam(distilled.action_heads[3].parameters(), lr=args.lr)
    xs = torch.tensor(obs[tr_m][:, :30], device=device)
    xv = torch.tensor(obs[va_m][:, :30], device=device)
    # positive-only distillation: allowed steps -> fire=1. The disallowed
    # steps are enforced by the mask at inference (fire=0), so the model
    # simply needs to flip the fire head toward always-fire (=ASAP).
    xs_a = xs[allowed_tr]
    ys_a = torch.ones(len(xs_a), dtype=torch.long, device=device)
    xv_a = xv[allowed_va]
    yv_a = torch.ones(len(xv_a), dtype=torch.long, device=device)

    best = None
    for epoch in range(args.epochs):
        distilled.action_heads[3].train()
        idxs = np.random.default_rng(epoch).permutation(len(xs_a))
        bs = 512
        tot = 0.0
        for s in range(0, len(xs_a), bs):
            e = min(s + bs, len(xs_a))
            i = idxs[s:e]
            feat = distilled.encoder(xs_a[i])
            logits = distilled.action_heads[3](feat)
            loss = torch.nn.functional.cross_entropy(logits, ys_a[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        distilled.action_heads[3].eval()
        with torch.no_grad():
            feat_v = distilled.encoder(xv_a)
            pred = distilled.action_heads[3](feat_v).argmax(1)
            val_rec = float((pred == yv_a).float().mean())
            feat_t = distilled.encoder(xs_a)
            pred_t = distilled.action_heads[3](feat_t).argmax(1)
            tr_rec = float((pred_t == ys_a).float().mean())
        print(f"  epoch {epoch+1}/{args.epochs} fire-CE={tot/len(xs_a):.4f} "
              f"allowed-fire recall: train={tr_rec*100:.2f}% val={val_rec*100:.2f}%")
        val_acc = val_rec
        if best is None or val_acc > best[0]:
            best = (val_acc, {k: v.detach().cpu().clone()
                              for k, v in distilled.state_dict().items()})
        if val_acc >= 0.995:
            print("  early stop: allowed-window fire acc >= 99.5%")
            break
    distilled.load_state_dict(best[1])
    distilled.eval()

    # ── gates ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        _feat_new = distilled.encoder(torch.tensor(fx[:, :30], device=device))
        new_logits = [head(_feat_new).detach()
                      for head in distilled.action_heads]
    hdg_spd_diff = max(float((ref_logits[0] - new_logits[0]).abs().max()),
                       float((ref_logits[1] - new_logits[1]).abs().max()))
    gates = {}
    gates["hdg_spd_logits_identical"] = hdg_spd_diff < 1e-9
    gates["hdg_spd_logits_max_diff"] = hdg_spd_diff

    # sequence identity of heading/speed on fixed seeds
    def seq_cols(model, seeds, scene_cfg):
        cfg = {"difficulty_level": 0.0, "obs_include_closure": True}
        cfg.update(scene_cfg)
        out = {}
        for s in seeds:
            env = BaseEnv(task=SingleCombatShootTask(cfg))
            obs, _ = env.reset(seed=s)
            cols = []
            for step in range(MAX_STEPS):
                act = policy_action(model, obs["p0"], device)
                cols.append(act[:2].tolist())
                obs, rews, terms, truncs, info = env.step({"p0": act})
                if terms.get("__all__") or truncs.get("__all__"):
                    break
            env.close()
            out[s] = cols
        return out

    seeds = list(range(args.eval_seeds))
    seq_bc = seq_cols(bc_model, seeds, {})
    seq_dist = seq_cols(distilled, seeds, {})
    # episodes may end earlier (distilled kills sooner): compare the shared
    # prefix of heading/speed columns only
    def _prefix_eq(a, b):
        m = min(len(a), len(b))
        return a[:m] == b[:m]
    seq_ok = all(_prefix_eq(seq_bc[s], seq_dist[s]) for s in seeds)
    gates["hdg_spd_sequence_identical"] = seq_ok

    # ── quick comparison eval: rule / distilled / bc on same seeds ─────────
    comp = {"bc": {}, "distilled": {}, "rule": {}}
    for mode in ("bc", "distilled", "rule"):
        m = bc_model if mode in ("bc", "rule") else distilled
        recs_id = [eval_episode(m, device, s, {}, mode) for s in seeds]
        recs_d2 = [eval_episode(m, device, s,
                                {"chase_dist_min": 2000.0,
                                 "chase_dist_max": 3000.0}, mode) for s in seeds]
        for tag, recs in (("id", recs_id), ("dist2_3k", recs_d2)):
            n = len(recs)
            kills = sum(r["kill"] for r in recs)
            launches = sum(r["launches"] for r in recs)
            allowed_tot = sum(r["allowed_steps"] for r in recs)
            fire_tot = sum(r["launches"] for r in recs)
            comp[mode][tag] = {
                "kill_rate": kills / n,
                "lost_rate": sum(r["lost"] for r in recs) / n,
                "launches_per_ep": launches / n,
                "reach4_rate": sum(r["reach4"] for r in recs) / n,
                "window_utilization": fire_tot / max(allowed_tot, 1),
                "bad_total": sum(r["bad"] for r in recs),
                "hit_rate": sum(r["hits"] for r in recs) / max(launches, 1),
            }
    gates["fire_allowed_agreement"] = bool(
        comp["distilled"]["id"]["window_utilization"] > 0.95 and
        comp["distilled"]["dist2_3k"]["window_utilization"] > 0.95)

    print("=" * 66)
    for mode in ("bc", "rule", "distilled"):
        print(f"[{mode:<9s}] "
              + "  ".join(f"{tag}: kill {comp[mode][tag]['kill_rate']*100:.1f}% "
                          f"lost {comp[mode][tag]['lost_rate']*100:.1f}% "
                          f"launch {comp[mode][tag]['launches_per_ep']:.2f} "
                          f"reach4 {comp[mode][tag]['reach4_rate']*100:.1f}% "
                          f"winutil {comp[mode][tag]['window_utilization']*100:.0f}% "
                          f"bad {comp[mode][tag]['bad_total']}"
                          for tag in ("id", "dist2_3k")))
    print(f"[p2a] gates: hdg/spd logits diff={hdg_spd_diff:.2e} "
          f"seq={seq_ok} fire-agreement={gates['fire_allowed_agreement']}")
    verdict = (gates["hdg_spd_logits_identical"] and seq_ok
               and gates["fire_allowed_agreement"]
               and comp["distilled"]["id"]["lost_rate"] == 0
               and comp["distilled"]["id"]["bad_total"] == 0
               and comp["distilled"]["dist2_3k"]["lost_rate"] == 0
               and comp["distilled"]["dist2_3k"]["bad_total"] == 0)
    print(f"[p2a] VERDICT: {'PASS' if verdict else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out_weights) or ".", exist_ok=True)
    torch.save({
        "state_dict": {k: v for k, v in distilled.state_dict().items()},
        "meta": {"base": args.weights, "lr": args.lr,
                 "epochs": args.epochs, "rollout_episodes": args.rollout_episodes,
                 "best_val_allowed_acc": best[0], "verdict": verdict},
        "gates": gates,
        "comparison": comp,
    }, args.out_weights)
    print(f"[p2a] saved: {args.out_weights}")
    with open("logs/p2a_distill.json", "w", encoding="utf-8") as f:
        json.dump({"gates": gates, "comparison": comp,
                   "best_val_allowed_acc": best[0]}, f, indent=2)


if __name__ == "__main__":
    main()








