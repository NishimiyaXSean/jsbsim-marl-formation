"""S4: train ONLY the speed head on the B2 teacher dataset.

Initialized from the ASAP-distilled checkpoint. encoder / heading / fire /
alt heads frozen; only action_heads[0] (speed [-20,0,+20]) is trainable.

Loss (per sample):
  L = w_speed * CE(teacher_speed, student)
      + (0.05..0.1 if NOT b2_region else 0) * KL(old_speed || student)

The old-policy KL protects the ~75-85% of states that must stay unchanged;
inside the B2 region the student is free to learn the new deceleration.

Checkpoint selection uses a fixed dev CLOSED-LOOP set every epoch (never
offline accuracy), ordered by:
  lost==0 & bad==0 -> ID kill -> dist2_3k kill -> 3-hit (lower) ->
  4th-window rate -> TTK p90.

Usage:
  python scripts/train_speed_head.py --epochs 12 --dev-id 40 --dev-d2 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

from src.environment.base_env import BaseEnv
from src.environment.singlecombat_shoot_task import SingleCombatShootTask
from scripts.train_shoot_bc import BCShootPolicy
from scripts.eval_bc_1v1 import policy_action

MAX_STEPS = 1500


def dev_episode(model, device, seed, cfg):
    env = BaseEnv(task=SingleCombatShootTask(
        {"difficulty_level": 0.0, "obs_include_closure": True, **cfg}))
    obs, _ = env.reset(seed=seed)
    reason = "timeout"
    launches = hits = 0
    bad = 0
    t4win = None
    fire3 = None
    from scripts.eval_paired_bc_vs_expert import launch_geometry, classify_launch
    for step in range(MAX_STEPS):
        mask = env.task.get_action_mask(env, "p0")
        act = policy_action(model, obs["p0"], device)
        obs, rews, terms, truncs, info = env.step({"p0": act})
        if env.task._has_launched_this_step.get("p0", False):
            launches += 1
            if launches == 3:
                fire3 = step
            if classify_launch(launch_geometry(env)) == "bad":
                bad += 1
        if env.task._hit_this_step.get("p0", 0) > 0:
            hits += 1
        if fire3 is not None and mask[10] == 1.0 and t4win is None \
                and step > fire3:
            t4win = step
        if terms.get("__all__") or truncs.get("__all__"):
            reason = info.get("p0", {}).get("termination_reason", "unknown")
            break
    env.close()
    return {"kill": reason == "target_killed", "lost": reason == "lost_target",
            "bad": bad, "3hit": reason == "timeout" and hits == 3,
            "4th": t4win is not None, "steps": step + 1}


def dev_eval(model, device, dev_id, dev_d2, dev_ood):
    recs_id = [dev_episode(model, device, 5000 + s, {})
               for s in range(dev_id)]
    recs_d2 = [dev_episode(model, device, 6000 + s,
                           {"chase_dist_min": 2000.0,
                            "chase_dist_max": 3000.0})
               for s in range(dev_d2)]
    recs_ood = []
    for s in range(dev_ood // 2):
        recs_ood.append(dev_episode(model, device, 7000 + s,
                                    {"alt_diff_m": 300.0}))
        recs_ood.append(dev_episode(model, device, 7100 + s,
                                    {"difficulty_level": 0.3}))
    out = {"lost": sum(r["lost"] for r in recs_id + recs_d2 + recs_ood),
           "bad": sum(r["bad"] for r in recs_id + recs_d2 + recs_ood),
           "id_kill": sum(r["kill"] for r in recs_id) / max(dev_id, 1),
           "d2_kill": sum(r["kill"] for r in recs_d2) / max(dev_d2, 1),
           "id_3hit": sum(r["3hit"] for r in recs_id),
           "d2_3hit": sum(r["3hit"] for r in recs_d2),
           "id_4th": sum(r["4th"] for r in recs_id) / max(dev_id, 1),
           "d2_4th": sum(r["4th"] for r in recs_d2) / max(dev_d2, 1),
           "ttk_p90_id": float(np.percentile(
               [r["steps"] * 0.2 for r in recs_id if r["kill"]], 90))
           if any(r["kill"] for r in recs_id) else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expert/speed_teacher_data.npz")
    parser.add_argument("--weights", default="data/expert/shoot_bc_asap_distilled.pth")
    parser.add_argument("--out", default="data/expert/shoot_bc_speed_distilled.pth")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--kl-lambda", type=float, default=0.05)
    parser.add_argument("--dev-id", type=int, default=40)
    parser.add_argument("--dev-d2", type=int, default=40)
    parser.add_argument("--dev-ood", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    d = np.load(args.data)
    obs = d["obs"]
    speed = d["speed"]
    orig = d["orig_speed"]
    b2_reg = d["b2_region"]
    print(f"[s4] data n={len(obs)} b2_region={b2_reg.mean()*100:.1f}% "
          f"speed_hist={np.bincount(speed, minlength=3).tolist()}")

    model = BCShootPolicy().to(device)
    ck = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()
    # cache old speed-head logits (frozen reference for the KL term)
    with torch.no_grad():
        feat_all = model.encoder(torch.tensor(obs[:, :30], device=device))
        old_logits = model.action_heads[0](feat_all).detach()
        old_logp = F.log_softmax(old_logits, dim=1)
    # freeze everything except the speed head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.action_heads[0].parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.action_heads[0].parameters(), lr=args.lr)

    xs = torch.tensor(obs[:, :30], device=device)
    ys = torch.tensor(speed, device=device)
    br = torch.tensor(b2_reg, device=device, dtype=torch.float32)
    # mild weight on the -20 class (local behavior, keep natural distribution)
    cls_w = torch.tensor([1.5, 1.0, 1.0], device=device)

    best = {"score": None, "state": None, "epoch": 0, "dev": None}
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        model.action_heads[0].train()
        idxs = rng.permutation(len(xs))
        tot = 0.0
        for s in range(0, len(xs), args.batch_size):
            e = min(s + args.batch_size, len(xs))
            i = idxs[s:e]
            feat = model.encoder(xs[i])
            logits = model.action_heads[0](feat)
            ce = F.cross_entropy(logits, ys[i], weight=cls_w)
            kl = F.kl_div(F.log_softmax(logits, dim=1), old_logp[i],
                          reduction="batchmean", log_target=True)
            lam = args.kl_lambda * (1.0 - br[i]).mean()
            loss = ce + lam * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_heads[0].parameters(),
                                           1.0)
            opt.step()
            tot += loss.item() * len(i)
        model.action_heads[0].eval()
        dev = dev_eval(model, device, args.dev_id, args.dev_d2, args.dev_ood)
        # agreement on the dataset (offline sanity)
        with torch.no_grad():
            feat_v = model.encoder(xs)
            pred = model.action_heads[0](feat_v).argmax(1)
            agr_b2 = float((pred[br == 1] == ys[br == 1]).float().mean()) \
                if int(br.sum()) else float("nan")
            agr_non = float((pred[br == 0] == ys[br == 0]).float().mean()) \
                if int((1 - br).sum()) else float("nan")
        print(f"  ep {epoch+1:3d} loss={tot/len(xs):.4f} "
              f"dev(lost={dev['lost']} bad={dev['bad']} "
              f"id_kill={dev['id_kill']*100:.1f}% d2_kill={dev['d2_kill']*100:.1f}% "
              f"3hit={dev['id_3hit']}/{dev['d2_3hit']} "
              f"agr(b2/non)={agr_b2*100:.1f}/{agr_non*100:.1f}%)")
        score = (
            0 if (dev["lost"] == 0 and dev["bad"] == 0) else -1000,
            dev["id_kill"], dev["d2_kill"], -dev["id_3hit"] - dev["d2_3hit"],
            dev["id_4th"] + dev["d2_4th"],
            -(dev["ttk_p90_id"] or 0.0),
        )
        if best["score"] is None or score > best["score"]:
            best["score"] = score
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            best["epoch"] = epoch + 1
            best["dev"] = dev
            best["agr"] = (agr_b2, agr_non)

    model.load_state_dict(best["state"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": best["state"],
        "meta": {"base": args.weights, "lr": args.lr,
                 "epochs": args.epochs, "kl_lambda": args.kl_lambda,
                 "best_epoch": best["epoch"]},
        "dev": best["dev"],
        "agreement_b2_non": best["agr"],
    }, args.out)
    print(f"[s4] best epoch={best['epoch']} dev={best['dev']} "
          f"agr={best['agr']}")
    print(f"[s4] saved {args.out}")
    with open("logs/s4_train.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best["epoch"], "dev": best["dev"],
                   "agreement": best["agr"]}, f, indent=2)


if __name__ == "__main__":
    main()
