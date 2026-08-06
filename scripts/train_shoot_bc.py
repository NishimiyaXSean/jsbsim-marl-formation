"""Behavioral cloning on the v2 stateless rule-expert shoot dataset.

Implements the Gate-3 BC requirements from the reviewed plan:
  * episode-level train/val/test split (80/10/10, rng=0 — identical to audit)
  * masked logits: invalid actions -> -1e9 (never skip fire=0 samples)
  * fire head trained ONLY on fire_allowed rows; B/C window sampling ~2:1
  * fire loss weight starts at 2.0; altitude head (dim=1) skipped
  * heading-balanced minibatches: heading-0 ~50-65%, non-zero ~35-50%
  * metrics: per-class recall, macro-F1, turn direction/magnitude accuracy,
    fire B/C precision/recall on allowed rows
  * unit compatibility test: BC logits == ShootMaskModel (RLlib) logits per
    head (value net ignored)

Usage:
  python scripts/train_shoot_bc.py --smoke
  python scripts/train_shoot_bc.py --epochs 40 --batch-size 512 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["JSBSIM_DEBUG"] = "0"
import warnings

warnings.filterwarnings("ignore")

ACTION_DIMS = [3, 5, 1, 2]  # speed, heading, altitude, fire
HDG0 = 2
FIRE_IDX = 10
MASK_NEG = -1e9


class BCShootPolicy(nn.Module):
    """Policy trunk + action heads; structurally identical to ShootMaskModel.

    State-dict keys (encoder.*, action_heads.*) match ShootMaskModel so the
    trained weights can be loaded straight into the RLlib policy (policy trunk
    + action heads only; the value net stays randomly initialized).
    """

    def __init__(self, obs_dim=30, action_dims=None, hidden=(256, 256, 128)):
        super().__init__()
        action_dims = list(action_dims or ACTION_DIMS)
        self.action_dims = action_dims
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden[-1], d) for d in action_dims]
        )

    def forward(self, obs):
        feat = self.encoder(obs)
        return [head(feat) for head in self.action_heads]


def load_data(path):
    d = np.load(path)
    keys = ["obs", "action", "mask", "episode_id", "fire_allowed",
            "fire_desired", "target_spd"]
    return {k: d[k] for k in keys}


def episode_split(eids, seed=0, val_frac=0.1, test_frac=0.1):
    """Same episode split as the audit (rng 0, 80/10/10)."""
    eps = np.unique(eids)
    rng = np.random.default_rng(seed)
    shuf = rng.permutation(eps)
    n_tr = int((1.0 - val_frac - test_frac) * len(shuf))
    n_va = int(val_frac * len(shuf))
    tr = set(shuf[:n_tr])
    va = set(shuf[n_tr:n_tr + n_va])
    te = set(shuf[n_tr + n_va:])
    return (
        np.isin(eids, list(tr)),
        np.isin(eids, list(va)),
        np.isin(eids, list(te)),
    )


class ExpertBatchSampler:
    """Rebalanced minibatch sampler.

    window_frac of each batch comes from fire_allowed rows with B:C ~ bc_ratio;
    the rest is heading-balanced (hdg0_frac heading-0, the remainder sampled
    uniformly across the four non-zero heading classes).
    """

    def __init__(self, train_idx, hdg, allowed, desired, batch_size,
                 window_frac=0.2, hdg0_frac=0.55, bc_ratio=2.0,
                 seed=42):
        self.batch_size = batch_size
        self.window_frac = window_frac
        self.hdg0_frac = hdg0_frac
        self.bc_ratio = bc_ratio
        self.rng = np.random.default_rng(seed)
        self.hdg0 = train_idx[hdg[train_idx] == HDG0]
        self.hdg_nz = train_idx[hdg[train_idx] != HDG0]
        self.b_rows = train_idx[(allowed[train_idx] == 1.0)
                                & (desired[train_idx] == 0.0)]
        self.c_rows = train_idx[(allowed[train_idx] == 1.0)
                                & (desired[train_idx] == 1.0)]

    @staticmethod
    def _pick(pool, k, rng):
        if len(pool) >= k:
            return rng.choice(pool, k, replace=False)
        return rng.choice(pool, k, replace=True)

    def sample_batch(self):
        n_win = max(int(self.batch_size * self.window_frac), 1)
        n_b = int(round(n_win * self.bc_ratio / (self.bc_ratio + 1.0)))
        n_c = n_win - n_b
        rest = self.batch_size - n_win
        n0 = int(round(rest * self.hdg0_frac))
        nnz = rest - n0
        idx = np.concatenate([
            self._pick(self.b_rows, n_b, self.rng),
            self._pick(self.c_rows, n_c, self.rng),
            self._pick(self.hdg0, n0, self.rng),
            self._pick(self.hdg_nz, nnz, self.rng),
        ])
        self.rng.shuffle(idx)
        return idx


def per_class_recall(y, p, n_cls):
    return [float((p[y == c] == c).mean()) if (y == c).sum() else 0.0
            for c in range(n_cls)]


def macro_f1(y, p, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def evaluate(model, obs, act, mask, allowed, idx, device, bs=4096):
    model.eval()
    hdg_y, hdg_p, spd_y, spd_p = [], [], [], []
    fire_y, fire_p = [], []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            e = min(s + bs, len(idx))
            i = idx[s:e]
            xb = torch.tensor(obs[i, :30], device=device)
            yb = act[i]
            logits = model(xb)
            hdg_p.append(logits[1].argmax(1).cpu().numpy())
            spd_p.append(logits[0].argmax(1).cpu().numpy())
            hdg_y.append(yb[:, 1])
            spd_y.append(yb[:, 0])
            al = allowed[i] == 1.0
            if al.any():
                fire_y.append(yb[al, 3])
                fire_p.append(logits[3][al].argmax(1).cpu().numpy())
    hdg_y = np.concatenate(hdg_y)
    hdg_p = np.concatenate(hdg_p)
    spd_y = np.concatenate(spd_y)
    spd_p = np.concatenate(spd_p)

    hdg_acc = float((hdg_p == hdg_y).mean())
    rec = per_class_recall(hdg_y, hdg_p, 5)
    mf1 = macro_f1(hdg_y, hdg_p, 5)
    zero_m = hdg_y == HDG0
    dir_acc = float((((hdg_p == HDG0) & zero_m)
                     | ((hdg_p != HDG0) & (~zero_m)
                        & (np.sign(hdg_p - HDG0) == np.sign(hdg_y - HDG0)))
                     ).mean())
    nz = hdg_y != HDG0
    mag_acc = float((np.abs(hdg_p[nz] - HDG0) == np.abs(hdg_y[nz] - HDG0)).mean()) \
        if nz.any() else float("nan")
    spd_acc = float((spd_p == spd_y).mean())

    out = {"hdg_acc": hdg_acc, "hdg_recall": rec, "hdg_macro_f1": mf1,
           "hdg_dir_acc": dir_acc, "hdg_mag_acc": mag_acc,
           "spd_acc": spd_acc, "n": len(idx)}
    if fire_y:
        fy = np.concatenate(fire_y)
        fp = np.concatenate(fire_p)
        tp = int(((fp == 1) & (fy == 1)).sum())
        fp_n = int(((fp == 1) & (fy == 0)).sum())
        fn = int(((fp == 0) & (fy == 1)).sum())
        prec = tp / (tp + fp_n) if tp + fp_n else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out["fire_allowed_n"] = int(len(fy))
        out["fire_precision"] = prec
        out["fire_recall"] = rec
        out["fire_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["fire_acc"] = float((fp == fy).mean())
        out["fire_pred_rate"] = float(fp.mean())
        out["fire_true_rate"] = float(fy.mean())
    else:
        out["fire_allowed_n"] = 0
        out["fire_f1"] = float("nan")
    return out


def compute_loss(model, xb, yb, mb, ab, spd_w, fire_weight):
    logits = model(xb)  # [spd, hdg, alt, fire]
    hdg_l = logits[1] + (1.0 - mb[:, 3:8]) * MASK_NEG
    spd_l = logits[0] + (1.0 - mb[:, 0:3]) * MASK_NEG
    l_hdg = F.cross_entropy(hdg_l, yb[:, 1])
    l_spd = F.cross_entropy(spd_l, yb[:, 0], weight=spd_w)
    l_fire = torch.zeros((), device=xb.device)
    if ab.any():
        f_l = logits[3][ab] + (1.0 - mb[ab, 9:11]) * MASK_NEG
        l_fire = F.cross_entropy(f_l, yb[ab, 3])
    total = l_hdg + l_spd + fire_weight * l_fire
    return total, (l_hdg.item(), l_spd.item(), l_fire.item())


def compat_check(model, device, n=256, tol=1e-4):
    """BC logits must match ShootMaskModel (RLlib) logits on the same obs."""
    from gymnasium import spaces
    from src.models.shoot_mask_model import ShootMaskModel

    obs_space = spaces.Box(-1e6, 1e6, (41,), np.float32)
    act_space = spaces.MultiDiscrete([3, 5, 1, 2])
    rllib = ShootMaskModel(obs_space, act_space, 11, {}, "bc").to(device)
    res = rllib.load_state_dict(model.state_dict(), strict=False)
    unexpected = set(res.unexpected_keys)
    missing = set(res.missing_keys)
    assert not unexpected, f"unexpected keys: {sorted(unexpected)}"
    assert all(k.startswith("value_net") for k in missing), \
        f"missing non-value keys: {sorted(missing)}"

    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, (n, 41)).astype(np.float32)
    x[:, 30:] = rng.integers(0, 2, (n, 11)).astype(np.float32)
    with torch.no_grad():
        bc_l = torch.cat(model(torch.tensor(x[:, :30], device=device)), dim=1)
        # apply the same mask penalty ShootMaskModel.forward applies
        mb = torch.tensor(x[:, 30:], device=device)
        off = 0
        for d in ACTION_DIMS:
            bc_l[:, off:off + d] = bc_l[:, off:off + d] \
                + (1.0 - mb[:, off:off + d]) * MASK_NEG
            off += d
        rl_l, _ = rllib({"obs": torch.tensor(x, device=device)}, [], None)
    diff = float((bc_l - rl_l).abs().max())
    ok = diff < tol
    print(f"[compat] BC vs ShootMaskModel logits: max diff {diff:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expert/shoot_rule_expert.npz")
    parser.add_argument("--output", default="data/expert/shoot_bc_weights.pth")
    parser.add_argument("--metrics-out", default="logs/bc_train_metrics.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-frac", type=float, default=0.2,
                        help="fraction of each batch from fire_allowed rows")
    parser.add_argument("--hdg0-frac", type=float, default=0.55,
                        help="heading-0 share of the non-window part")
    parser.add_argument("--bc-ratio", type=float, default=2.0,
                        help="B:C ratio inside the fire-window part")
    parser.add_argument("--fire-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on a small episode subset")
    parser.add_argument("--no-compat", action="store_true",
                        help="skip the RLlib logits compatibility test")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── data ──────────────────────────────────────────────────────────────
    d = load_data(args.data)
    obs, act, mask = d["obs"], d["action"], d["mask"]
    eids, allowed, desired = (d["episode_id"], d["fire_allowed"],
                              d["fire_desired"])
    assert obs.shape[1] == 41 and np.abs(obs[:, -11:] - mask).max() == 0
    n = len(obs)
    tr_m, va_m, te_m = episode_split(eids)
    train_idx = np.flatnonzero(tr_m)
    val_idx = np.flatnonzero(va_m)
    test_idx = np.flatnonzero(te_m)

    print("=" * 66)
    print(f"BC TRAIN — {args.data}")
    print(f"transitions={n}  train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}")
    c = (allowed == 1.0) & (desired == 1.0)
    for name, m in (("train", tr_m), ("val", va_m), ("test", te_m)):
        nc = int((c & m).sum())
        na = int(((allowed == 1.0) & m).sum())
        print(f"  {name:<5s} allowed={na:>6d} C={nc:>5d}")
    hdg = act[:, 1]
    hc = np.bincount(hdg[train_idx], minlength=5)
    print(f"  train heading dist {hc.tolist()} "
          f"({(hc / hc.sum() * 100).round(1).tolist()}%)")
    spd = act[:, 0]
    sc = np.bincount(spd[train_idx], minlength=3)
    print(f"  train speed dist {sc.tolist()} "
          f"({(sc / sc.sum() * 100).round(1).tolist()}%)")

    if args.smoke:
        tr_eps = np.unique(eids[train_idx])[:12]
        va_eps = np.unique(eids[val_idx])[:3]
        te_eps = np.unique(eids[test_idx])[:3]
        train_idx = np.flatnonzero(np.isin(eids, tr_eps))
        val_idx = np.flatnonzero(np.isin(eids, va_eps))
        test_idx = np.flatnonzero(np.isin(eids, te_eps))
        args.epochs = 2
        args.batch_size = min(args.batch_size, 128)
        print(f"[smoke] reduced to train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)}, epochs={args.epochs}")

    sampler = ExpertBatchSampler(
        train_idx, hdg, allowed, desired, args.batch_size,
        window_frac=args.window_frac, hdg0_frac=args.hdg0_frac,
        bc_ratio=args.bc_ratio, seed=args.seed)
    print(f"  sampler pools: hdg0={len(sampler.hdg0)} hdg_nz={len(sampler.hdg_nz)} "
          f"B={len(sampler.b_rows)} C={len(sampler.c_rows)}")

    # gentle speed class weights from the train distribution
    freq = np.bincount(spd[train_idx], minlength=3).astype(np.float64)
    w = np.sqrt(np.median(freq) / np.maximum(freq, 1.0))
    w = np.clip(w, 0.5, 2.0)
    spd_w = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"  speed class weights: {w.round(3).tolist()}")

    model = BCShootPolicy().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model params={n_param:,} device={device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    n_batches = max(len(train_idx) // args.batch_size, 1)
    best = {"score": -1.0}
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        comp_hdg0, comp_win, comp_bc = [], [], []
        for _ in range(n_batches):
            idx = sampler.sample_batch()
            xb = torch.tensor(obs[idx, :30], device=device)
            yb = torch.tensor(act[idx], device=device)
            mb = torch.tensor(mask[idx], device=device)
            ab = torch.tensor(allowed[idx] == 1.0, device=device)
            loss, _parts = compute_loss(model, xb, yb, mb, ab, spd_w,
                                        args.fire_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item()
            nb += 1
            comp_hdg0.append(float((yb[:, 1] == HDG0).float().mean()))
            comp_win.append(float(ab.float().mean()))
            aw = ab.cpu().numpy()
            if aw.any():
                comp_bc.append(float((desired[idx[aw]] == 1.0).mean()))
        scheduler.step()
        train_loss = tot / max(nb, 1)
        val = evaluate(model, obs, act, mask, allowed, val_idx, device)
        score = val["hdg_macro_f1"] + (val.get("fire_f1") or 0.0)
        ep = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "batch_hdg0_frac": float(np.mean(comp_hdg0)),
            "batch_window_frac": float(np.mean(comp_win)),
            "batch_c_frac_of_window": float(np.mean(comp_bc)) if comp_bc else 0.0,
            **{f"val_{k}": (v if not isinstance(v, list) else v)
               for k, v in val.items()},
            "val_composite": score,
        }
        history.append(ep)
        if score > best["score"]:
            best["score"] = score
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch + 1
            best["val"] = val
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
                f"hdg_acc={val['hdg_acc']*100:5.1f}% mF1={val['hdg_macro_f1']:.3f} "
                f"dir={val['hdg_dir_acc']*100:5.1f}% "
                f"fire(prec/rec)={val.get('fire_precision', float('nan')):.3f}/"
                f"{val.get('fire_recall', float('nan')):.3f} "
                f"comp(hdg0/win)={ep['batch_hdg0_frac']*100:.0f}%/"
                f"{ep['batch_window_frac']*100:.0f}%  {time.time()-t0:.0f}s")

    # ── final: best weights on val + test ─────────────────────────────────
    model.load_state_dict(best["state"])
    test = evaluate(model, obs, act, mask, allowed, test_idx, device)
    val = evaluate(model, obs, act, mask, allowed, val_idx, device)
    print("=" * 66)
    print(f"BEST epoch={best['epoch']}  val_composite={best['score']:.4f}")
    for name, m in (("val", val), ("test", test)):
        print(f"  {name:<4s} hdg_acc={m['hdg_acc']*100:.1f}% "
              f"mF1={m['hdg_macro_f1']:.3f} dir={m['hdg_dir_acc']*100:.1f}% "
              f"recall={np.round(m['hdg_recall'], 3).tolist()} "
              f"spd_acc={m['spd_acc']*100:.1f}% "
              f"fire_prec={m.get('fire_precision', float('nan')):.3f} "
              f"fire_rec={m.get('fire_recall', float('nan')):.3f} "
              f"fire_f1={m.get('fire_f1', float('nan')):.3f} "
              f"(allowed_n={m['fire_allowed_n']})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "state_dict": best["state"],
        "meta": {
            "data": args.data, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "window_frac": args.window_frac,
            "hdg0_frac": args.hdg0_frac, "bc_ratio": args.bc_ratio,
            "fire_weight": args.fire_weight, "seed": args.seed,
            "best_epoch": best["epoch"],
        },
        "val": {k: v for k, v in best["val"].items()
                if not isinstance(v, list)},
        "test": {k: v for k, v in test.items() if not isinstance(v, list)},
    }, args.output)
    print(f"saved weights: {args.output}")

    if not args.no_compat:
        compat_check(model, device)

    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"history": history,
                   "best_epoch": best["epoch"],
                   "val": {k: v for k, v in best["val"].items()
                           if not isinstance(v, list)},
                   "test": {k: v for k, v in test.items()
                            if not isinstance(v, list)},
                   "config": vars(args)}, f, indent=2)
    print(f"saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()



