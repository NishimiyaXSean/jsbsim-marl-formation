"""Paired McNemar comparison of two closed-loop evaluation runs.

WHY
---
The small paper's headline (BC -> SPC kill rate, +48 pp) is currently a
point estimate with no significance test. This script turns two
`eval_bc_1v1.py` outputs into a paired test, which requires the two runs to
share initial seeds -- otherwise the "pairing" is fiction.

IDENTITY GUARDS (mandatory)
---------------------------
Before computing anything the script asserts, from each file's own
``run_meta`` block (never from the filename):
  * ``action_mode`` and ``difficulty`` are equal across arms
  * ``geometry`` is equal across arms
  * the seed sets are identical  -> this is what makes the test paired
  * ``model_id`` differs across arms  -> otherwise you are comparing a model
    with itself, which is exactly the error that produced the withdrawn
    "87% -> 95%" claim.

Usage:
  python scripts/paired_mcnemar.py \
      --a results/shoot_eval/E7_bc_round1_d0_n400_s20000.json \
      --b results/shoot_eval/E7_spc_d0_n400_s20000.json \
      --label-a bc_round1 --label-b spc \
      --out results/shoot_eval/E7_paired_bc_vs_spc_d0_n400.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval_meta import build_run_meta  # noqa: E402


def load_arm(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_by_seed(arm: dict) -> dict:
    """Map seed -> per-episode record. Index by seed, never by position."""
    return {int(r["seed"]): r for r in arm.get("episodes_detail", [])}


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score interval; no scipy required."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via the binomial distribution.

    Under H0 a discordant pair favours either arm with probability 1/2, so
    p = 2 * P(X <= min(b, c)) with X ~ Binomial(b + c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def chi2_mcnemar(b: int, c: int) -> float:
    """McNemar chi-square with Edwards continuity correction, 1 df."""
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / n
    if stat <= 0:
        return 1.0
    # survival function of chi2 with 1 df == erfc(sqrt(stat/2))
    return math.erfc(math.sqrt(stat / 2.0))


def check_identity(meta_a: dict, meta_b: dict, path_a: str, path_b: str) -> list:
    """Return a list of blocking problems; empty list means the comparison is sound."""
    problems = []
    for name, meta, path in (("A", meta_a, path_a), ("B", meta_b, path_b)):
        if not meta:
            problems.append(f"arm {name} ({path}) has no run_meta block; "
                            f"cannot verify identity. Re-run with the current "
                            f"eval_bc_1v1.py.")
    if problems:
        return problems

    for key in ("difficulty", "action_mode", "geometry"):
        if meta_a.get(key) != meta_b.get(key):
            problems.append(f"{key} differs: A={meta_a.get(key)!r} "
                            f"B={meta_b.get(key)!r} -> not a valid paired test")
    if meta_a.get("model_id") == meta_b.get("model_id"):
        problems.append(f"both arms report model_id={meta_a.get('model_id')!r}; "
                        f"these are the same policy, not a comparison")
    if meta_a.get("checkpoint_sha256") == meta_b.get("checkpoint_sha256"):
        problems.append("both arms loaded identical checkpoint bytes "
                        "(same sha256) -> not a comparison")
    if meta_a.get("seed_range") != meta_b.get("seed_range"):
        problems.append(f"seed_range differs: A={meta_a.get('seed_range')} "
                        f"B={meta_b.get('seed_range')}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm A json (baseline)")
    ap.add_argument("--b", required=True, help="arm B json (intervention)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm_a, arm_b = load_arm(args.a), load_arm(args.b)
    meta_a, meta_b = arm_a.get("run_meta", {}), arm_b.get("run_meta", {})

    problems = check_identity(meta_a, meta_b, args.a, args.b)
    if problems:
        print("[FATAL] identity checks failed; refusing to report a paired p-value:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)

    ep_a, ep_b = index_by_seed(arm_a), index_by_seed(arm_b)
    common = sorted(set(ep_a) & set(ep_b))
    if not common:
        print("[FATAL] no shared seeds between arms")
        sys.exit(2)
    if len(common) != len(ep_a) or len(common) != len(ep_b):
        print(f"[warn] seed sets are not identical: A={len(ep_a)} B={len(ep_b)} "
              f"common={len(common)}; using the {len(common)} shared seeds")

    both = sum(1 for s in common if ep_a[s]["kill"] and ep_b[s]["kill"])
    neither = sum(1 for s in common if not ep_a[s]["kill"] and not ep_b[s]["kill"])
    only_b = sum(1 for s in common if not ep_a[s]["kill"] and ep_b[s]["kill"])
    only_a = sum(1 for s in common if ep_a[s]["kill"] and not ep_b[s]["kill"])
    n = len(common)

    ka = both + only_a
    kb = both + only_b
    pa, pb = ka / n, kb / n
    lo_a, hi_a = wilson_ci(ka, n)
    lo_b, hi_b = wilson_ci(kb, n)

    p_exact = exact_mcnemar(only_b, only_a)
    p_chi2 = chi2_mcnemar(only_b, only_a)

    allowed_a = sum(int(ep_a[s].get("fire_allowed_steps", 0)) for s in common)
    fires_a = sum(int(ep_a[s].get("fire_commanded_on_allowed", 0)) for s in common)
    allowed_b = sum(int(ep_b[s].get("fire_allowed_steps", 0)) for s in common)
    fires_b = sum(int(ep_b[s].get("fire_commanded_on_allowed", 0)) for s in common)

    result = {
        # This file is derived from two evaluations, so it identifies itself by
        # the comparison it represents (plus the two parents below), keeping the
        # one-file-one-identity contract of scripts/eval_meta.py.
        "run_meta": build_run_meta(
            model_id=f"{args.label_a}_vs_{args.label_b}",
            checkpoint="",
            first_seed=meta_a.get("seed_range", [common[0]])[0],
            episodes=len(common),
            difficulty=meta_a.get("difficulty", float("nan")),
            min_heading_bias_deg=meta_a.get("min_heading_bias_deg"),
            action_mode=meta_a.get("action_mode", ""),
            script="scripts/paired_mcnemar.py",
            extra={
                "artifact_kind": "derived_paired_comparison",
                "derived_from": [args.a, args.b],
                "parent_model_ids": [meta_a.get("model_id"), meta_b.get("model_id")],
                "parent_checkpoint_sha256": [meta_a.get("checkpoint_sha256"),
                                             meta_b.get("checkpoint_sha256")],
            },
        ),
        "comparison": {"arm_a": args.label_a, "arm_b": args.label_b,
                       "file_a": args.a, "file_b": args.b},
        "run_meta_a": meta_a,
        "run_meta_b": meta_b,
        "paired_seeds": {"n": n, "first": common[0], "last": common[-1]},
        "contingency": {
            "both_kill": both, "neither_kill": neither,
            "only_b_kill": only_b, "only_a_kill": only_a,
            "discordant_total": only_a + only_b,
            "discordant_favouring_b": only_b,
        },
        "kill_rate": {
            args.label_a: {"kills": ka, "n": n, "rate": pa,
                           "wilson95": [lo_a, hi_a]},
            args.label_b: {"kills": kb, "n": n, "rate": pb,
                           "wilson95": [lo_b, hi_b]},
            "paired_diff_pp": (pb - pa) * 100.0,
        },
        "mcnemar": {
            "exact_two_sided_p": p_exact,
            "chi2_continuity_corrected_p": p_chi2,
            "significant_at_0.05": bool(p_exact < 0.05),
        },
        "clr_on_shared_seeds": {
            args.label_a: {"fire_allowed_steps": allowed_a,
                           "fire_commanded": fires_a,
                           "clr": (fires_a / allowed_a) if allowed_a else None},
            args.label_b: {"fire_allowed_steps": allowed_b,
                           "fire_commanded": fires_b,
                           "clr": (fires_b / allowed_b) if allowed_b else None},
        },
    }

    print("=" * 70)
    print(f"PAIRED McNEMAR — {args.label_a} vs {args.label_b}")
    print(f"  shared seeds: {n}  ({common[0]}..{common[-1]})")
    print(f"  geometry={meta_a.get('geometry')}  difficulty={meta_a.get('difficulty')}  "
          f"action_mode={meta_a.get('action_mode')}")
    print(f"  model_id: A={meta_a.get('model_id')}  B={meta_b.get('model_id')}")
    print("-" * 70)
    print(f"  kill rate  {args.label_a:<16} {ka}/{n} = {pa*100:.2f}%  "
          f"Wilson95 [{lo_a*100:.2f}, {hi_a*100:.2f}]")
    print(f"  kill rate  {args.label_b:<16} {kb}/{n} = {pb*100:.2f}%  "
          f"Wilson95 [{lo_b*100:.2f}, {hi_b*100:.2f}]")
    print(f"  paired difference            {(pb-pa)*100:+.2f} pp")
    print("-" * 70)
    print(f"  both kill {both:>4}   neither {neither:>4}   "
          f"only {args.label_b} {only_b:>4}   only {args.label_a} {only_a:>4}")
    print(f"  discordant {only_a+only_b}, favouring {args.label_b} "
          f"{only_b}/{only_a+only_b} "
          f"({100*only_b/max(only_a+only_b,1):.0f}%)")
    print(f"  exact McNemar p = {p_exact:.3e}   "
          f"chi2(cc) p = {p_chi2:.3e}   "
          f"{'SIGNIFICANT' if p_exact < 0.05 else 'NOT significant'} at 0.05")
    print("-" * 70)
    if allowed_a and allowed_b:
        print(f"  CLR  {args.label_a:<16} {fires_a}/{allowed_a} = "
              f"{100*fires_a/allowed_a:.2f}%")
        print(f"  CLR  {args.label_b:<16} {fires_b}/{allowed_b} = "
              f"{100*fires_b/allowed_b:.2f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
