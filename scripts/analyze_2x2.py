"""Analyse the 2x2 heading-bias control produced by scripts/_run_geom2x2.sh.

Cells
  base_geoOld   baseline weights @ U(30,60)   <- reproduction check, expect 87/100
  geomA_geoNew  geomA weights    @ U(0,60)    <- reproduction check, expect 95/100
  base_geoNew   baseline weights @ U(0,60)
  geomA_geoOld  geomA weights    @ U(30,60)

Because both cells of a contrast share the same seed and the same geometry, the
per-episode outcomes are paired on (seed, min_heading_bias_deg) and an exact
McNemar test is valid -- unlike the original 87 vs 95 comparison.

Usage:
  /home/sean/miniconda3/envs/marl_env/bin/python scripts/analyze_2x2.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAGS = ("base_geoOld", "geomA_geoNew", "base_geoNew", "geomA_geoOld")
CELLS = {}
REPRO_CHECK = {"base_geoOld": (87, 100), "geomA_geoNew": (95, 100)}


def cell_paths(eps: int, seed: int):
    """Mirror the output naming rule of scripts/_run_geom2x2.sh."""
    sfx = "" if eps == 100 else "_n%d" % eps
    return {t: "results/shoot_eval/eval_2x2_%s%s_s%d.json" % (t, sfx, seed)
            for t in TAGS}
Z = 1.959963985


def wilson(k: int, n: int):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe(k1: int, n1: int, k2: int, n2: int):
    lo1, hi1 = wilson(k1, n1)
    lo2, hi2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = p1 - p2 - math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2)
    hi = p1 - p2 + math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2)
    return lo, hi


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d

    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)

    p_obs = p(a)
    tot = 0.0
    for x in range(max(0, a - d), min(a + b, a + c) + 1):
        px = p(x)
        if px <= p_obs + 1e-12:
            tot += px
    return tot


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c = discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def load(tag: str):
    path = os.path.join(ROOT, CELLS[tag])
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=int, default=100,
                        help="episodes per cell; any value != 100 expects the "
                             "_n<eps> output suffix written by _run_geom2x2.sh")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    global CELLS
    CELLS = cell_paths(args.eps, args.seed)

    data = {tag: load(tag) for tag in CELLS}
    missing = [t for t, d in data.items() if d is None]
    if missing:
        print("NOTE: cells not found, skipped: %s" % ", ".join(missing))
        print("      (run scripts/_run_geom2x2.sh --eps %d to produce them)" % args.eps)
    if all(d is None for d in data.values()):
        print("no cells available at eps=%d seed=%d" % (args.eps, args.seed))
        return 2

    print("=" * 78)
    print("A. per-cell kill rate (n = 100, seed 42, difficulty 0.0)")
    print("=" * 78)
    ks = {}
    for tag in CELLS:
        d = data[tag]
        if d is None:
            continue
        k = d["termination_reasons"].get("target_killed", 0)
        n = d["episodes"]
        ks[tag] = (k, n)
        lo, hi = wilson(k, n)
        note = ""
        if args.eps == 100 and tag in REPRO_CHECK:
            ek, en = REPRO_CHECK[tag]
            note = ("  REPRO OK (== %d/%d)" % (ek, en) if (k, n) == (ek, en)
                    else "  REPRO MISMATCH (expected %d/%d)" % (ek, en))
        print("  %-13s %3d/%d = %.3f  Wilson95%% [%.3f, %.3f]%s"
              % (tag, k, n, k / n, lo, hi, note))
    print("  weights: base = shoot_bc_asap_distilled.pth, "
          "geomA = shoot_bc_asap_geomA.pth")
    print("  geometry: geoOld = min_heading_bias_deg 30 (U(30,60)), "
          "geoNew = 0 (U(0,60))")

    # ---- init heading bias actually realised ------------------------------
    print()
    print("=" * 78)
    print("B. realised initial heading bias |bias| (deg)")
    print("=" * 78)
    for tag in ("base_geoOld", "geomA_geoNew", "base_geoNew", "geomA_geoOld"):
        d = data[tag]
        if d is None:
            continue
        bs = [abs(e["init_heading_bias_deg"]) for e in d["episodes_detail"]]
        print("  %-13s mean %.2f  median %.2f  min %.2f  max %.2f"
              % (tag, sum(bs) / len(bs), sorted(bs)[len(bs) // 2],
                 min(bs), max(bs)))

    # ---- paired contrasts --------------------------------------------------
    def paired(tag_a: str, tag_b: str, label: str):
        if data[tag_a] is None or data[tag_b] is None:
            print()
            print("-" * 78)
            print("%s" % label)
            print("  skipped: cell not available at this eps")
            return
        da, db = data[tag_a], data[tag_b]
        ka = {e["seed"]: e["kill"] for e in da["episodes_detail"]}
        kb = {e["seed"]: e["kill"] for e in db["episodes_detail"]}
        seeds = sorted(set(ka) & set(kb))
        n11 = n10 = n01 = n00 = 0
        for s in seeds:
            if ka[s] and kb[s]:
                n11 += 1
            elif ka[s] and not kb[s]:
                n10 += 1
            elif kb[s] and not ka[s]:
                n01 += 1
            else:
                n00 += 1
        b, c = n10, n01
        p = mcnemar_exact(b, c)
        diff = (sum(ka[s] for s in seeds) - sum(kb[s] for s in seeds)) / len(seeds)
        print()
        print("-" * 78)
        print("%s" % label)
        print("  paired seeds: %d   A kills %d/%d, B kills %d/%d"
              % (len(seeds), sum(ka[s] for s in seeds), len(seeds),
                 sum(kb[s] for s in seeds), len(seeds)))
        print("  diff (A - B) = %+.1f pp" % (diff * 100))
        print("  concordant: both kill %d, neither kill %d" % (n11, n00))
        print("  discordant: A-only kill %d, B-only kill %d" % (b, c))
        print("  exact McNemar two-sided p = %.4f" % p)
        if p < 0.05:
            print("  => SIGNIFICANT at 0.05 (paired test)")
        else:
            print("  => not significant at 0.05 (paired test)")

    print()
    print("=" * 78)
    print("C. paired contrasts")
    print("=" * 78)
    paired("geomA_geoNew", "base_geoNew",
           "C1  POLICY effect under identical NEW geometry (geomA vs base, both U(0,60))")
    paired("geomA_geoOld", "base_geoOld",
           "C2  POLICY effect under identical OLD geometry (geomA vs base, both U(30,60))")
    paired("base_geoNew", "base_geoOld",
           "C3  GEOMETRY effect at fixed baseline weights (U(0,60) vs U(30,60))")
    paired("geomA_geoNew", "geomA_geoOld",
           "C4  GEOMETRY effect at fixed geomA weights (U(0,60) vs U(30,60))")

    # ---- unpaired cross-check on the headline pair ------------------------
    print()
    print("=" * 78)
    print("D. unpaired cross-check (for reference; these are NOT the same scenario)")
    print("=" * 78)
    for tag_a, tag_b, label in (("geomA_geoNew", "base_geoOld",
                                 "geomA@U(0,60) vs base@U(30,60)  [the original +8 pp claim]"),
                                ("geomA_geoNew", "base_geoNew",
                                 "geomA@U(0,60) vs base@U(0,60)")):
        if tag_a not in ks or tag_b not in ks:
            continue
        ka, na = ks[tag_a]
        kb, nb = ks[tag_b]
        lo, hi = newcombe(ka, na, kb, nb)
        p = fisher_two_sided(ka, na - ka, kb, nb - kb)
        print("  %s" % label)
        print("    %d/%d vs %d/%d = %+.1f pp   Newcombe95%% [%+.1f, %+.1f] pp   Fisher p = %.4f"
              % (ka, na, kb, nb, 100 * (ka / na - kb / nb),
                 100 * lo, 100 * hi, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
