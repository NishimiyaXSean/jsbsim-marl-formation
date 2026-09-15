"""Stratified companions to scripts/analyze_2x2.py.

Adds three things the headline report needs:
  1. Stratified (pooled) exact McNemar over both geometry strata, plus the
     sample size at which the observed effect would reach p < 0.05.
  2. Kill rate stratified by the realised |init_heading_bias_deg|, so the
     policy effect can be read at MATCHED difficulty instead of on average.
  3. Secondary metrics at matched geometry.

Usage: /home/sean/miniconda3/envs/marl_env/bin/python scripts/analyze_2x2_strata.py
"""

from __future__ import annotations

import json
import os
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = {
    "base_geoOld": "eval_2x2_base_geoOld_s42.json",
    "geomA_geoNew": "eval_2x2_geomA_geoNew_s42.json",
    "base_geoNew": "eval_2x2_base_geoNew_s42.json",
    "geomA_geoOld": "eval_2x2_geomA_geoOld_s42.json",
}


def load(tag):
    with open(os.path.join(ROOT, "results/shoot_eval", FILES[tag]), encoding="utf-8") as fh:
        return json.load(fh)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def discordant(da, db):
    ka = {e["seed"]: e["kill"] for e in da["episodes_detail"]}
    kb = {e["seed"]: e["kill"] for e in db["episodes_detail"]}
    seeds = sorted(set(ka) & set(kb))
    b = sum(1 for s in seeds if ka[s] and not kb[s])
    c = sum(1 for s in seeds if kb[s] and not ka[s])
    return b, c, len(seeds)


D = {t: load(t) for t in FILES}

print("=" * 76)
print("E. stratified (pooled) exact McNemar")
print("=" * 76)
POLICY = [("geomA_geoNew", "base_geoNew"), ("geomA_geoOld", "base_geoOld")]
GEOM = [("base_geoNew", "base_geoOld"), ("geomA_geoNew", "geomA_geoOld")]

for label, pairs in (("POLICY effect (geomA - base), both geometries pooled", POLICY),
                     ("GEOMETRY effect (U(0,60) - U(30,60)), both policies pooled", GEOM)):
    tb = tc = tn = 0
    for a, b in pairs:
        x, y, n = discordant(D[a], D[b])
        tb += x
        tc += y
        tn += n
        print("  stratum %-24s A-only %d, B-only %d (n=%d)" % (a + " vs " + b, x, y, n))
    print("  POOLED: A-only %d, B-only %d over n=%d   exact McNemar p = %.4f"
          % (tb, tc, tn, mcnemar_exact(tb, tc)))
    rate_b, rate_c = tb / tn, tc / tn
    print("  discordance rates: A-only %.3f, B-only %.3f" % (rate_b, rate_c))
    print("  n needed to reach p<0.05 at these rates:")
    for n in (100, 200, 300, 400, 500, 600, 800, 1000):
        bb, cc = round(rate_b * n), round(rate_c * n)
        print("     n=%4d -> discordant %2d/%2d, p = %.4f" % (n, bb, cc, mcnemar_exact(bb, cc)))
    print()

print("=" * 76)
print("F. kill rate by realised |init_heading_bias_deg| (matched difficulty)")
print("=" * 76)
EDGES = [(0, 15), (15, 30), (30, 45), (45, 61)]
for geo, pairs in (("geoNew U(0,60)", [("base_geoNew", "base  "), ("geomA_geoNew", "geomA ")]),
                   ("geoOld U(30,60)", [("base_geoOld", "base  "), ("geomA_geoOld", "geomA ")])):
    print("  %s" % geo)
    for tag, name in pairs:
        d = D[tag]
        row = []
        for lo, hi in EDGES:
            sel = [e for e in d["episodes_detail"] if lo <= abs(e["init_heading_bias_deg"]) < hi]
            row.append("%2d-%2d: %2d/%2d (%.0f%%)"
                       % (lo, hi, sum(e["kill"] for e in sel), len(sel),
                          100 * sum(e["kill"] for e in sel) / len(sel)) if sel
                       else "%2d-%2d:    -" % (lo, hi))
        print("    %s %s" % (name, "  ".join(row)))
    print()

print("=" * 76)
print("G. secondary metrics at matched geometry (policy effect)")
print("=" * 76)
for geo, a, b in (("geoNew U(0,60)", "geomA_geoNew", "base_geoNew"),
                  ("geoOld U(30,60)", "geomA_geoOld", "base_geoOld")):
    da, db = D[a], D[b]
    print("  %s  geomA vs base" % geo)
    for key in ("hit_rate", "launches_per_episode", "wez_reach_rate", "mean_steps"):
        print("    %-22s %.4f vs %.4f" % (key, da[key], db[key]))
    print("    %-22s %s vs %s" % ("termination_reasons",
                                  da["termination_reasons"], db["termination_reasons"]))
