"""Numerical validation of scripts/paired_mcnemar.py.

These are checked against *independently computed* values already recorded in
the repository, so they validate the implementation rather than restate it:

  * exact McNemar for 19 vs 6 discordant pairs must equal 0.0146
    (results/shoot_eval/geom2x2_n400_report.md, n=400 geometry comparison)
  * Wilson 95% intervals must match the intervals printed in the same report
    for 362/400 and 375/400

Run:  python tests/test_paired_mcnemar_stats.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.paired_mcnemar import chi2_mcnemar, exact_mcnemar, wilson_ci

FAILURES = []


def check(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got:.6f} want={want:.6f}")
    if not ok:
        FAILURES.append(name)


print("exact McNemar vs geom2x2_n400_report.md (b=19, c=6)")
check("exact_two_sided_p", exact_mcnemar(19, 6), 0.0146)
# the test is two-sided, so swapping the discordant arms must not change p
assert exact_mcnemar(19, 6) == exact_mcnemar(6, 19), "exact McNemar is not symmetric"
print("  [ok ] symmetry under arm swap")

print("chi-square (continuity corrected) sanity")
p = chi2_mcnemar(19, 6)
# 2*(1-Phi(z)) with z = (|19-6|-1)/sqrt(25) = 2.4 -> ~0.0164
check("chi2_cc_p", p, 0.0164, tol=2e-3)

print("degenerate / boundary cases")
assert exact_mcnemar(0, 0) == 1.0
assert exact_mcnemar(1, 0) == 1.0
assert chi2_mcnemar(0, 0) == 1.0
print("  [ok ] no discordant pairs -> p = 1.0")

print("Wilson 95% intervals vs geom2x2_n400_report.md")
lo, hi = wilson_ci(362, 400)
check("wilson_lo(362/400)", lo, 0.872, tol=1.5e-3)
check("wilson_hi(362/400)", hi, 0.930, tol=1.5e-3)
lo, hi = wilson_ci(375, 400)
check("wilson_lo(375/400)", lo, 0.909, tol=1.5e-3)
check("wilson_hi(375/400)", hi, 0.957, tol=1.5e-3)

print("companion-table sanity: 400 = both + neither + discordant")
both, neither, only_b, only_a = 356, 19, 19, 6
assert both + neither + only_b + only_a == 400
print("  [ok ] 356 + 19 + 19 + 6 = 400")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
