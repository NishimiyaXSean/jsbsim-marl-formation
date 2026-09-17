"""Assemble the paper's comparison matrix directly from result run_meta blocks.

WHY
---
The withdrawn "87% -> 95%" claim came from hand-assembling a table out of result
files whose *names* did not match the models they held. This script removes the
hand-assembly step: it reads every result file's own ``run_meta``, keys the rows
on (model_id, difficulty, geometry, action_mode), and refuses to include a file
that is incomplete or whose identity is unverifiable.

The output is a Markdown table ready to paste into the paper, plus a warning
list for anything it had to skip.

Usage:
  python scripts/collect_matrix.py --glob 'results/shoot_eval/E*_*.json'
  python scripts/collect_matrix.py --glob 'results/shoot_eval/E5_*.json' \
      --out results/shoot_eval/matrix_d03.md
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Non-evaluation artifacts that live in the same directory, and files that were
# deliberately retired. "EXCLUDED" matters: reused-env expert runs disagree with
# the fresh-env protocol every other arm uses, so their numbers must never drift
# back into an auto-generated table by accident.
_SKIP_SUFFIX = ("_paired_", "matrix_", "EXCLUDED")


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.2f}%"


def _num(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _int(v):
    return "n/a" if v is None else str(v)


def load_rows(paths):
    rows, skipped, legacy = [], [], []
    for p in sorted(paths):
        base = os.path.basename(p)
        if "EXCLUDED" in base:
            skipped.append((base, "marked EXCLUDED (measured under a protocol that "
                                  "is not comparable to the fresh-env arms)"))
            continue
        if any(s in base for s in _SKIP_SUFFIX):
            skipped.append((base, "derived comparison / matrix file, not a single-arm eval"))
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append((base, f"unreadable: {e}"))
            continue
        m = d.get("run_meta")
        if not m:
            skipped.append((base, "no run_meta -> identity unverifiable, refusing to guess"))
            continue
        if "complete" in m:
            if not m["complete"]:
                skipped.append((base, f"incomplete ({m.get('episodes_completed')} episodes) "
                                      f"-> not admissible as evidence"))
                continue
        else:
            # Files written before the resumable refactor (7546cdb) have no
            # 'complete' key. Those scripts only wrote at the very end, so the
            # result *is* complete -- but say so out loud rather than silently
            # treating "unknown" as "fine".
            legacy.append(base)
        key = (m.get("model_id"), m.get("difficulty"), m.get("geometry"),
               m.get("action_mode"))
        rows.append({
            "file": base,
            "model_id": m.get("model_id"),
            "difficulty": m.get("difficulty"),
            "geometry": m.get("geometry"),
            "action_mode": m.get("action_mode"),
            "n": d.get("episodes"),
            "kill": d.get("kill_rate"),
            "clr": d.get("clr"),
            "launches": d.get("launches_per_episode"),
            "hit": d.get("hit_rate"),
            "lost": d.get("lost_target_rate"),
            "sha8": (m.get("checkpoint_sha256") or "n/a")[:8],
            # A rule-based policy legitimately has no checkpoint: report that as
            # 'n/a', not as 'MISSING', so it is not mistaken for a lost artifact.
            "has_ckpt": bool(m.get("checkpoint")),
            "key": key,
        })
    return rows, skipped, legacy


def load_expert_arms(paths):
    """Harvest the rule-expert arm out of the paired two-arm artifacts.

    WHY THIS EXISTS
    ---------------
    ``eval_paired_bc_vs_expert.py`` writes both arms into one file
    (``d["expert"]`` / ``d["bc"]``), so the single-arm reader above cannot see
    them and skips the file as a "derived comparison". The rule expert is
    measurable through **no** single-arm path that carries its own run_meta,
    which meant the paper's central comparison was the one table whose expert
    row still had to be copied by hand -- exactly the failure mode this script
    exists to remove. Only the expert arm is taken here; the BC arm is already
    present from E7/E5 and duplicating it would create a false identity clash.
    """
    out, skipped = [], []
    for p in sorted(paths):
        base = os.path.basename(p)
        if "EXCLUDED" in base:
            # Already reported (with the fuller reason) by load_rows.
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append((base, f"unreadable: {e}"))
            continue
        m = d.get("run_meta") or {}
        if m.get("artifact_kind") != "paired_two_arm_eval":
            continue  # single-arm artifact: already handled by load_rows
        arm = d.get("expert")
        if m.get("complete") is False:
            skipped.append((base, f"incomplete ({m.get('episodes_completed')} "
                                  f"episodes) -> not admissible as evidence"))
            continue
        if not isinstance(arm, dict) or arm.get("kill_rate") is None:
            skipped.append((base, "paired file carries no usable expert arm"))
            continue
        out.append({
            "file": base,
            "model_id": "rule_expert (fresh env)",
            "difficulty": m.get("difficulty"),
            "geometry": m.get("geometry"),
            "action_mode": m.get("action_mode"),
            "n": arm.get("n"),
            "kill": arm.get("kill_rate"),
            # Present only in the runs that instrumented CLR (2026-09-17 on).
            "clr": arm.get("clr"),
            "launches": arm.get("launches_per_episode"),
            "hit": arm.get("hit_rate"),
            "lost": arm.get("lost_rate"),
            "sha8": "n/a",
            "has_ckpt": False,
            "key": ("rule_expert (fresh env)", m.get("difficulty"),
                    m.get("geometry"), m.get("action_mode")),
        })
    return out, skipped


def pick_best(matches):
    """Prefer the row that carries the most information; report the alternates.

    The same (model, difficulty, geometry, mode) identity can legitimately
    appear in several files -- E3 d=0 exists both with and without CLR
    instrumentation. Preferring the CLR-carrying run keeps the table from
    silently losing a column, and the alternates are printed rather than
    dropped so the choice is visible.
    """
    ordered = sorted(matches, key=lambda r: (r["clr"] is None, r["file"]))
    return ordered[0], ordered[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/shoot_eval/E*_*.json")
    ap.add_argument("--out", default=None, help="write the Markdown table here")
    args = ap.parse_args()

    paths = _glob.glob(args.glob)
    if not paths:
        print(f"[matrix] no files matched {args.glob!r}")
        return
    rows, skipped, legacy = load_rows(paths)
    expert_rows, expert_skipped = load_expert_arms(paths)
    # Both readers visit the same directory, so a file can be skipped twice.
    # Report each reason once, in first-seen order.
    for entry in expert_skipped:
        if entry not in skipped:
            skipped.append(entry)
    rows += expert_rows
    if not rows:
        print("[matrix] no admissible rows")
    else:
        # Report duplicate identities loudly: two files claiming the same
        # (model, difficulty, geometry) usually means one is stale.
        seen = {}
        for r in rows:
            seen.setdefault(r["key"], []).append(r["file"])
        dupes = {k: v for k, v in seen.items() if len(v) > 1}

        diffs = sorted({r["difficulty"] for r in rows})
        models = sorted({r["model_id"] for r in rows if r["model_id"]})

        lines = []
        lines.append("| policy | model_id | difficulty | n | kill rate | CLR | launches/ep | hit | lost | sha256 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for diff in diffs:
            for mi in models:
                match = [r for r in rows
                         if r["difficulty"] == diff and r["model_id"] == mi]
                if not match:
                    continue
                r, alternates = pick_best(match)
                if alternates:
                    alt_note = ", ".join(a["file"] for a in alternates)
                    print(f"[matrix] note: {r['file']} chosen over {alt_note} "
                          f"for {mi} @ difficulty={diff} (same identity; the "
                          f"chosen row carries the CLR column if a run has it)")
                sha_cell = f"`{r['sha8']}`" if r["has_ckpt"] else "n/a (rule-based)"
                lines.append(
                    f"| {mi} | {r['file']} | {diff} | {_int(r['n'])} | "
                    f"{_pct(r['kill'])} | {_pct(r['clr'])} | "
                    f"{_num(r['launches'])} | {_num(r['hit'], 4)} | "
                    f"{_num(r['lost'], 4)} | {sha_cell} |")
        table = "\n".join(lines)
        print(table)

        geoms = sorted({r["geometry"] for r in rows})
        modes = sorted({r["action_mode"] for r in rows})
        print(f"\n[matrix] {len(rows)} admissible arm(s) from {len(paths)} file(s)")
        print(f"[matrix] geometry: {geoms}")
        print(f"[matrix] action_mode: {modes}")
        if len(geoms) > 1 or len(modes) > 1:
            print("[matrix] WARNING: rows span more than one geometry/action_mode — "
                  "do not compare them as if they were matched.")
        if dupes:
            print("[matrix] WARNING: duplicate (model, difficulty, geometry, mode) identities:")
            for k, v in dupes.items():
                print(f"    {k} <- {v}")

        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(table + "\n")
            print(f"[matrix] saved {args.out}")

    if skipped:
        print("\n[matrix] SKIPPED (not admissible as evidence):")
        for name, why in skipped:
            print(f"    {name}: {why}")
    if legacy:
        print("\n[matrix] NOTE: no 'complete' key (written before the resumable "
              "refactor, so complete by construction) -- included above:")
        for name in legacy:
            print(f"    {name}")


if __name__ == "__main__":
    main()
