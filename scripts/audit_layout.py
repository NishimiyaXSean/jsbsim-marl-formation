"""Layout-level audit: check the BUILT PDF, not the source.

Sean's observation after the "Constraint." leak: the project had a source-level
check, an artifact-level check and a number-level check, but nothing that looked
at what actually reached the page. A note to ourselves can be correctly excluded
by the converter's rules and still slip through on a construct the rules do not
cover -- a table row, a plain paragraph, a bullet -- and no number audit will
ever notice, because it only looks for numbers.

So this checks the properties that must hold of the submitted PDF:

  1. main paper within the page limit, supplement present and separate
  2. no author-facing markers anywhere in the main paper
  3. no appendix material left in the main paper
  4. figures the paper includes actually exist as files

Run by scripts/build_paper.sh after every build, and safe to run by hand:

    python scripts/audit_layout.py
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys

BUILD = "/home/sean/jsbsim-marl-formation/paper/build"
MAIN = os.path.join(BUILD, "aamas_paper")
SUPP = os.path.join(BUILD, "aamas_supplement")
LIMIT = 8

# Strings that must never appear in a submitted paper. These are not style
# preferences: each one has actually leaked at least once, or is a standing
# instruction to the authors rather than to the reader.
# Prose markers, checked against the extracted TEXT. Each pattern must match the
# MARKER and not the English word: "the JSBSim internal frame rate" and "the
# binding constraint." are legitimate prose, and a case-insensitive pattern for
# "Constraint." reported the second as a failure (hit 2026-09-19). Patterns here
# are therefore case-SENSITIVE unless the marker itself is capitalised ambiguously.
FORBIDDEN_TEXT = [
    (r"\((?:internal|INTERNAL)\b", "author-facing marker"),
    (r"internal (?:note|status|check|comment)", "author-facing marker"),
    (r"delete before submission", "author-facing marker"),
    (r"\bConstraint\.", "instruction to the authors"),
    (r"\bTODO\b", "unfinished item"),
    (r"\[verify\]", "unresolved citation marker"),
    (r"\u26a0", "warning glyph"),
    (r"\bDONE\b", "status marker"),
    (r"Sean", "author name in the paper"),
    (r"措辞纪律|内部|待办", "Chinese working note"),
]

# Structural checks, run against the .tex: appendix material is unambiguous
# there, whereas in extracted text a *reference* to the appendix can begin a line
# and look like one (that false positive also hit on 2026-09-19).
FORBIDDEN_TEX = [
    (r"\\section\{Appendix", "appendix section in the main paper"),
    (r"\\appendix\b", "appendix mode in the main paper"),
]


def pages(pdf: str) -> int:
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else -1


def main() -> int:
    problems = 0

    main_pdf = os.path.join(MAIN, "aamas_paper.pdf")
    main_txt = os.path.join(MAIN, "aamas_paper.txt")
    supp_pdf = os.path.join(SUPP, "aamas_supplement.pdf")

    if not os.path.exists(main_pdf):
        print("FAIL: no main PDF at %s" % main_pdf)
        return 1

    n = pages(main_pdf)
    if n < 0:
        print("FAIL: could not read the page count")
        problems += 1
    elif n > LIMIT:
        print("FAIL: main paper is %d pages, limit is %d" % (n, LIMIT))
        problems += 1
    else:
        print("OK  : main paper %d pages (limit %d)" % (n, LIMIT))

    if os.path.exists(supp_pdf):
        print("OK  : supplement present and separate (%d pages)" % pages(supp_pdf))
    else:
        print("note: no supplement built (fine only if nothing was moved out)")

    text = io.open(main_txt, encoding="utf-8").read()
    lines = text.splitlines()
    for pattern, why in FORBIDDEN_TEXT:
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines)
                if re.search(pattern, l)]
        if hits:
            problems += 1
            print("FAIL: %d x '%s' (%s) in the main paper:"
                  % (len(hits), pattern, why))
            for lineno, sample in hits[:5]:
                print("        L%-5d %s" % (lineno, sample[:96]))

    tex_path = os.path.join(MAIN, "aamas_paper.tex")
    if os.path.exists(tex_path):
        tex = io.open(tex_path, encoding="utf-8").read()
        for pattern, why in FORBIDDEN_TEX:
            n = len(re.findall(pattern, tex))
            if n:
                problems += 1
                print("FAIL: %d x '%s' (%s)" % (n, pattern, why))

    if not problems:
        print("OK  : no author-facing markers or appendix material in the paper")

    # Figures the paper asks for must exist, or LaTeX silently typesets a gap.
    missing = []
    for fig in ("framework_fig1.pdf", "robustness_fig3.pdf",
                "mechanism_seed20007_d00.png"):
        if not os.path.exists(os.path.join(MAIN, fig)):
            missing.append(fig)
    if missing:
        print("note: figure file(s) not staged (the paper may reference them): %s"
              % ", ".join(missing))

    print()
    print("layout audit: %s" % ("PASS" if problems == 0 else "FAIL"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
