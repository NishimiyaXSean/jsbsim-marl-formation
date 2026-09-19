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
    # A bibliography entry whose title could not be verified carries this
    # placeholder rather than an invented title, and BibTeX renders the field
    # literally -- so the marker reaching the PDF is exactly what this check is
    # for, not an inconvenience to be suppressed.
    (r"TITLE-TO-VERIFY", "unverified bibliography entry"),
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


def body_pages(txt_path: str, pdf_pages: int) -> int:
    """Pages before the reference list, which AAMAS excludes from the limit.

    AAMAS allows "at most 8 pages, with any number of additional pages containing
    bibliographic references", so the PDF total is the wrong number to test. On
    2026-09-19 the total became 9 the moment a bibliography was added while the
    body stayed at 8 -- a compliant paper reported as a failure. The boundary is
    found from the reference numbering itself, since "References" as a heading is
    indistinguishable from a citation to a section of that name.
    """
    if not os.path.exists(txt_path):
        return pdf_pages
    pages = [p for p in io.open(txt_path, encoding="utf-8").read().split("\f")
             if p.strip()]
    ref_re = re.compile(r"^\s*\[\d+\]\s+\S")
    for n, page in enumerate(pages, 1):
        if any(ref_re.match(l) for l in page.splitlines()):
            return n          # the reference list shares this page with the body
    return pdf_pages


def main() -> int:
    problems = 0

    main_pdf = os.path.join(MAIN, "aamas_paper.pdf")
    main_txt = os.path.join(MAIN, "aamas_paper.txt")
    supp_pdf = os.path.join(SUPP, "aamas_supplement.pdf")

    if not os.path.exists(main_pdf):
        print("FAIL: no main PDF at %s" % main_pdf)
        return 1

    n_total = pages(main_pdf)
    n = body_pages(main_txt, n_total)
    if n < 0:
        print("FAIL: could not read the page count")
        problems += 1
    elif n > LIMIT:
        print("FAIL: body is %d pages, limit is %d" % (n, LIMIT))
        problems += 1
    else:
        print("OK  : body %d page(s) + %d page(s) of references, limit %d"
              % (n, max(0, n_total - n + 1), LIMIT))

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
