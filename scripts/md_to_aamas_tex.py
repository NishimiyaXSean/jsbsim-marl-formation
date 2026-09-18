"""Convert paper/small_paper_draft.md into an AAMAS-2026/27 LaTeX paper.

WHY A CONVERTER RATHER THAN A HAND-TRANSCRIPTION
------------------------------------------------
The Markdown draft is the single source of truth: audit_paper_numbers.py guards
the numbers in *it*, and the prose is edited there. Hand-copying it into .tex
would create a second copy that silently drifts -- which is the exact failure
this project has already been bitten by twice (the 43.2/91.2 pair, and the two
CLR "diagnostic" rows that were labelled with the wrong geometry). Generating
the .tex keeps one source and makes every rebuild reproducible:

    python scripts/md_to_aamas_tex.py            # writes paper/aamas_paper.tex
    bash   scripts/build_paper.sh                # compiles + audits the PDF text

WHAT IT DOES
------------
Only the paper is emitted. Draft-only material is dropped by rule, not by hand:
sections titled 6./7./8./9. are bookkeeping, and any blockquote marked
"(internal, delete before submission)" is a note to the authors.

Markdown constructs handled: ATX headings, paragraphs, bullet and numbered
lists, bold/italic/code spans, pipe tables (rendered as booktabs tabular),
blockquotes, and the Unicode symbols actually used in this draft. Anything it
cannot classify is passed through escaped, so a compile error points at the one
line that needs a human.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No citation processing: the draft cites inline ("Li et al. (2024)"), so the
baseline PDF has no bibliography yet. Converting to \\cite{} plus a .bib is a
separate step, and pretending otherwise here would hide the remaining work.
"""

from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))
SRC = os.path.join(ROOT, "paper", "small_paper_draft.md")
OUT = os.path.join(ROOT, "paper", "aamas_paper.tex")

TITLE = ("Diagnosing and Surgically Correcting Conservative Decision Bias "
         "in Imitation-Learned Air Combat Policies")

KEYWORDS = ("imitation learning, behaviour cloning, expert-induced bias, "
            "causal identification, policy intervention, air combat")

# Top-level draft sections that are project bookkeeping, not paper content.
SKIP_SECTIONS = (
    "## 6.", "## 7.", "## 8.", "## 9.",
    "## Title Candidates", "## Appendix A.",
)

# Unicode seen in this draft -> LaTeX. Longest first so multi-char forms win.
UNICODE = [
    # Multi-character forms first: "χ²" must not be split into $\chi$ then $^2$,
    # which compiles but reads as two separate objects.
    ("χ²", r"$\chi^2$"), ("χ2", r"$\chi^2$"),
    ("≈", r"$\approx$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
    ("±", r"$\pm$"), ("≪", r"$\ll$"), ("≫", r"$\gg$"), ("⩽", r"$\leq$"),
    ("∈", r"$\in$"), ("∉", r"$\notin$"), ("⊂", r"$\subset$"), ("⊆", r"$\subseteq$"),
    ("∅", r"$\emptyset$"), ("∀", r"$\forall$"), ("∃", r"$\exists$"),
    ("→", r"$\rightarrow$"), ("←", r"$\leftarrow$"), ("↔", r"$\leftrightarrow$"),
    ("⇒", r"$\Rightarrow$"), ("⇔", r"$\Leftrightarrow$"), ("↑", r"$\uparrow$"),
    ("↓", r"$\downarrow$"), ("×", r"$\times$"), ("÷", r"$\div$"),
    ("·", r"$\cdot$"), ("⋅", r"$\cdot$"), ("∘", r"$\circ$"), ("√", r"$\sqrt{}$"),
    ("∑", r"$\sum$"), ("∂", r"$\partial$"), ("∫", r"$\int$"),
    ("∼", r"$\sim$"), ("≈", r"$\approx$"), ("≡", r"$\equiv$"), ("≠", r"$\neq$"),
    ("∝", r"$\propto$"), ("µ", r"$\mu$"), ("μ", r"$\mu$"),
    ("°", r"$^\circ$"), ("θ", r"$\theta$"), ("Δ", r"$\Delta$"),
    ("α", r"$\alpha$"), ("ε", r"$\epsilon$"), ("λ", r"$\lambda$"),
    ("χ", r"$\chi$"), ("²", r"$^2$"), ("³", r"$^3$"), ("¹", r"$^1$"),
    ("−", "-"), ("–", "--"), ("—", "---"), ("‒", "--"), ("―", "---"),
    ("’", "'"), ("‘", "'"), ("“", "``"), ("”", "''"),
    ("«", "``"), ("»", "''"), ("…", r"\ldots{}"), ("⋯", r"$\cdots$"),
    ("¼", r"1/4"), ("¾", r"3/4"), ("½", r"1/2"),
    ("✓", r"$\checkmark$"), ("⚠", ""), ("✅", ""), ("✗", r"$\times$"),
    ("†", r"\dag{}"), ("‡", r"\ddag{}"), ("§", r"\S{}"),
    ("\u00a0", " "), ("\u2009", r"\,"), ("\u200b", ""),
]

CITE_RE = re.compile(r"`([^`]+)`")


def escape(text: str) -> str:
    """Escape LaTeX specials for ordinary prose, leaving $...$ math alone."""
    out = []
    in_math = False
    for ch in text:
        if ch == "$":
            in_math = not in_math
            out.append(ch)
            continue
        if not in_math:
            if ch in "&%#":
                out.append("\\" + ch)
                continue
            if ch == "_":
                out.append(r"\_")
                continue
            if ch == "\\":
                out.append(r"\textbackslash{}")
                continue
        out.append(ch)
    return "".join(out)


def inline(text: str) -> str:
    """Inline markdown -> LaTeX, escaping everything not explicitly marked."""
    # Unicode FIRST, before any span is stashed: a symbol sitting inside a code
    # span (``hit_rate ≈ 0.997``) would otherwise be preserved verbatim inside
    # \texttt{} and pdflatex would reject it as "Unicode character ≈ (U+2248)".
    for src, dst in UNICODE:
        text = text.replace(src, dst)

    slots: list[str] = []

    def stash(payload: str) -> str:
        slots.append(payload)
        return "\x00%d\x00" % (len(slots) - 1)

    text = CITE_RE.sub(lambda m: stash(r"\texttt{%s}" % escape(m.group(1))), text)
    # Bold before italic so ** is not eaten by *.
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: stash(r"\textbf{%s}" % escape(m.group(1))), text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)",
                  lambda m: stash(r"\emph{%s}" % escape(m.group(1))), text)

    text = escape(text)
    text = text.replace(r"\$", "$")          # math delimiters we introduced
    # Resolve to a fixed point, not in one pass: a bold span containing a code
    # span stores the inner placeholder inside the outer slot's payload, so a
    # single re.sub leaves the inner one unresolved and a raw NUL reaches the
    # .tex -- which pdflatex reports as "Text line contains an invalid
    # character" at the line of the enclosing phrase, with no hint that a
    # converter placeholder is at fault.
    for _ in range(10):
        if not re.search(r"\x00\d+\x00", text):
            break
        text = re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)
    if "\x00" in text:
        # Never emit a control character silently.
        print("WARNING: unresolved inline placeholder in: %r" % text[:120])
        text = text.replace("\x00", "")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep_row(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def table_to_latex(rows: list[list[str]], caption: str | None, label: str) -> str:
    ncols = max(len(r) for r in rows)
    spec = "l" + "r" * (ncols - 1) if ncols > 1 else "l"
    # Wide tables must span both columns. A 4+ column table inside one column of
    # a two-column layout overflows by hundreds of points and -- worse than ugly
    # -- pdftotext shows the cell contents breaking mid-token, so "3.79e-11"
    # extracts as "3.79" on one line and "e-11" on another. That is a number
    # corrupted by typesetting, which is precisely what the PDF text audit
    # exists to catch (found on the first build, 2026-09-19).
    wide = ncols >= 4
    env = "table*" if wide else "table"
    size = r"\footnotesize" if wide else r"\small"
    body = []
    for i, row in enumerate(rows):
        cells = [inline(c) for c in row] + [""] * (ncols - len(row))
        body.append(" & ".join(cells) + r" \\")
        if i == 0:
            body.append(r"\midrule")
    cap = r"\caption{%s}" % inline(caption) if caption else ""
    return "\n".join([
        r"\begin{%s}[t]" % env,
        r"\centering",
        size,
        cap,
        r"\label{%s}" % label,
        r"\begin{tabular}{%s}" % spec,
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{%s}" % env,
    ])


def flush_table(rows: list[list[str]], out: list[str], caption: str | None,
                counter: list[int]) -> None:
    if not rows:
        return
    counter[0] += 1
    out.append(table_to_latex(rows, caption, "tab:auto%d" % counter[0]))
    out.append("")


def convert(md: str) -> tuple[str, list[str]]:
    lines = md.splitlines()
    out: list[str] = []
    notes: list[str] = []
    i = 0
    in_skip = False
    pending_caption: str | None = None
    table_rows: list[list[str]] = []
    counter = [0]
    list_mode: str | None = None

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            out.append(r"\end{%s}" % list_mode)
            out.append("")
            list_mode = None

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # --- section gating -------------------------------------------------
        if stripped.startswith("## "):
            close_list()
            flush_table(table_rows, out, pending_caption, counter)
            table_rows, pending_caption = [], None
            in_skip = any(stripped.startswith(s) for s in SKIP_SECTIONS)
            if in_skip:
                notes.append("skipped section: %s" % stripped)
            else:
                # Emit the section. Numbering is stripped because LaTeX numbers
                # it; appendix letters are KEPT in the title because the prose
                # cross-references them by letter ("Appendix E.2"), and letting
                # \appendix renumber them would silently invalidate every such
                # reference.
                title = stripped[3:].strip()
                title = re.sub(r"^\d+\.\s*", "", title)
                out.append(r"\section{%s}" % inline(title))
                out.append("")
            i += 1
            continue

        if in_skip:
            i += 1
            continue

        # --- internal notes are dropped by rule -----------------------------
        if stripped.startswith(">") and "internal" in stripped.lower():
            notes.append("dropped internal note at source line %d" % (i + 1))
            while i < len(lines) and lines[i].strip().startswith(">"):
                i += 1
            continue

        # --- abstract and headings -----------------------------------------
        if stripped.startswith("# Small Paper Draft") or stripped == "---":
            i += 1
            continue
        if stripped.startswith("### "):
            close_list()
            flush_table(table_rows, out, pending_caption, counter)
            table_rows, pending_caption = [], None
            sub = re.sub(r"^\d+(\.\d+)?\s*", "", stripped[4:].strip())
            out.append(r"\subsection{%s}" % inline(sub))
            out.append("")
            i += 1
            continue
        if stripped.startswith("## "):
            i += 1
            continue

        # --- tables ---------------------------------------------------------
        if stripped.startswith("|"):
            if is_sep_row(stripped):
                i += 1
                continue
            table_rows.append(split_row(stripped))
            # A caption is the last non-empty prose line before the table.
            j = len(out) - 1
            while j >= 0 and out[j] == "":
                j -= 1
            if pending_caption is None and j >= 0:
                cand = out[j]
                if cand.startswith(("**Table", "**Measured", "**Same", "*(", "**Pair")):
                    pending_caption = cand
                    out[j] = ""
            i += 1
            continue
        flush_table(table_rows, out, pending_caption, counter)
        table_rows, pending_caption = [], None

        # --- lists ----------------------------------------------------------
        if re.match(r"^[-*] ", stripped):
            if list_mode != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                list_mode = "itemize"
            out.append(r"  \item %s" % inline(stripped[2:]))
            i += 1
            continue
        if re.match(r"^\d+\. ", stripped):
            if list_mode != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            out.append(r"  \item %s" % inline(re.sub(r"^\d+\. ", "", stripped)))
            i += 1
            continue
        if re.match(r"^\s+[-*] ", raw) and list_mode:
            out.append(r"  \begin{itemize}")
            out.append(r"    \item %s" % inline(stripped[2:]))
            i += 1
            continue

        if stripped == "":
            close_list()
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        # --- ordinary paragraph --------------------------------------------
        close_list()
        if stripped.startswith(">"):
            body = stripped.lstrip("> ").strip()
            if body:
                out.append(inline(body))
                out.append("")
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue
        out.append(inline(stripped))
        out.append("")
        i += 1

    close_list()
    flush_table(table_rows, out, pending_caption, counter)
    return "\n".join(out), notes


def main() -> int:
    md = io.open(SRC, encoding="utf-8").read()

    # The abstract is its own draft section; take it verbatim.
    m = re.search(r"## Abstract[^\n]*\n\n(.*?)\n\n>", md, re.S)
    abstract = inline(m.group(1)) if m else "ABSTRACT NOT FOUND"

    # Body = everything from the Introduction on.
    start = md.index("## 1. Introduction")
    body_md = md[start:]
    body_tex, notes = convert(body_md)

    tex = f"""%% Generated by scripts/md_to_aamas_tex.py -- edit paper/small_paper_draft.md
%% and regenerate. Do not edit this file by hand; the Markdown is the source of
%% truth that audit_paper_numbers.py guards.
\\documentclass[sigconf,anonymous]{{aamas}}

\\usepackage{{balance}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
%% aamas.cls (like acmart) already defines \\Bbbk; amssymb redefines it and
%% aborts the build. \\let\\Bbbk\\relax is the workaround the official AAMAS
%% template uses, so keep it between the two packages.
\\let\\Bbbk\\relax
\\usepackage{{amssymb}}
\\usepackage{{stfloats}}
\\usepackage{{url}}

\\setcopyright{{ifaamas}}
\\acmConference[AAMAS '27]{{Proc.\\@ of the 26th International Conference on
  Autonomous Agents and Multiagent Systems (AAMAS 2027)}}{{May 2027}}{{}}{{ }}
\\copyrightyear{{2027}}
\\acmYear{{2027}}
\\acmDOI{{}}
\\acmPrice{{}}
\\acmISBN{{}}
\\acmSubmissionID{{000}}

\\begin{{document}}

\\title[{TITLE}]{TITLE}

\\author{{Anonymous}}
\\affiliation{{\\institution{{Anonymous submission}}\\city{{}}\\country{{}}}}
\\email{{anon@example.com}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\keywords{{{KEYWORDS}}}

\\maketitle

{body_tex}

\\balance
\\end{{document}}
"""

    io.open(OUT, "w", encoding="utf-8").write(tex)
    print("wrote %s (%d lines, %d chars)" % (OUT, tex.count("\n") + 1, len(tex)))
    for n in notes:
        print("  " + n)

    # Fail loudly on any character pdflatex cannot typeset, and list ALL of them
    # at once. Without this the build surfaces one symbol per compile, which
    # turns a two-minute fix into a dozen compile cycles -- and worse, a symbol
    # could be silently dropped and reach the PDF as a gap nobody notices.
    leftover: dict[str, int] = {}
    for ch in tex:
        if ord(ch) > 127:
            leftover[ch] = leftover.get(ch, 0) + 1
    if leftover:
        print()
        print("UNMAPPED NON-ASCII CHARACTERS -- add these to UNICODE and re-run:")
        for ch, n in sorted(leftover.items(), key=lambda kv: -kv[1]):
            print("  U+%04X  %r  x%d" % (ord(ch), ch, n))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
