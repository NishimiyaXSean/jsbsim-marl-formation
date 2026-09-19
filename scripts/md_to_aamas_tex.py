"""Convert paper/small_paper_draft.md into AAMAS LaTeX sources.

Produces TWO documents, and the split is the point:

    paper/aamas_paper.tex       main paper: sections 1-5, the causal chain only
    paper/aamas_supplement.tex  appendices B-E plus the provenance records

WHY A CONVERTER RATHER THAN A HAND-TRANSCRIPTION
------------------------------------------------
The Markdown draft is the single source of truth: audit_paper_numbers.py guards
the numbers in *it*, and the prose is edited there. Hand-copying it into .tex
would create a second copy that silently drifts -- which is the exact failure
this project has already been bitten by twice (the 43.2/91.2 pair, and the two
CLR diagnostic rows labelled with the wrong geometry).

WHAT GOES WHERE, AND WHY
------------------------
The first AAMAS build (2026-09-19) put internal audit material in the submitted
PDF: artifact provenance blockquotes, the ablation status tracker, superseded-
number notes. That material is what makes the work trustworthy, and it belongs in
supplementary material, not in the paper the reviewers read. So the converter
routes it rather than leaving it to discipline:

  main        sections 1-5, tables, figures, no appendix
  supplement  appendix sections, plus every provenance record collected out of
              the body, under one heading

The main text keeps a single sentence pointing at the supplement, so a reader who
wants to check a number knows where it lives.

Markdown constructs handled: ATX headings, paragraphs, bullet and numbered
lists, bold/italic/code spans, pipe tables (booktabs), blockquotes, and the
Unicode symbols actually used in this draft. It FAILS, listing all of them at
once, on any character pdflatex cannot typeset -- otherwise a symbol is either
dropped silently or surfaced one per compile cycle.

NOT DONE HERE
-------------
No citation processing: the draft cites inline, so the baseline has no
bibliography. Converting to \\cite{} plus a .bib is a separate step, and the
order matters -- BibTeX changes reference length and page breaks, so it comes
after the page flow is settled.
"""

from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))
SRC = os.path.join(ROOT, "paper", "small_paper_draft.md")
OUT_MAIN = os.path.join(ROOT, "paper", "aamas_paper.tex")
OUT_SUPP = os.path.join(ROOT, "paper", "aamas_supplement.tex")

TITLE = ("Diagnosing and Surgically Correcting Conservative Decision Bias "
         "in Imitation-Learned Air Combat Policies")

KEYWORDS = ("imitation learning, behaviour cloning, expert-induced bias, "
            "causal identification, policy intervention, air combat")

# Top-level draft sections that never reach either document: project bookkeeping.
SKIP_SECTIONS = (
    "## 6.", "## 7.", "## 8.", "## 9.",
    "## Title Candidates", "## Appendix A.",
)

# A blockquote opening with one of these is an audit/provenance record: it is
# collected out of the body and re-emitted in the supplement. "Citation status"
# is on the list because the verified venue/page record is exactly the material
# a .bib needs, and a reader of the paper must see [1] Pomerleau -- not the
# authors' account of having previously got the citation wrong.
PROVENANCE_PREFIXES = ("Artifact status", "**Artifact status", "Provenance",
                       "Citation status", "**Citation status")

# Unicode seen in this draft -> LaTeX. Multi-character forms first, so that
# "chi^2" written with a superscript is not split into two separate objects.
UNICODE = [
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
    ("∼", r"$\sim$"), ("≡", r"$\equiv$"), ("≠", r"$\neq$"),
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
    # Unicode FIRST, before any span is stashed: a symbol inside a code span
    # would otherwise reach pdflatex verbatim inside \texttt{} and be rejected
    # as "Unicode character ... is not set up for use with LaTeX".
    for src, dst in UNICODE:
        text = text.replace(src, dst)

    slots: list[str] = []

    def stash(payload: str) -> str:
        slots.append(payload)
        return "\x00%d\x00" % (len(slots) - 1)

    def corpus(payload: str) -> str:
        """Render a code span, inserting break opportunities in long tokens.

        TeX will not break inside \\texttt by default, and the escaped
        underscore (\\_) is not a break point either, so an artifact path such as
        results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json overflows its column
        by 20-40pt. \\allowbreak after each separator gives TeX legal places to
        break without changing what the reader sees. Only applied to long spans:
        a short \\texttt{a\\_fire} should never be split.
        """
        body = escape(payload)
        if len(payload) > 24:
            body = body.replace(r"\_", r"\_\allowbreak{}")
            body = body.replace("/", r"/\allowbreak{}")
            body = body.replace("-", r"-\allowbreak{}")
        return r"\texttt{%s}" % body

    text = CITE_RE.sub(lambda m: stash(corpus(m.group(1))), text)
    text = re.sub(r"\*\*(.+?)\*\*",
                  lambda m: stash(r"\textbf{%s}" % escape(m.group(1))), text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)",
                  lambda m: stash(r"\emph{%s}" % escape(m.group(1))), text)

    text = escape(text)
    text = text.replace(r"\$", "$")
    # Fixed point, not one pass: a bold span containing a code span stores the
    # inner placeholder inside the outer slot, so a single re.sub leaves a raw
    # NUL in the .tex -- reported by pdflatex only as "invalid character" on the
    # enclosing phrase's line.
    for _ in range(10):
        if not re.search(r"\x00\d+\x00", text):
            break
        text = re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)
    if "\x00" in text:
        print("WARNING: unresolved inline placeholder in: %r" % text[:120])
        text = text.replace("\x00", "")
    return re.sub(r"\s{2,}", " ", text).strip()


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep_row(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


# Approximate AAMAS/acmart widths in points: one column, and the full text
# block. Used to decide whether a table fits in a column or must span both.
COL_PT = 239.0
TEXT_PT = 505.0
# Mean glyph advance. Measured against the first working build rather than
# assumed for \footnotesize: these tables are set in \small, which is wider.
CHAR_PT = 4.9
# A fixed (l/r) column never wraps, so it costs its longest cell; headers are
# short in practice, and capping stops a single long header from starving the
# wrapping columns.
FIXED_CAP_CHARS = 26


def col_spec(rows: list[list[str]]) -> tuple[bool, str]:
    """Decide the float width and the column types for one table.

    Returns (wide, spec), where wide means "span both columns".

    Two guesses were tried and both overflowed. Plain "l" columns never wrap, so
    a three-column table whose first cell holds a sentence overflowed by 581pt.
    Computing p{} fractions by hand then asked for 1.38x the available width,
    and the "fix" silently rescaled the fractions back to the share it had
    computed -- so the clamp meant nothing and a five-column table got 0.70 of
    the text width for its first column. tabularx was worse still: an X column
    cannot shrink below the width of its widest unbreakable word, so it reports
    the whole table overfull (74 of them).

    So the width is now treated as a constraint rather than a guess: estimate
    the natural width, promote the table to full width if it needs it, and wrap
    only the columns that are long, sizing them so the columns add up to what is
    actually available.
    """
    ncols = max(len(r) for r in rows)
    if ncols == 1:
        return False, "l"
    widths = [max((len(r[c]) if c < len(r) else 0) for r in rows)
              for c in range(ncols)]

    pad = 8.0 * ncols                      # \tabcolsep 4pt on both sides
    natural = sum(w * CHAR_PT for w in widths) + pad
    # A compact table stays a plain tabular -- but the test has to be the TOTAL
    # width, not the longest cell. Testing only the longest cell sent the CLR
    # table (four columns summing to ~264pt in a 239pt column) and the E6 table
    # into single-column tabulars, where they overflowed by 34pt and 86pt.
    if natural <= 0.95 * COL_PT:
        return False, "l" + "r" * (ncols - 1)

    wide = natural > COL_PT
    unit = TEXT_PT if wide else COL_PT
    fixed = [c for c in range(ncols) if widths[c] <= 30]
    wrap = [c for c in range(ncols) if widths[c] > 30]
    # Nothing needs to wrap yet: the table fits once it has the full width.
    if not wrap:
        return wide, "l" + "r" * (ncols - 1)

    fixed_pt = sum(min(widths[c], FIXED_CAP_CHARS) * CHAR_PT for c in fixed) + pad
    avail = unit - fixed_pt
    if avail < 90 and not wide:
        # Nothing left for the wrapping columns: give the table the full width.
        wide = True
        unit = TEXT_PT
        avail = unit - fixed_pt
    if avail < 90:
        avail = max(90.0, unit - pad)      # last resort; overfull is reported
        fixed = [c for c in range(ncols)]  # treat nothing as fixed

    total = float(sum(widths[c] for c in wrap)) or 1.0
    fracs = {c: avail * widths[c] / total for c in wrap}

    spec = []
    for c in range(ncols):
        if c in fracs:
            share = max(0.12, fracs[c] / unit)
            spec.append("p{%.2f%s}" % (share, r"\textwidth" if wide
                                       else r"\linewidth"))
        else:
            spec.append("l" if c == 0 else "r")
    return wide, "".join(spec)


def table_to_latex(rows: list[list[str]], caption: str | None, label: str) -> str:
    ncols = max(len(r) for r in rows)
    # Wide tables span both columns. pdftotext shows an over-wide cell breaking
    # mid-token, so "3.79e-11" extracts as "3.79" then "e-11" -- a number
    # corrupted by typesetting, which is what the PDF text audit is for.
    wide, spec = col_spec(rows)
    env = "table*" if wide else "table"
    body = []
    for i, row in enumerate(rows):
        cells = [inline(c) for c in row] + [""] * (ncols - len(row))
        body.append(" & ".join(cells) + r" \\")
        if i == 0:
            body.append(r"\midrule")
    cap = r"\caption{%s}" % inline(caption) if caption else ""
    return "\n".join([
        r"\begin{%s}[t]" % env, r"\centering",
        r"\small",
        # Tabular columns separated by 6pt on each side cost 12pt per column;
        # 4pt is what Sean suggested and buys back ~24pt on a 4-column table
        # without looking cramped.
        r"\setlength{\tabcolsep}{4pt}",
        cap, r"\label{%s}" % label,
        r"\begin{tabular}{%s}" % spec, r"\toprule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{%s}" % env,
    ])


def convert(md: str) -> tuple[str, str, list[str], list[str]]:
    """Return (body_tex, appendix_tex, provenance_records, notes)."""
    lines = md.splitlines()
    sinks = {"body": [], "appendix": []}
    sink = sinks["body"]
    notes: list[str] = []
    provenance: list[str] = []
    i = 0
    in_skip = False
    in_appendix = False
    pending_caption: str | None = None
    table_rows: list[list[str]] = []
    drop_table = False
    counter = [0]
    list_mode: str | None = None

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            sink.append(r"\end{%s}" % list_mode)
            sink.append("")
            list_mode = None

    def emit_table() -> None:
        nonlocal table_rows, pending_caption, drop_table
        if table_rows and not drop_table:
            counter[0] += 1
            sink.append(table_to_latex(table_rows, pending_caption,
                                       "tab:auto%d" % counter[0]))
            sink.append("")
        table_rows, pending_caption, drop_table = [], None, False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # --- section gating and sink switching ------------------------------
        if stripped.startswith("## "):
            close_list()
            emit_table()
            in_skip = any(stripped.startswith(s) for s in SKIP_SECTIONS)
            if stripped.startswith("## Appendix") and not in_skip:
                in_appendix = True
                sink = sinks["appendix"]
            if in_skip:
                notes.append("skipped section: %s" % stripped)
            else:
                title = re.sub(r"^\d+\.\s*", "", stripped[3:].strip())
                # Appendix letters stay in the title: the prose cross-references
                # them by letter ("Appendix E.2"), so renumbering would silently
                # invalidate every such reference.
                sink.append(r"\section{%s}" % inline(title))
                sink.append("")
            i += 1
            continue

        if in_skip:
            i += 1
            continue

        # --- internal notes and provenance records --------------------------
        if stripped.startswith(">"):
            # A contiguous run of ">" lines can hold MORE THAN ONE paragraph,
            # and they can have different fates: the citation-verification record
            # is worth keeping (it is the raw material for the .bib) while the
            # revision log one line below it must be dropped. Treating the run as
            # a single block discarded both as soon as any line mentioned
            # "internal" -- caught 2026-09-19, when the Citation status record
            # vanished from both output documents.
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip("> ").strip())
                i += 1
            paragraph: list[str] = []
            for piece in block + [""]:
                if piece:
                    paragraph.append(piece)
                    continue
                if not paragraph:
                    continue
                joined = " ".join(paragraph)
                paragraph = []
                if "internal" in joined.lower():
                    notes.append("dropped an internal note")
                elif in_appendix:
                    sink.append(inline(joined))
                    sink.append("")
                elif joined.lstrip("*").startswith(PROVENANCE_PREFIXES):
                    provenance.append(joined)
                    notes.append("routed a provenance record to the supplement")
                else:
                    sink.append(inline(joined))
                    sink.append("")
            continue

        if stripped.startswith("# Small Paper Draft") or stripped == "---":
            i += 1
            continue
        if stripped.startswith("### "):
            close_list()
            # Strip "3.1", "4.3b" and similar draft numbering: LaTeX numbers the
            # heading itself, and leaving the draft's own label in produces
            # "b Figure 2 --- the mechanism" in the PDF.
            sub = re.sub(r"^\d+(\.\d+)*[a-z]?\s*", "", stripped[4:].strip())
            sink.append(r"\subsection{%s}" % inline(sub))
            sink.append("")
            i += 1
            continue

        # --- tables ----------------------------------------------------------
        if stripped.startswith("|"):
            if is_sep_row(stripped):
                i += 1
                continue
            row = split_row(stripped)
            if any("internal" in c.lower() for c in row):
                # An internal tracker row means the whole table is bookkeeping.
                drop_table = True
                notes.append("dropped an internal status table")
            else:
                table_rows.append(row)
            j = len(sink) - 1
            while j >= 0 and sink[j] == "":
                j -= 1
            if pending_caption is None and not drop_table and j >= 0:
                cand = sink[j]
                if cand.startswith(("**Table", "**Measured", "**Same", "*(",
                                    "**Pair")):
                    pending_caption = cand
                    sink[j] = ""
            i += 1
            continue
        emit_table()

        # --- lists -----------------------------------------------------------
        if re.match(r"^[-*] ", stripped):
            if list_mode != "itemize":
                close_list()
                sink.append(r"\begin{itemize}")
                list_mode = "itemize"
            sink.append(r"  \item %s" % inline(stripped[2:]))
            i += 1
            continue
        if re.match(r"^\d+\. ", stripped):
            if list_mode != "enumerate":
                close_list()
                sink.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            sink.append(r"  \item %s" % inline(re.sub(r"^\d+\. ", "", stripped)))
            i += 1
            continue
        if re.match(r"^\s+[-*] ", line) and list_mode:
            sink.append(r"  \begin{itemize}")
            sink.append(r"    \item %s" % inline(stripped[2:]))
            i += 1
            continue

        if stripped == "":
            close_list()
            if sink and sink[-1] != "":
                sink.append("")
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue

        close_list()
        sink.append(inline(stripped))
        sink.append("")
        i += 1

    close_list()
    emit_table()
    return ("\n".join(sinks["body"]), "\n".join(sinks["appendix"]),
            provenance, notes)


PREAMBLE = r"""%% Generated by scripts/md_to_aamas_tex.py -- edit paper/small_paper_draft.md
%% and regenerate. Do not edit this file by hand: the Markdown is the source of
%% truth that audit_paper_numbers.py guards.
\documentclass[sigconf,anonymous]{{aamas}}

\usepackage{{balance}}
\usepackage{{booktabs}}
%% tabularx gives one column type X and lets LaTeX split whatever space the
%% fixed columns leave. Two hand-rolled attempts to compute column widths from
%% character counts overflowed; this removes the guessing entirely.
\usepackage{{tabularx}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
%% aamas.cls (like acmart) already defines \Bbbk; amssymb redefines it and
%% aborts the build. \let\Bbbk\relax is the workaround the official template
%% uses, so keep it between the two packages.
\let\Bbbk\relax
\usepackage{{amssymb}}
\usepackage{{stfloats}}
\usepackage{{url}}
%% artifact paths, checkpoint hashes and CLI flags are typeset in \texttt, and
%% TeX will not break a line inside \texttt by default: a single
%% "results/shoot_eval/E1_fire_oracle_dist2_3k_s60.json" then overflows its
%% column by 30-210pt. [htt] lets those tokens hyphenate, which fixes the whole
%% class of overfull boxes at once instead of shortening each path by hand.
\usepackage[htt]{{hyphenat}}

\setcopyright{{ifaamas}}
\acmConference[AAMAS '27]{{Proc.\@ of the 26th International Conference on
  Autonomous Agents and Multiagent Systems (AAMAS 2027)}}{{May 2027}}{{}}{{ }}
\copyrightyear{{2027}}
\acmYear{{2027}}
\acmDOI{{}}
\acmPrice{{}}
\acmISBN{{}}
\acmSubmissionID{{000}}

\begin{{document}}

\title[{title}]{{{title}}}

\author{{Anonymous}}
\affiliation{{\institution{{Anonymous submission}}\city{{}}\country{{}}}}
\email{{anon@example.com}}
"""


def build_main(abstract: str, body: str) -> str:
    return PREAMBLE.format(title=TITLE) + f"""
\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\keywords{{{KEYWORDS}}}

\\maketitle

{body}

\\balance
\\end{{document}}
"""


def build_supplement(appendix: str, provenance: list[str]) -> str:
    supp_title = "Supplementary Material: " + TITLE
    records = "\n\n".join(inline(p) for p in provenance) if provenance else \
        "No provenance records were collected from the body."
    return PREAMBLE.format(title=supp_title) + f"""
\\begin{{abstract}}
Supplementary material for the paper above. Section E records the provenance of
every number quoted in the main text; nothing here is required in order to
follow the argument.
\\end{{abstract}}

\\keywords{{reproducibility, provenance}}

\\maketitle

{appendix}

\\section{{Appendix E.6: Artifact provenance records}}
The records below were collected out of the main text, where they interrupted the
argument. Each states which artifact backs which claim.

{records}

\\balance
\\end{{document}}
"""


def main() -> int:
    md = io.open(SRC, encoding="utf-8").read()

    m = re.search(r"## Abstract[^\n]*\n\n(.*?)\n\n>", md, re.S)
    abstract = inline(m.group(1)) if m else "ABSTRACT NOT FOUND"

    body, appendix, provenance, notes = convert(md[md.index("## 1. Introduction"):])
    main_tex = build_main(abstract, body)
    supp_tex = build_supplement(appendix, provenance)

    io.open(OUT_MAIN, "w", encoding="utf-8").write(main_tex)
    io.open(OUT_SUPP, "w", encoding="utf-8").write(supp_tex)
    print("wrote %s (%d lines)" % (OUT_MAIN, main_tex.count("\n") + 1))
    print("wrote %s (%d lines)" % (OUT_SUPP, supp_tex.count("\n") + 1))
    for n in notes:
        print("  " + n)

    # Fail loudly on any character pdflatex cannot typeset, listing ALL of them:
    # otherwise a symbol is dropped silently, or surfaced one per compile cycle.
    leftover: dict[str, int] = {}
    for ch in main_tex + supp_tex:
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
