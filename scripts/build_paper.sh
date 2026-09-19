#!/usr/bin/env bash
# Build the AAMAS sources and audit the RESULT, not just the source.
#
# RUN IN THE WSL UBUNTU SHELL:
#     bash /home/sean/jsbsim-marl-formation/scripts/build_paper.sh
#     bash /home/sean/jsbsim-marl-formation/scripts/build_paper.sh path/to/one.tex
#
# With no arguments it builds both generated documents:
#     paper/aamas_paper.tex       the submission (8-page limit applies)
#     paper/aamas_supplement.tex  appendices + provenance (no page limit)
#
# Why a script rather than one latex command: a paper with cross-references,
# tables and figures needs several passes, and the check that matters happens
# AFTER the PDF exists. This does the passes, reports pages and overflow, then
# extracts the PDF text and runs the number audit against THAT -- so a value
# which survives in the source but is mangled by typesetting is caught.
#
# latexmk is used when present, otherwise pdflatex runs three times.

set -euo pipefail

REPO="/home/sean/jsbsim-marl-formation"
TEMPLATE_DIR="/home/sean/aamas_template"
BUILD_ROOT="$REPO/paper/build"
LIMIT=8   # official AAMAS main-track limit: 8 pages, references excluded

# The project interpreter, by absolute path: `python` is not on PATH in this WSL
# shell, and a bare `python` once skipped this whole check silently -- the one
# step whose entire purpose is to not be skipped.
PY="/home/sean/miniconda3/envs/marl_env/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

TEXES=("$@")
if [ ${#TEXES[@]} -eq 0 ]; then
  TEXES=("$REPO/paper/aamas_paper.tex" "$REPO/paper/aamas_supplement.tex")
fi

build_one() {
  local TEX="$1"
  if [ ! -f "$TEX" ]; then
    echo "skip (not generated): $TEX"
    return 0
  fi
  local BASE BUILD_DIR
  BASE="$(basename "$TEX" .tex)"
  BUILD_DIR="$BUILD_ROOT/$BASE"
  mkdir -p "$BUILD_DIR"

  # The class file has to sit next to the document (or be on TEXINPUTS).
  if [ -f "$TEMPLATE_DIR/aamas.cls" ]; then
    cp -f "$TEMPLATE_DIR/aamas.cls" "$BUILD_DIR/aamas.cls"
  else
    echo "warning: $TEMPLATE_DIR/aamas.cls not found; the build will fail unless"
    echo "         the class is already on TEXINPUTS."
  fi
  # Figures are referenced relative to the .tex.
  cp -f "$REPO"/results/shoot_eval/framework_fig1.pdf   "$BUILD_DIR/" 2>/dev/null || true
  cp -f "$REPO"/results/shoot_eval/robustness_fig3.pdf   "$BUILD_DIR/" 2>/dev/null || true
  cp -f "$REPO"/results/shoot_eval/mechanism_seed20007_d00.png "$BUILD_DIR/" 2>/dev/null || true

  cp -f "$TEX" "$BUILD_DIR/$BASE.tex"
  cd "$BUILD_DIR"

  echo "=== compiling $BASE ==="
  if command -v latexmk >/dev/null 2>&1; then
    echo "  driver: latexmk"
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$BASE.tex" > latexmk.log 2>&1 || {
      echo "  BUILD FAILED. Errors:"; grep -nE '^!|^l\.[0-9]+' latexmk.log | head -n 25; return 1; }
  else
    echo "  driver: pdflatex x3 (latexmk not installed)"
    for pass in 1 2 3; do
      if ! pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" > "pass$pass.log" 2>&1; then
        echo "  BUILD FAILED on pass $pass. Errors:"
        grep -nE '^!|^l\.[0-9]+' "pass$pass.log" | head -n 25
        return 1
      fi
    done
  fi

  [ -f "$BASE.pdf" ] || { echo "  no PDF produced"; return 1; }

  echo
  echo "=== result: $BASE ==="
  local PAGES
  PAGES="$(pdfinfo "$BASE.pdf" | awk '/^Pages:/ {print $2}')"
  echo "  pdf        : $BUILD_DIR/$BASE.pdf ($(stat -c%s "$BASE.pdf") bytes)"
  if [[ "$BASE" == *supplement* ]]; then
    echo "  pages      : $PAGES  (supplementary material: no page limit)"
  else
    echo "  pages      : $PAGES  (AAMAS main-track limit: $LIMIT, references excluded)"
    if [ "$PAGES" -gt "$LIMIT" ]; then
      echo "  OVER BUDGET by $((PAGES - LIMIT)) page(s)"
    else
      echo "  within budget"
    fi
  fi

  # Overfull boxes are the honest signal of what will look wrong in two columns.
  local OVER
  OVER="$(grep -c 'Overfull \\hbox' "$BASE.log" 2>/dev/null || true)"
  echo "  overfull   : ${OVER:-0} hbox warning(s)"

  echo
  echo "=== auditing the extracted PDF text (not the source) ==="
  # Two extractions, two jobs. Plain reading order keeps a number contiguous
  # when it wraps at a column edge, which is what the audit needs. -layout keeps
  # the visual grid -- by putting BOTH columns on one text line, which is fine
  # for eyeballing a table and useless for auditing.
  pdftotext "$BASE.pdf" "$BASE.txt"
  pdftotext -layout "$BASE.pdf" "$BASE.layout.txt"
  "$PY" "$REPO/scripts/audit_paper_numbers.py" "$BUILD_DIR/$BASE.txt"
  echo
}

status=0
TEXTS=()
for tex in "${TEXES[@]}"; do
  build_one "$tex" || status=1
  base="$(basename "$tex" .tex)"
  [ -f "$BUILD_ROOT/$base/$base.txt" ] && TEXTS+=("$BUILD_ROOT/$base/$base.txt")
done

# The number audit runs against the WHOLE SUBMISSION, not per document. A
# headline value may legitimately live in the supplement (A5's 99.94% does),
# while a per-document audit -- which is what the first two-document build did
# on 2026-09-19 -- reports it MISSING from the paper and half the body's numbers
# MISSING from the supplement. Concatenating preserves what the check is for:
# the value must survive typesetting somewhere in what we actually submit.
if [ ${#TEXTS[@]} -gt 1 ]; then
  COMBINED="$BUILD_ROOT/submission_combined.txt"
  cat "${TEXTS[@]}" > "$COMBINED"
  echo "=== auditing the COMBINED submission text (paper + supplement) ==="
  echo "  input: ${TEXTS[*]}"
  "$PY" "$REPO/scripts/audit_paper_numbers.py" "$COMBINED" || status=1
  echo
fi

echo "done. PDFs and logs are under $BUILD_ROOT/"
exit $status
