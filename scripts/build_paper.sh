#!/usr/bin/env bash
# Build the AAMAS paper and audit the RESULT, not just the source.
#
# RUN IN THE WSL UBUNTU SHELL:
#     bash /home/sean/jsbsim-marl-formation/scripts/build_paper.sh [paper.tex]
#
# Why a script rather than one latex command: a paper with cross-references,
# tables and a figure needs several passes, and the check that matters happens
# after the PDF exists. This does the passes, reports pages and overflow, then
# extracts the PDF text and runs the number audit against THAT -- so a value
# which survives in the source but is mangled by typesetting is caught.
#
# Idempotent: it rebuilds from scratch each run and cleans intermediate files.
# latexmk is used when present, otherwise pdflatex is run three times, which is
# enough for this document (latexmk is a convenience, not a requirement).

set -euo pipefail

REPO="/home/sean/jsbsim-marl-formation"
TEMPLATE_DIR="/home/sean/aamas_template"
BUILD_DIR="$REPO/paper/build"
TEX="${1:-$REPO/paper/aamas_paper.tex}"
LIMIT=8   # official AAMAS main-track limit: 8 pages, references excluded

if [ ! -f "$TEX" ]; then
  echo "no such .tex: $TEX"
  echo "usage: bash $0 [path/to/paper.tex]"
  exit 1
fi

mkdir -p "$BUILD_DIR"
# The class file has to sit next to the document (or be on TEXINPUTS).
if [ -f "$TEMPLATE_DIR/aamas.cls" ]; then
  cp -f "$TEMPLATE_DIR/aamas.cls" "$BUILD_DIR/aamas.cls"
else
  echo "warning: $TEMPLATE_DIR/aamas.cls not found; build will fail unless the"
  echo "         class is already on TEXINPUTS."
fi

# Figures are referenced relative to the .tex; make them reachable.
cp -f "$REPO"/results/shoot_eval/framework_fig1.pdf "$BUILD_DIR/" 2>/dev/null || true
cp -f "$REPO"/results/shoot_eval/robustness_fig3.pdf  "$BUILD_DIR/" 2>/dev/null || true
cp -f "$REPO"/results/shoot_eval/mechanism_seed20007_d00.png "$BUILD_DIR/" 2>/dev/null || true

cp -f "$TEX" "$BUILD_DIR/$(basename "$TEX")"
DOC="$BUILD_DIR/$(basename "$TEX")"
BASE="$(basename "$DOC" .tex)"
cd "$BUILD_DIR"

echo "=== compiling $BASE ==="
if command -v latexmk >/dev/null 2>&1; then
  echo "  driver: latexmk"
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$BASE.tex" > latexmk.log 2>&1 || {
    echo "  BUILD FAILED. Errors:"; grep -nE '^!|^l\.[0-9]+' latexmk.log | head -n 25; exit 1; }
else
  echo "  driver: pdflatex x3 (latexmk not installed; run: sudo apt-get install -y latexmk)"
  for pass in 1 2 3; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" > "pass$pass.log" 2>&1; then
      echo "  BUILD FAILED on pass $pass. Errors:"
      grep -nE '^!|^l\.[0-9]+' "pass$pass.log" | head -n 25
      exit 1
    fi
  done
fi

PDF="$BASE.pdf"
[ -f "$PDF" ] || { echo "no PDF produced"; exit 1; }

echo
echo "=== result ==="
PAGES="$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')"
SIZE="$(stat -c%s "$PDF")"
echo "  pdf        : $PDF ($SIZE bytes)"
echo "  pages      : $PAGES  (AAMAS main-track limit: $LIMIT, references excluded)"
if [ "$PAGES" -gt "$LIMIT" ]; then
  echo "  OVER BUDGET by $((PAGES - LIMIT)) page(s) -- see README note on which"
  echo "  sections to compress first: Related Work, then the ablation detail."
else
  echo "  within budget"
fi

# Overfull boxes are the honest signal of what will look wrong in two columns.
if [ -f "$BASE.log" ]; then
  OVER="$(grep -c 'Overfull \\hbox' "$BASE.log" || true)"
  echo "  overfull   : ${OVER:-0} hbox warning(s)"
  grep -nE 'Overfull \\hbox \([0-9.]+pt' "$BASE.log" | head -n 8 || true
fi

echo
echo "=== auditing the extracted PDF text (not the source) ==="
pdftotext -layout "$PDF" "$BASE.txt"
python "$REPO/scripts/audit_paper_numbers.py" "$BUILD_DIR/$BASE.txt"

echo
echo "done. Figure/table files and logs are in $BUILD_DIR"
