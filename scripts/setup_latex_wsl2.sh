#!/usr/bin/env bash
# Install the LaTeX toolchain needed to typeset the AAMAS paper.
#
# RUN THIS IN THE WSL UBUNTU SHELL (not cmd.exe, not PowerShell):
#     bash /home/sean/jsbsim-marl-formation/scripts/setup_latex_wsl2.sh
#
# It needs sudo, so you will be asked for your WSL password. Everything else is
# automatic. Idempotent: apt skips what is already installed, so re-running is
# safe and cheap.
#
# Why these packages (2026-09-18):
#   texlive-latex-base/recommended/extra  engine + the packages aamas.cls needs
#                                         (microtype, environ, totpages, booktabs,
#                                         refcount, xstring, xkeyval, etoolbox)
#   texlive-fonts-recommended             base fonts
#   texlive-fonts-extra                   Libertine -- aamas.cls REQUIRES it and
#                                         forbids substituting another typeface
#   texlive-publishers                     acmart family, which aamas.cls derives from
#   latexmk                                convenient multi-pass driver. NOT part of
#                                         any texlive-* package -- omitting it is why
#                                         the 2026-09-19 run stopped at step 3/4.
#   poppler-utils                         pdftotext, for the post-typesetting
#                                         number audit on the extracted PDF text
#
# Expect roughly 1-2 GB of download from the USTC mirror. The script finishes by
# compiling a minimal aamas document, so a green "SMOKE TEST OK" means the
# toolchain, the class file and the Libertine fonts are all genuinely working --
# not merely installed.

set -euo pipefail

REPO="/home/sean/jsbsim-marl-formation"
TEMPLATE_DIR="/home/sean/aamas_template"
PACKAGES=(
  texlive-latex-base
  texlive-latex-recommended
  texlive-latex-extra
  texlive-fonts-recommended
  texlive-fonts-extra
  texlive-publishers
  latexmk
  poppler-utils
)

echo "=== 1/4  apt update ==="
sudo apt-get update

echo
echo "=== 2/4  installing ${#PACKAGES[@]} packages (this is the slow part) ==="
sudo apt-get install -y "${PACKAGES[@]}"

echo
echo "=== 3/4  verifying binaries ==="
# pdflatex and pdftotext are hard requirements. latexmk is a convenience driver
# only: the build can fall back to repeated pdflatex runs, so its absence must
# not abort the installation (it did on 2026-09-19, which is what this branch
# exists to prevent).
for bin in pdflatex pdftotext; do
  if command -v "$bin" >/dev/null 2>&1; then
    printf '  %-10s %s\n' "$bin" "$(command -v "$bin")"
  else
    printf '  %-10s MISSING -- the paper cannot be built without it\n' "$bin"
    printf '  fix: sudo apt-get install -y texlive-latex-base poppler-utils\n'
    exit 1
  fi
done
if command -v latexmk >/dev/null 2>&1; then
  printf '  %-10s %s\n' latexmk "$(command -v latexmk)"
else
  printf '  %-10s missing (optional) -- run: sudo apt-get install -y latexmk\n' latexmk
  printf '  the build will use repeated pdflatex runs instead.\n'
fi
pdflatex --version | head -n 1

echo
echo "=== 4/4  smoke test: compile a minimal aamas document ==="
if [ ! -f "$TEMPLATE_DIR/aamas.cls" ]; then
  echo "  aamas.cls not found in $TEMPLATE_DIR -- skipping the compile test."
  echo "  The toolchain is installed; the class file still needs fetching."
  exit 0
fi

SMOKE="$TEMPLATE_DIR/smoke"
mkdir -p "$SMOKE"
cat > "$SMOKE/smoke.tex" <<'TEX'
\documentclass[sigconf,anonymous]{aamas}
\usepackage{balance}
\usepackage{booktabs}
\setcopyright{ifaamas}
\acmConference[AAMAS '27]{Proc.\@ of the 26th International Conference on
  Autonomous Agents and Multiagent Systems (AAMAS 2027)}{May 2027}{}{}
\copyrightyear{2027}
\acmYear{2027}
\acmDOI{}
\acmPrice{}
\acmISBN{}
\acmSubmissionID{000}
\begin{document}
\title[Smoke test]{Toolchain smoke test}
\author{Anonymous}
\affiliation{\institution{None}\city{None}\country{None}}
\email{none@example.com}
\begin{abstract}
Checks that aamas.cls, its package dependencies and the Libertine fonts all
resolve, and that a two-column body actually compiles.
\end{abstract}
\keywords{smoke test}
\maketitle
\section{Body}
Two columns, one table, one citation-free paragraph.
\begin{table}[t]
\caption{A test caption with a protocol anchor.}
\label{tab:smoke}
\begin{tabular}{lrr}
\toprule
Policy & $d=0$ & $d=0.3$ \\
\midrule
SPC & 90.75\% & 90.75\% \\
\bottomrule
\end{tabular}
\end{table}
\balance
\end{document}
TEX

# Copy the class next to the smoke document so it resolves without a path hack.
cp -f "$TEMPLATE_DIR/aamas.cls" "$SMOKE/aamas.cls"

cd "$SMOKE"
if pdflatex -interaction=nonstopmode -halt-on-error smoke.tex > smoke.log 2>&1; then
  echo "  SMOKE TEST OK -- smoke.pdf built ($(stat -c%s smoke.pdf) bytes)"
  echo "  If the paper build later fails, compare its log against $SMOKE/smoke.log."
else
  echo "  SMOKE TEST FAILED. Last 25 lines of the log:"
  tail -n 25 smoke.log
  exit 1
fi

echo
echo "Done. Re-run any time; already-installed packages are skipped."
