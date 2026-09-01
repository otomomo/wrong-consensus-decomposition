#!/usr/bin/env bash
# Compile the paper.
# Usage: ./compile.sh   (requires a working LaTeX: pdflatex/tectonic/latexmk)
set -euo pipefail
cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main || true
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
else
  echo "No LaTeX engine found. Install latexmk/pdflatex/tectonic, or compile on Overleaf." >&2
  exit 1
fi
echo "OK: paper/main.pdf"
