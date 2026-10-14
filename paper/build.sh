#!/usr/bin/env bash
# Build manuscript.pdf from manuscript.tex
#
# WHY THIS SCRIPT EXISTS
#   The TeX distribution on this machine lives in a NON-STANDARD location:
#       ~/Library/TinyTeX/bin/universal-darwin/
#   It is NOT on PATH, and NOT at /Library/TeX/texbin or /usr/local/texlive.
#   So a bare `pdflatex` / `which pdflatex` fails, and it is easy to wrongly
#   conclude "there is no LaTeX on this machine". pdflatex.fmt IS present
#   (~/Library/TinyTeX/texmf-var/web2c/pdftex/pdflatex.fmt, ~2.25 MB).
#   Do NOT use tectonic: its bundle is not downloadable from this sandbox.
#
# USAGE
#   cd paper && ./build.sh            # -> manuscript.pdf in this directory
#
set -euo pipefail

export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$(dirname "$0")"

# latexmk handles the bibtex + multi-pass rerun sequence automatically.
latexmk -pdf -bibtex -interaction=nonstopmode -outdir=build manuscript.tex

cp build/manuscript.pdf ./manuscript.pdf

# Fail loudly on unresolved citations so a broken ref list never ships.
if grep -q "Citation.*undefined\|Reference.*undefined" build/manuscript.log; then
  echo "ERROR: undefined citations/references -- check references.bib" >&2
  grep "undefined" build/manuscript.log | head -20 >&2
  exit 1
fi

echo "OK: manuscript.pdf"
grep "Output written" build/manuscript.log | tail -1
