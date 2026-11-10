#!/usr/bin/env bash
# Builds the manuals and the guides: their figures, the PDFs and the Markdown, into docs/.
#
#     docs/manual/build.sh                 # everything
#     docs/manual/build.sh contact_dynamics
#
# The guides' tables of tuned scenes are tables/*.tex, written by make_tables.py from
# tools/performance_tuning/tuned/ (no GPU); run it after re-tuning.
#
# Figures: every fig/*.tex is a standalone TikZ or pgfplots picture, compiled to fig/NAME.pdf
# (which the manuals include) and rendered to fig/NAME.png (which the Markdown shows). The
# data some of them plot is in data/, written by make_data.py, which needs the GPU; the
# figures need only LaTeX and make_figures.py. Documents: three pdflatex passes for the references, then
# tex2md.py reads the aux file to number the Markdown as the PDF is numbered.
set -euo pipefail
cd "$(dirname "$0")"
P=${PYTHON:-$(cd ../.. && pwd)/.venv/bin/python}
DOCS=${*:-installation_guide development_guide collision_detection contact_dynamics creating_new_sim_from_mjcf performance_tuning speedup_analysis visualizer reading_guide}
mkdir -p build fig

# The data figures, when their script or their data changed (make_figures.py needs no GPU).
if [ ! -e fig/fig_slope.pdf ] || [ make_figures.py -nt fig/fig_slope.pdf ] || [ -n "$(find data -newer fig/fig_slope.pdf)" ]; then
  $P make_figures.py
fi

for f in fig/*.tex; do
  [ "$(basename "$f")" = common.tex ] && continue
  n=$(basename "$f" .tex)
  if [ ! -e "fig/$n.pdf" ] || [ "$f" -nt "fig/$n.pdf" ] || [ fig/common.tex -nt "fig/$n.pdf" ] \
     || [ -n "$(find data -newer "fig/$n.pdf" -name "${n#fig_}*" 2>/dev/null)" ]; then
    if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$f" > "build/$n.buildlog" 2>&1; then
      grep -A5 "^!" "build/$n.buildlog" | head -30
      echo "figure $n failed: build/$n.buildlog"
      exit 1
    fi
    cp "build/$n.pdf" "fig/$n.pdf"
    pdftoppm -png -r 200 -singlefile "fig/$n.pdf" "fig/$n"
    echo "figure $n"
  fi
done

for doc in $DOCS; do
  for pass in 1 2 3; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$doc.tex" > "build/$doc.buildlog" 2>&1; then
      grep -A5 "^!" "build/$doc.buildlog" | head -30
      echo "$doc failed: build/$doc.buildlog"
      exit 1
    fi
  done
  grep -E "Warning: (Reference|Citation).*undefined|multiply defined" "build/$doc.buildlog" | sort -u || true
  cp "build/$doc.pdf" "../$doc.pdf"
  $P tex2md.py "$doc.tex" "build/$doc.aux" "../$doc.md" --figdir manual/fig
  echo "$doc: $(pdfinfo "../$doc.pdf" | grep Pages | tr -s ' ')"
done
