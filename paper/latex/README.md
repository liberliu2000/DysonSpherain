# DysonSpherain LaTeX Paper Bundle

This directory contains the LaTeX conversion of the current paper draft.

## Contents

- `main.tex`: main LaTeX manuscript.
- `tables/`: compact artifact-backed paper tables.
- `figures/`: completed TikZ/PGFPlots figures.
- `image2_prompts.md`: optional prompts for raster replacements if a submission
  workflow requires image files instead of TikZ figures.

## Build

This machine currently does not have `pdflatex`, `xelatex`, or `latexmk`
installed. On a machine with TeX Live or MacTeX:

```bash
cd paper/latex
pdflatex main.tex
pdflatex main.tex
```

The figure sources are embedded through `\input{figures/*.tex}` and do not need
external image files.
