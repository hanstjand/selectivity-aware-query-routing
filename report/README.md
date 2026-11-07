# IEEE Conference Report

`main.pdf` is the final report, *Selectivity-Aware Query Routing for Filtered Vector Search*. It is an 8-page IEEE
conference paper (`IEEEtran`, conference mode) built from the measured results in `../results/full/`.

```
report/
├── main.tex              document root (title, authors, abstract, keywords)
├── references.bib        verified BibTeX entries (+ IEEEtran control entry: >6 authors -> "et al.")
├── sections/             introduction, related_work, proposed_method, experimental_setup,
│                         results, discussion, conclusion
├── figures/              vector PDFs used in the paper (generated, see below)
├── scripts/make_figures.py   regenerates figures/ from ../results/full (no new measurements)
└── main.pdf              compiled paper
```

## Rebuild the PDF

Requirements: a TeX distribution with `IEEEtran`, `tikz`, `algorithms` and `booktabs` (TeX Live 2026 was used).

```bash
cd report
latexmk -pdf main.tex          # runs pdflatex + bibtex as needed -> main.pdf
latexmk -c                     # optional: remove auxiliary files
```

Overleaf: upload the whole `report/` folder (or a zip of it) and compile `main.tex` with pdfLaTeX. The figures are
included as PDFs and the architecture diagram (Fig. 1) is drawn in TikZ inside `sections/proposed_method.tex`, so no
other files are needed.

## Regenerate the figures (optional)

The plotted figures (Figs. 2–5) are redrawn at IEEE column width from the saved results; the measurements are not
changed. From the repository root:

```bash
uv run python report/scripts/make_figures.py
```

Inputs: `results/full/tables/summary_test_by_level.csv`, `results/full/verify/estimation.parquet`,
`results/full/calibration/threshold_curve.parquet` and `results/full/calibration/frozen_router.json`.

## Provenance of the numbers

Every number in the paper comes from files under `results/full/` (mainly `report.md`, `tables/*.csv`,
`calibration/frozen_router.json`, `verify/consistency.json`, `env/*.json`) and from
`data/processed/full/manifest.json`. All references were checked against Crossref DOI records, the publishers'
proceedings pages, arXiv, or the live documentation URLs on 2026-11-07.
