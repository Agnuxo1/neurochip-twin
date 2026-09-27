# NeuroChip Twin v2: project website

Static, dependency-free site for GitHub Pages: `index.html`, `assets/` (CSS, JS, figures) and precomputed data in `data/`.
No build step, no backend, no cookies, no tracking and no external requests. Every number on the page is read at load
time from `data/*.json`, which `scripts/precompute_site.py` writes from `results/*.json` and `data_bundle/`.

```bash
python scripts/precompute_site.py                 # rebuild data/ and assets/img/ (CPU, offline, about a minute)
python scripts/precompute_site.py --assets-only   # only re-copy the report figures and rebuild summary.json
python -m http.server 8000 -d site                # then open http://localhost:8000
```

Opening `index.html` as a `file://` URL does not load the data (browsers block `fetch` there); use any static server.

## Before publication

Fill the placeholders in the `CONFIG` block at the top of `assets/app.js`: `REPORT_URL`, `REPO_URL`, `VIDEO_URL`,
`KAGGLE_URL`. Until they are set, the buttons render greyed out with the label "soon". Drop the demo video at
`video/teaser.mp4` and set `VIDEO_FILE: "video/teaser.mp4"` in the same block (see `video/README.md`).

## Data files

- `data/summary.json`: key-number cards (each with the `results/<file>:<path>` it comes from), the cross-validation
  and calibration tables, data scale, and SHA-256 hashes of every input.
- `data/index.json`: feature metadata, fold info, the band, abstention, hazard and gate rules with their site-level
  checks, the example chemicals (the report's percentile rule, figure 4) and a per-chemical summary.
- `data/chem/cNNN.json`: one file per chemical, forecast by the ensemble of its own cross-validation fold (trained
  without it) for k = 1, 2, 3 evenly spread measured concentrations: forecast mean and cross-conformal 90 % half-width
  on a 32-point log-dose grid (17 features x 4 DIV, integers x100), level means, single wells, errors against the
  baselines, the abstention flag, the hazard probability per k and the two DIV 7 gates.
- `data/figures.json`: source path and SHA-256 of every report figure copied into `assets/img/`.

## Checks built into the precompute

- Hazard calls at k = 3 and from all doses must match `results/dnt_r4_calls.csv` (98 of 98 reference chemicals).
- DIV 7 gate decisions are re-derived from the fold thresholds of `results/r11_early_exit.json` (0 mismatches).
- The site's conformal band is reported with its observed coverage per k (`index.json` -> `band.site_coverage_by_k`).
