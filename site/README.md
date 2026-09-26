# NeuroChip Twin: static explorer

Serverless site (GitHub Pages ready): `index.html`, `assets/` and precomputed data in `data/`.
Nothing runs server side; every number on the page is read at runtime from `data/*.json`.

```bash
python scripts/precompute_site.py          # rebuild data/ from data_bundle/ (CPU, offline, about 40 s)
python -m http.server 8000 -d site         # then open http://localhost:8000
```

- `data/index.json`: feature metadata, fold info, abstention thresholds and a per-chemical summary.
- `data/chem/cNNN.json`: one file per chemical. It holds the held-out forecast of its own cross-validation
  fold for k = 1, 2, 3 measured concentrations ('spread' design of `demo.py`), with the mean and 90 % half-width on a
  40-point log-dose grid (17 features x 4 DIV, integers x100), level means, wells and errors against the baselines.
- `data/summary.json`: headline numbers copied from `results/*.json`, with SHA-256 hashes of the sources.

Plotly is loaded from `cdn.jsdelivr.net`. If the CDN is unreachable, the tables still render.
Opening `index.html` as a `file://` URL does not work, because browsers block `fetch` there. Use any static server.
