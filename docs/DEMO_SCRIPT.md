# Five-minute demo script

0:00–0:30 — State the problem: organ-on-chip movies are dynamic, but endpoint-only analysis discards information.

0:30–1:20 — Run `python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180`; show the two microscopy frames.

1:20–2:10 — Explain segmentation, centroids, Hungarian tracking, and the phenotype table.

2:10–3:10 — Show the static baseline and temporal reservoir ablation. Read the exact ROC-AUC values from `metrics.json`.

3:10–4:10 — Show the probability, uncertainty proxy, confusion matrix, and one failure mode. Say explicitly that the benchmark is synthetic.

4:10–5:00 — Show the test suite and report. End with the real-data validation plan and the no-clinical-claim boundary.
