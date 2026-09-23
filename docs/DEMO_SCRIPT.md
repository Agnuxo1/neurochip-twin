# Five-minute demo script

0:00–0:30 — State the problem: organ-on-chip movies are dynamic, but endpoint-only analysis discards information.

0:30–1:20 — Run `python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific`; show the two microscopy frames and compound context.

1:20–2:10 — Explain segmentation, centroids, Hungarian tracking, and the phenotype table.

2:10–3:10 — Show the static, physics-only, temporal and multimodal ablations. Read the exact ROC-AUC values from `metrics.json` and explain why the harder scenario prevents a dose-only shortcut.

3:10–3:40 — Show the probability, uncertainty proxy, Brier/ECE calibration diagnostics, confusion matrix, and one failure mode. Say explicitly that the benchmark is synthetic and calibration is not a clinical confidence interval.

3:40–4:20 — Show the separate BBBC038 instance-segmentation audit: 40-image calibration, 80-image evaluation, connected components versus optional distance watershed, and the paired per-image F1 delta. State prominently that BBBC038 is a nuclei-segmentation portability check, not organ-on-chip response validation; pixel Dice did not improve.

4:20–5:00 — Show the test suite and report. End with the authorized neural OoC validation plan and the no-clinical-claim boundary.
