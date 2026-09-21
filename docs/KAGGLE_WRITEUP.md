# NeuroChip Twin — Kaggle Writeup draft

## Category: End-to-End System

### Demo video

Attach `outputs/demo/neurochip_twin_demo.mp4` (maximum five minutes) or a public mirror. The video should show: generated time-lapse, segmentation/tracks, static-vs-temporal ablation, probability/uncertainty, and the reproducibility command.

### Code repository

`https://github.com/Agnuxo1/neurochip-twin` (public repository; replace only if GitHub assigns a different final URL).

### Summary

NeuroChip Twin is an interpretable temporal digital-twin prototype for organ-on-chip microscopy. It converts a time-lapse into cell tracks and phenotype trajectories, then fuses them with dose, flow, shear and clearance context using a compact physics-informed readout. Separate heads estimate toxicity, viability and IC50, while a flow counterfactual shows how the predicted response changes under altered perfusion. The output includes measurable reasons behind the prediction: cell-count change, morphology, intensity, motion, track persistence and exposure context. Static and temporal baselines are included as ablations, so the value of dynamics and multimodal context is testable rather than assumed.

On the repository's deterministic synthetic organ-on-chip-like benchmark (seed 42, 180 sequences, 25% held out), the static baseline ROC-AUC is 0.280, the temporal model is 0.996, and the multimodal physics-informed model is 0.998 (F1 0.984). The auxiliary heads obtain viability R² 0.970 and IC50 R² 0.973. A ten-seed audit gives multimodal ROC-AUC 0.992 ± 0.010, F1 0.963 ± 0.023, viability R² 0.964 ± 0.013, and IC50 R² 0.968 ± 0.011. These values are a synthetic stress test, not clinical performance. The project intentionally exposes this limitation and provides the data contract and validation plan for authorized public microscopy and organ-on-chip data.

As an additional leakage diagnostic, a ten-seed grouped acquisition-batch split gives ROC-AUC 0.995 ± 0.005, F1 0.973 ± 0.017, viability R² 0.962 ± 0.020, and IC50 R² 0.965 ± 0.018. These synthetic batch IDs are explicitly only a proxy; real evaluation must use measured chip/experiment IDs.

The image-analysis front-end was also audited on 24 held-out BBBC038v1 microscopy images after parameter calibration on 12 separate images: mean pixel IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. BBBC038 is public CC0 microscopy data; this is a portability check for segmentation only, not OoC response validation or clinical performance. The repository documents the exact download, license and command.

An important ablation finding is that the physics-only baseline also reaches ROC-AUC 0.998 on this proxy. We therefore make no claim that the multimodal interactions improve accuracy until the system is tested on real or harder compound-specific data; their current value is interpretability, counterfactual analysis, and an extensible interface for measured OoC covariates.

### Technical report

See [`docs/TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md). It documents data generation, segmentation, Hungarian tracking, temporal reservoir equation, metrics, failure modes, legal/compliance scope, and the real-data validation plan.

### Why this matters

Organ-on-chip experiments are dynamic. A final image can miss delayed death, transient morphology, or motility changes. A transparent temporal layer can prioritize wells for review and make the reason for a flag visible to an experimental scientist. The system is a research-assistance tool, not a medical device.

### Reproducibility

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180
```

### Limitations and ethics

The included benchmark is synthetic and contains no clinical data. The system must not be used to make patient or dosing decisions. Any real deployment requires data-use authorization, biological replicates, batch-aware splits, calibration, domain review, and external validation. Public datasets and third-party software must retain their original licences.

### Prior work disclosure

The temporal design was informed by the author's QESN_MABe project; scientific-audit ideas were informed by CAJAL; JEV was used as the local planning/control plane. The competition-specific implementation, benchmark, ablation, report, and limitations are included in this repository.
