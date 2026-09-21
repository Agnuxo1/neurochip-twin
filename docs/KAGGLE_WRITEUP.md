# NeuroChip Twin — Kaggle Writeup draft

## Category: End-to-End System

### Demo video

Attach `outputs/demo/neurochip_twin_demo.mp4` (maximum five minutes) or a public mirror. The video should show: generated time-lapse, segmentation/tracks, static-vs-temporal ablation, probability/uncertainty, and the reproducibility command.

### Code repository

`https://github.com/Agnuxo1/neurochip-twin` (public repository; replace only if GitHub assigns a different final URL).

### Summary

NeuroChip Twin is an interpretable temporal digital-twin prototype for organ-on-chip microscopy. It converts a time-lapse into cell tracks and phenotype trajectories, then fuses them with dose, flow, shear and clearance context using a compact physics-informed readout. Separate heads estimate toxicity, viability and IC50, while a flow counterfactual shows how the predicted response changes under altered perfusion. The output includes measurable reasons behind the prediction: cell-count change, morphology, intensity, motion, track persistence and exposure context. Static and temporal baselines are included as ablations, so the value of dynamics and multimodal context is testable rather than assumed.

On the repository's deterministic compound-specific synthetic organ-on-chip-like benchmark (seed 42, 180 sequences, 25% held out), the static baseline ROC-AUC is 0.567, the physics-only baseline is 0.692, the temporal model is 0.968, and the multimodal physics-informed model is 0.990 with F1 0.958. The auxiliary heads obtain viability R² 0.957 and IC50 R² 0.957. A ten-seed audit gives multimodal ROC-AUC 0.987, F1 0.936, viability R² 0.902 and IC50 R² 0.908; the grouped audit gives multimodal ROC-AUC 0.983 and F1 0.944. These values are a synthetic stress test, not clinical performance. The project intentionally exposes this limitation and provides the data contract and validation plan for authorized public microscopy and organ-on-chip data.

The primary scenario includes seeded compound descriptors and a latent susceptibility factor visible only through temporal phenotype. This makes the physics-only ablation materially weaker and tests whether dynamics add information. The ten-seed multimodal-versus-temporal ROC-AUC gain averages +0.0023 in random splits and +0.0035 in grouped splits; these synthetic batch IDs are explicitly only a proxy, and real evaluation must use measured chip/experiment IDs.

The image-analysis front-end was also audited on 24 held-out BBBC038v1 microscopy images after parameter calibration on 12 separate images: mean pixel IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. BBBC038 is public CC0 microscopy data; this is a portability check for segmentation only, not OoC response validation or clinical performance. The repository documents the exact download, license and command.

The older `exposure_only` scenario remains available as a control and is intentionally not used as the headline result because its label is too tightly coupled to dose and shear. The primary scenario is still synthetic and makes no biological generalization claim; the value of the system is the auditable temporal/multimodal workflow and its explicit path to authorized OoC data.

### Technical report

See [`docs/TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md). It documents data generation, segmentation, Hungarian tracking, temporal reservoir equation, metrics, failure modes, legal/compliance scope, and the real-data validation plan.

### Why this matters

Organ-on-chip experiments are dynamic. A final image can miss delayed death, transient morphology, or motility changes. A transparent temporal layer can prioritize wells for review and make the reason for a flag visible to an experimental scientist. The system is a research-assistance tool, not a medical device.

### Reproducibility

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific
```

### Limitations and ethics

The included benchmark is synthetic and contains no clinical data. The system must not be used to make patient or dosing decisions. Any real deployment requires data-use authorization, biological replicates, batch-aware splits, calibration, domain review, and external validation. Public datasets and third-party software must retain their original licences.

### Prior work disclosure

The temporal design was informed by the author's QESN_MABe project; scientific-audit ideas were informed by CAJAL; JEV was used as the local planning/control plane. The competition-specific implementation, benchmark, ablation, report, and limitations are included in this repository.
