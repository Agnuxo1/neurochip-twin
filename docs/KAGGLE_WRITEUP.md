# NeuroChip Twin — Kaggle Writeup draft

## Category: End-to-End System

### Demo video

Attach `outputs/demo/neurochip_twin_demo.mp4` (maximum five minutes) or a public mirror. The video should show: generated time-lapse, segmentation/tracks, static-vs-temporal ablation, probability/uncertainty, and the reproducibility command.

### Code repository

`https://github.com/Agnuxo1/neurochip-twin` (public repository; replace only if GitHub assigns a different final URL).

### Summary

NeuroChip Twin is an interpretable temporal digital-twin prototype for organ-on-chip microscopy. It converts a time-lapse into cell tracks and phenotype trajectories, then uses a fixed recurrent reservoir with a small logistic readout to estimate a treatment-response/toxicity state. The output includes the measurable reasons behind the prediction: cell-count change, morphology, intensity, motion, and track persistence. A static first-frame baseline is included as an ablation, so the value of temporal information is testable rather than assumed.

On the repository's deterministic synthetic organ-on-chip-like benchmark (seed 42, 180 sequences, 25% held out), the static baseline ROC-AUC is 0.345 and the temporal model ROC-AUC is 1.000 (Δ=+0.655). These values are a synthetic stress test, not clinical performance. The project intentionally exposes this limitation and provides the data contract and validation plan for authorized public microscopy and organ-on-chip data.

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
