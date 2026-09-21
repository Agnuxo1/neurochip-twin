## Project Summary

NeuroChip Twin is an interpretable temporal digital-twin prototype for organ-on-chip microscopy. It converts a time-lapse into cell tracks and phenotype trajectories, then fuses them with dose, flow, shear and clearance context using a compact physics-informed readout. Separate heads estimate toxicity, viability and IC50, while a flow counterfactual shows how the predicted response changes under altered perfusion. The output includes measurable reasons behind each prediction: cell-count change, morphology, intensity, motion, track persistence and exposure context. Static and temporal baselines are included as ablations, so the value of dynamics and multimodal context is testable rather than assumed.

On the repository's deterministic compound-specific synthetic organ-on-chip-like benchmark (seed 42, 180 sequences, 25% held out), the static baseline ROC-AUC is 0.567, the physics-only baseline is 0.692, the temporal model is 0.968, and the primary additive multimodal context readout is 0.990 with F1 0.958. The explicit physics-interaction readout is retained as an ablation with the same seed-42 score. The auxiliary heads obtain viability R² 0.957 and IC50 R² 0.957. A ten-seed audit gives multimodal ROC-AUC 0.987, F1 0.936, viability R² 0.902 and IC50 R² 0.908; the grouped audit gives multimodal ROC-AUC 0.983 and F1 0.944. These values are a synthetic stress test, not clinical performance. The project intentionally exposes this limitation and provides the data contract and validation plan for authorized public microscopy and organ-on-chip data.

The probability readout is audited as well as ranked: seed 42 gives a multimodal Brier score of 0.041 and ten-bin ECE of 0.064; across ten seeds these average 0.053 and 0.069, while the grouped audit averages 0.048 and 0.071. These are calibration diagnostics for the synthetic proxy, not clinical confidence intervals.

The primary scenario includes seeded compound descriptors and a latent susceptibility factor visible only through temporal phenotype. This makes the physics-only ablation materially weaker and tests whether dynamics add information. The ten-seed multimodal-versus-temporal ROC-AUC gain averages +0.0023 in random splits and +0.0035 in grouped splits; these synthetic batch IDs are explicitly only a proxy, and real evaluation must use measured chip/experiment IDs.

As a stricter generalization check, a ten-seed compound-holdout audit keeps each compound entirely out of either train or test. The additive multimodal readout reaches ROC-AUC 0.991 ± 0.007 and F1 0.954 ± 0.028, compared with 0.989 for the temporal-only readout and 0.986/0.938 for the explicit interaction readout. This makes additive fusion the conservative unseen-compound choice; all values remain synthetic regression-test evidence, not biological validation.

The image-analysis front-end was also audited on 24 held-out BBBC038v1 microscopy images after parameter calibration on 12 separate images: mean pixel IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. BBBC038 is public CC0 microscopy data; this is a portability check for segmentation only, not OoC response validation or clinical performance. The repository documents the exact download, license and command.

The older exposure_only scenario remains available as a control and is intentionally not used as the headline result because its label is too tightly coupled to dose and shear. The primary scenario is still synthetic and makes no biological generalization claim; the value of the system is the auditable temporal/multimodal workflow and its explicit path to authorized OoC data.

## Technical Method

The pipeline performs morphology and intensity extraction, connected-component segmentation, Hungarian frame-to-frame association, trajectory aggregation and a fixed temporal reservoir. A physics feature block represents administered dose, flow, shear proxy, clearance and effective exposure. Cross-modal interaction terms are explicit and auditable. The final readout is a regularized multi-task model with toxicity classification, viability regression and IC50 regression heads. Reports include predictions, phenotype tables, confusion matrices, metric JSON, counterfactual CSV/PNG and an HTML overview.

## Why This Matters

Organ-on-chip experiments are dynamic. A final image can miss delayed death, transient morphology, or motility changes. A transparent temporal layer can prioritize wells for review and make the reason for a flag visible to an experimental scientist. The system is a research-assistance tool, not a medical device.

## Reproducibility

The public repository contains source code, tests, report and generated demo artifacts. Run:

    python -m pip install -r requirements.txt
    python -m pytest -q
    python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific

Technical report: https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/TECHNICAL_REPORT.md

## Limitations and Ethics

The included benchmark is synthetic and contains no clinical data. The system must not be used to make patient or dosing decisions. Any real deployment requires data-use authorization, biological replicates, batch-aware splits, calibration, domain review, and external validation. Public datasets and third-party software must retain their original licences. BBBC038 is used only as a segmentation portability audit and is not biological response validation.

## Prior Work Disclosure

The temporal design was informed by the author's QESN_MABe project; scientific-audit ideas were informed by CAJAL; JEV was used as the local planning/control plane. The competition-specific implementation, benchmark, ablation, report, and limitations are included in this repository.
