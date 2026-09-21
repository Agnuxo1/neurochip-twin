# Evidence-based analysis of the leading visible Kaggle project

**Checked:** 2026-09-21. **Competition:** [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien)

## What “first place” means at this stage

There is no official leaderboard or judged first place yet. Kaggle shows the writeups as submitted but hidden until the hackathon closes; the competition page states that expert review and the final round happen after the October 10 deadline. The public Code tab exposed one notebook sorted by Hotness, `BioFluidNet-OoC: Multimodal Phenotypic Profiling`, with 14 votes and a Bronze badge. It is therefore the strongest public reference available, not a verified winner.

Reference: [public BioFluidNet-OoC notebook](https://www.kaggle.com/code/avikdas567/biofluidnet-ooc-multimodal-phenotypic-profiling).

## Techniques worth adapting

The notebook combines three modalities: molecular descriptors and compound context; high-content morphological features inspired by Cell Painting; and microfluidic covariates such as flow, wall shear and clearance/exposure.

Its architecture uses cross-attention, multi-task heads for toxicity, mechanism of action, IC50 and viability, and a physics-consistency penalty. It also reports classical baselines, architectural ablations, gradient-based feature attribution, a deterministic seed, and a flow counterfactual.

## Important caution

The notebook reports a fully synthetic benchmark mapped to BBBC/JUMP-Cell Painting concepts, with perfect or near-perfect validation metrics. That makes it a strong presentation template, not proof of biological generalization. NeuroChip Twin adopts the useful structure but keeps the synthetic-to-real limitation explicit, uses a held-out split, and does not present synthetic results as real OoC evidence.

## Reconfiguration applied to NeuroChip Twin

- Added hydrodynamic covariates: flow rate, a documented shear proxy, clearance factor and effective dose.
- Added a physics-informed synthetic response generator with high-shear penalty and exposure attenuation.
- Added an explicit low-order cross-modal fusion layer, smaller and more inspectable than copying a deep attention network.
- Added multi-task readouts for binary toxicity, viability and IC50.
- Added flow counterfactual output: `outputs/demo/counterfactual_flow.csv` and `counterfactual_flow.png`.
- Added a stronger reproducibility contract and tests before considering any public submission.

## Current local evidence

For seed 42 and 180 generated sequences, the multimodal model reports ROC-AUC 0.998, average precision 0.999, balanced accuracy 0.967, accuracy 0.978, F1 0.984, viability RMSE 5.222/R² 0.970 and IC50 RMSE 0.190/R² 0.973. The physics-only ablation also reaches ROC-AUC 0.998, so this local proxy does not support a claim that multimodal fusion is superior; the label is currently too tightly coupled to effective exposure. Across ten independent seeds, multimodal ROC-AUC averaged 0.992 (range 0.973–1.000) and F1 averaged 0.963 (range 0.925–1.000); regression R² averaged 0.964 for viability and 0.968 for IC50. These are synthetic proxy metrics. The next score-critical step is validation on licensed public microscopy data and, ideally, authorized OoC data with chip-level splits.

## Stability follow-up

The ten-seed audit now reports multimodal ROC-AUC mean 0.992 (range 0.973–1.000), F1 mean 0.963 (range 0.925–1.000), viability R² mean 0.964 (range 0.932–0.977), and IC50 R² mean 0.968 (range 0.941–0.978). The regression heads use only temporal and physics blocks, ridge regularization (`alpha=10`) and train-range clipping; this removes the catastrophic extrapolation observed in one seed while keeping the classifier's interaction features unchanged. The multimodal ROC-AUC gain over the temporal reservoir averages +0.0053 and is not positive for every seed. This is a stronger reproducibility statement, not evidence of biological generalization: the score-critical next step remains authorized real-data validation with grouped splits by experiment/chip.

The new grouped acquisition-batch audit reports multimodal ROC-AUC 0.995 ± 0.005, F1 0.973 ± 0.017, viability R² 0.962 ± 0.020, and IC50 R² 0.965 ± 0.018 across ten seeds. It is deliberately labelled as a synthetic leakage diagnostic: replacing the proxy batch IDs with real chip/experiment IDs is required before treating the result as evidence of transfer.

We also added a real-data front-end check on BBBC038v1. After calibrating on 12 images and freezing the parameters, 24 held-out images gave mean IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. This improves the validation story compared with a synthetic-only submission, while the modest recall and non-OoC domain are disclosed rather than hidden. The next score-critical experiment is authorized OoC or matched perturbation data with measured chip/experiment IDs.
