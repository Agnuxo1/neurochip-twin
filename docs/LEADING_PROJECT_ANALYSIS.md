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

For seed 42 and 180 generated sequences in the compound-specific scenario, the multimodal model reports ROC-AUC 0.990, average precision 0.991, balanced accuracy 0.955, accuracy 0.956, F1 0.958, viability RMSE 5.856/R² 0.957 and IC50 RMSE 0.216/R² 0.957. Physics-only ROC-AUC is 0.692 and the temporal reservoir reaches 0.968, so the benchmark tests both temporal dynamics and context fusion rather than allowing a dose-only shortcut. Across ten independent seeds, multimodal ROC-AUC averaged 0.987 and F1 0.936; grouped batches gave ROC-AUC 0.983 and F1 0.944. These are synthetic proxy metrics. The next score-critical step is validation on licensed public microscopy data and, ideally, authorized OoC data with chip-level splits.

## Stability follow-up

The ten-seed compound-specific audit reports multimodal ROC-AUC mean 0.987 (range 0.968–0.998), F1 mean 0.936, viability R² mean 0.902 and IC50 R² mean 0.908. The grouped version reports ROC-AUC 0.983 and F1 0.944. The regression heads use temporal and context blocks, ridge regularization (`alpha=10`) and train-range clipping; the classifier's explicit cross-modal terms remain inspectable. The multimodal ROC-AUC gain over the temporal reservoir averages +0.0023 in random splits and +0.0035 in grouped splits, and is not positive for every seed. This is reproducibility evidence, not biological generalization.

The grouped compound-specific audit reports multimodal ROC-AUC 0.983 ± 0.023, F1 0.944 ± 0.051, viability R² 0.945 ± 0.019, and IC50 R² 0.942 ± 0.027 across ten seeds. It is deliberately labelled as a synthetic leakage diagnostic: replacing the proxy batch IDs with real chip/experiment IDs is required before treating the result as evidence of transfer.

The stronger compound-holdout audit puts every compound on one side of the split. Its additive multimodal readout averages ROC-AUC 0.991 ± 0.007 and F1 0.954 ± 0.028, while the explicit interaction readout averages 0.986/0.938. The strategy is therefore to select additive fusion for conservative unseen-compound generalization and retain interaction terms as a transparent ablation.

The probability audit now reports multimodal Brier/ECE of 0.053/0.069 for random splits and 0.048/0.071 for grouped splits. These diagnostics improve transparency around the uncertainty proxy but do not convert synthetic probabilities into clinical confidence.

We also added a real-data front-end check on BBBC038v1. After calibrating on 12 images and freezing the parameters, 24 held-out images gave mean IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. This improves the validation story compared with a synthetic-only submission, while the modest recall and non-OoC domain are disclosed rather than hidden. The next score-critical experiment is authorized OoC or matched perturbation data with measured chip/experiment IDs.
