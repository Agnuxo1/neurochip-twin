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

For seed 42 and 180 generated sequences, the multimodal model reports ROC-AUC 0.998, average precision 0.999, balanced accuracy 0.967, accuracy 0.978, F1 0.984, viability RMSE 5.448/R² 0.967 and IC50 RMSE 0.196/R² 0.972. The physics-only ablation also reaches ROC-AUC 0.998, so this local proxy does not support a claim that multimodal fusion is superior; the label is currently too tightly coupled to effective exposure. Across five independent seeds, multimodal ROC-AUC averaged 0.997 (range 0.994–1.000) and F1 averaged 0.970 (range 0.958–0.984); regression R² was less stable, averaging 0.901 for viability and 0.910 for IC50. These are synthetic proxy metrics. The next score-critical step is validation on licensed public microscopy data and, ideally, authorized OoC data with chip-level splits.
