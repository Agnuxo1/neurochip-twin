# Analysis of the most visible public Kaggle code notebook (not a ranked submission)

**Initial check:** 2026-09-21; live code-page review updated 2026-09-23.
**Competition:** [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien)

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
- Added a dependency-light audit of the public OOC metadata spreadsheet,
  preserving its SHA-256 and missingness/condition coverage without treating
  sample-quality labels as toxicity labels.
- Added a stronger reproducibility contract and tests before considering any public submission.

## Current local evidence

For seed 42 and 180 generated sequences in the compound-specific scenario, the multimodal model reports ROC-AUC 0.990, average precision 0.991, balanced accuracy 0.955, accuracy 0.956, F1 0.958, viability RMSE 5.856/R² 0.957 and IC50 RMSE 0.216/R² 0.957. Physics-only ROC-AUC is 0.692 and the temporal reservoir reaches 0.968, so the benchmark tests both temporal dynamics and context fusion rather than allowing a dose-only shortcut. Across ten independent seeds, multimodal ROC-AUC averaged 0.987 and F1 0.936; grouped batches gave ROC-AUC 0.983 and F1 0.944. These are synthetic proxy metrics. The next score-critical step is validation on licensed public microscopy data and, ideally, authorized OoC data with chip-level splits.

## Stability follow-up

The ten-seed compound-specific audit reports multimodal ROC-AUC mean 0.987 (range 0.968–0.998), F1 mean 0.936, viability R² mean 0.902 and IC50 R² mean 0.908. The grouped version reports ROC-AUC 0.983 and F1 0.944. The regression heads use temporal and context blocks, ridge regularization (`alpha=10`) and train-range clipping; the classifier's explicit cross-modal terms remain inspectable. The multimodal ROC-AUC gain over the temporal reservoir averages +0.0023 in random splits and +0.0035 in grouped splits, and is not positive for every seed. This is reproducibility evidence, not biological generalization.

The grouped compound-specific audit reports multimodal ROC-AUC 0.983 ± 0.023, F1 0.944 ± 0.051, viability R² 0.945 ± 0.019, and IC50 R² 0.942 ± 0.027 across ten seeds. It is deliberately labelled as a synthetic leakage diagnostic: replacing the proxy batch IDs with real chip/experiment IDs is required before treating the result as evidence of transfer.

The stronger compound-holdout audit puts every compound on one side of the split. Its additive multimodal readout averages ROC-AUC 0.991 ± 0.007 and F1 0.954 ± 0.028, while the explicit interaction readout averages 0.986/0.938. The strategy is therefore to select additive fusion for conservative unseen-compound generalization and retain interaction terms as a transparent ablation.

The probability audit now reports multimodal Brier/ECE of 0.053/0.069 for random splits and 0.048/0.071 for grouped splits. These diagnostics improve transparency around the uncertainty proxy but do not convert synthetic probabilities into clinical confidence.

We also added a real-data front-end check on BBBC038v1. After calibrating on 12 images and freezing the parameters, 24 held-out images gave mean IoU 0.520, Dice 0.575, precision 0.842 and recall 0.578. The new OOC metadata audit adds direct domain coverage and missingness evidence, but still no response score. This improves the validation story compared with a synthetic-only submission, while the modest recall and non-OoC segmentation domain are disclosed rather than hidden. The next score-critical experiment is authorized OoC image/quality evaluation or matched perturbation data with measured chip/experiment IDs.

## External neural-assay audit — 2026-09-23 (exploratory, not OoC validation)

To add a measured-biology branch without implying that it validates the image
model, we implemented a reproducible audit of the public [EPA acute
embryonic-rat DRG workbook](https://catalog.data.gov/dataset/kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319),
citing its [license/disclaimer](https://pasteur.epa.gov/license/sciencehub-license.html)
and [source paper](https://doi.org/10.1016/j.tiv.2020.104989). For MEA firing,
leave-one-treatment-label-out validation (12 labels; 155 non-control rows;
controls training-only) gives a fixed dose + pre-rate Random Forest R² 0.335,
MAE 0.391 versus Ridge R² 0.171, MAE 0.449. The module reports equal-label
macro-MAE 0.384 (95% bootstrap interval 0.299–0.482) versus 0.440
(0.356–0.533); paired forest-minus-Ridge difference -0.056 (-0.084 to
-0.028; 95% cluster-bootstrap interval, 10,000 resamples). This resamples
treatment labels instead of treating rows as independent replicates.
Recorded-date MEA validation also gives context Random Forest R²
0.308 over 10 dates; LDH, NLPN and NPF grouped results are weak or negative.
This remains a modest exploratory association: the source uses acute rat DRG
cultures, recorded dates are only batch proxies, and its endpoints are not
paired with our microscopy sequences. It does not validate the image encoder,
OoC transfer, human biology or clinical outcomes.

The Kaggle CLI recheck on Sep 23 returned `userHasEntered=True` and
`userRank=0`; the official competition leaderboard command returned “No
results found.” Participation is reflected by Kaggle, but there is no official
rank yet and no basis to call any entry first/second. The competition closes
Oct 10, 2026 at 15:59:59 UTC (17:59:59 Madrid time).

## Live review — 2026-09-23 (not a placement)

Before the NeuroChip Twin notebook was published, the Kaggle API's
competition-code query, sorted by hotness, returned the public BioFluidNet-OoC
notebook as the only visible result on its first page; it had 15 votes at the
time checked. After publication, the same query returns BioFluidNet-OoC (15
votes) and NeuroChip Twin (0 votes). Votes measure community activity, **not**
official rank, winner, or evidence that either project is first or second.
This is a manually judged hackathon; expert review follows the Oct 10
submission deadline.

I downloaded and statically inspected its public notebook without executing
any cells: [BioFluidNet-OoC notebook](https://www.kaggle.com/code/avikdas567/biofluidnet-ooc-multimodal-phenotypic-profiling).
The review identifies methodological risks to account for when comparing
reported scores:

- The notebook calls `fit_transform` on each complete feature matrix before
  making the train/test partition. That allows held-out rows to influence
  preprocessing statistics.
- It uses a stratified sample-level fold rather than a compound-, chip-, or
  experiment-held-out split. Its synthetic generator assigns toxicity and
  mechanism labels from a fixed compound table, while compound descriptors
  recur across the random split.
- The same test partition is evaluated every epoch and is used for checkpoint
  selection, so it is not an untouched final test set.
- Several ablation scores are literal constants in the plotted comparison
  table rather than outputs recomputed by the notebook's ablation code. Those
  comparisons therefore cannot be independently reproduced from that table.
- Its perfect headline metrics come from its own synthetic benchmark. The
  notebook's text presents biological interpretation, but the displayed
  metrics are not measured neural OoC response validation.

These are observations about the visible code, not allegations of intent and
not a verified competition ranking. Useful ideas to retain are multimodal
context, explicit physics features, multiple task heads and flow
counterfactuals. NeuroChip Twin should keep the smaller, auditable hybrid and
prefer group/compound holdouts, train-only preprocessing, a final test set not
used for checkpoint selection, and executable ablations. Its own synthetic
metrics still do not establish biological transfer.

Recent primary/domain sources reinforce this direction: the ISO/CD 25591
committee draft is developing OoC/MPS metadata and experimental-design
requirements for digital twins; the PReP MPS reproducibility protocol
emphasizes metadata plus CV, ANOVA and ICC; and a 2026 cancer-on-chip study
reports that longitudinal, baseline-normalized viability features improved
dose-response sensitivity in its own system. These support better metadata,
replicate-aware evaluation and temporal measurements; none validates our
neural OoC model. See [ISO/CD 25591](https://www.iso.org/es/contents/data/standard/09/08/90834.html?browse=tc),
[MPS reproducibility analytics](https://pmc.ncbi.nlm.nih.gov/articles/PMC12256941/),
and [longitudinal cancer-on-chip imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC13159154/).

### Operational consequence

The current public writeup still needs a fresh, action-time-approved update.
The [Kaggle foundational rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/rules)
also require code shared publicly during the event to be shared on its Kaggle
discussion or code page; GitHub alone does not satisfy that separate
condition. The [CPU-only NeuroChip Twin Kaggle notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo)
now embeds the source mirror and is linked to the competition. Version 2 ran
successfully on Kaggle (49 seconds); it regenerated ROC-AUC 0.990 and F1
0.958 from 180 synthetic sequences, with no Internet or GPU. Kaggle labels the
notebook copy Apache 2.0; the GitHub repository remains MIT. The output is not
biological validation and the notebook is not a leaderboard submission.
