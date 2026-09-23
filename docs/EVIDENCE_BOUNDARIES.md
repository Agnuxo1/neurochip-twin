# Evidence Boundaries and Validation Matrix

**Snapshot:** 2026-09-23

**Purpose:** Let reviewers see exactly which NeuroChip Twin claims each dataset supports. The evidence streams below are deliberately kept separate unless the same experimental units and outcomes are genuinely paired.

## Evidence matrix

| Evidence stream | Data and target | Evaluation grouping | What it supports | What it does not support |
|---|---|---|---|---|
| Synthetic temporal benchmark | Locally generated organ-on-chip-like sequences; synthetic toxicity, viability, and IC50 targets | Random, synthetic-batch grouped, and compound-held-out splits | Reproducible software stress tests and controlled ablations of static, temporal, and context features | Biological performance, transfer to measured chips, clinical prediction, or evidence that the proxy's simulator is biologically correct |
| Human midbrain-organoid neurotoxicity candidate | Public HCS imaging and feature-table directory for 6-OHDA-exposed human midbrain organoids; candidate only, with dataset license and experimental-group metadata not yet verified | The paper describes repeated random 5-fold CV; no held-out organoid-batch/experiment evaluation has been verified | If rights and group metadata permit, a separately scoped neural-organoid audit | NeuroChip Twin results, neural organ-on-chip validation, chip-held-out generalization, or clinical prediction; no measurements have been evaluated by this project |
| OoC sample-image quality | Public brightfield image dataset; expert `good`/`bad` sample-quality labels across six non-neural cell lines | Leave one cell line out; no chip/experiment identifiers are available | Exploratory upstream image-quality triage; the current Inception-plus-metadata macro ROC-AUC is 0.773 on 3,072 matched images | Neural-cell transfer, toxicity or drug-response prediction, chip-held-out generalization, or an independently replicated test. The Zenodo record says CC-BY-4.0 while its dataset paper says CC-BY-SA; source images and model weights are not redistributed while that conflict remains unresolved |
| Rat DRG electrophysiology | Public acute embryonic-rat dorsal-root-ganglion assay; dose and pre-dose firing used to predict post-dose MEA firing | Treatment-label-held-out evaluation | An exploratory response-modeling branch for this measured rat assay; its targets and inputs are documented separately | OoC validation, human neural-chip validation, image-model validation, or a paired multimodal predictor |
| BBBC038 segmentation audit | Public microscopy nuclei images with instance masks | Fixed 40-image calibration / 80-image evaluation split; previous audit cases excluded | Portability of the segmentation front-end and an exploratory comparison of connected components with distance watershed | OoC-domain transfer, neural phenotype validity, toxicity prediction, or response validation. It is a segmentation audit only |
| GSE255606 neural-chip omics candidate | Public human spinal-cord-chip ALS/control bulk RNA-seq; GEO lists 88 samples and chip/experiment metadata | Not evaluated by this project | A candidate for a separately scoped disease-state transcriptomics study, subject to a preregistered protocol and batch-aware donor/experiment splits | Matched microscopy time series, treatment-response labels, validation of NeuroChip Twin's image model, or permission to join its labels to another assay. The study reports multiple experiments and technical batch/library-preparation effects |

The OoC quality results and exact caveats are recorded in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) and [`ooc_quality_public_summary.json`](../outputs/ooc_quality_public_summary.json). The rat DRG, BBBC038, and synthetic results are likewise documented in the technical report and their reproducible audit commands.

The human midbrain-organoid source is only a due-diligence candidate; see [`EXTERNAL_DATASET_DUE_DILIGENCE.md`](EXTERNAL_DATASET_DUE_DILIGENCE.md) for its unresolved license, grouping, and preregistration gates. A separate human sensory-neuron MPS paper gives a useful architectural precedent for distinct neurite-toxicity and cytotoxicity outputs plus an orthogonal NF-L assay, but it is peripheral-neuron literature, not a public benchmark or validation of this project.

## Non-negotiable interpretation rules

1. **Do not pool labels across unpaired experiments.** The OoC quality images, rat DRG firing assay, BBBC038 nuclei masks, synthetic response sequences, and GSE255606 omics samples do not share matched sample-level image/response outcomes. They cannot be treated as one multimodal training or validation table.
2. **Name each task and domain beside its score.** A metric without its dataset, target, split unit, and evidence class is not a project-level performance claim.
3. **Keep the current response predictor explicitly provisional.** Its strong synthetic scores demonstrate behavior on the local generator only. The real OoC image audit is quality-control evidence, not response evidence; BBBC038 is segmentation-only; the rat assay is separate electrophysiology evidence.
4. **Require paired, authorized data for a biological response claim.** Before fitting a real neural OoC response model, obtain use rights and sample-level time-lapse images, measured outcomes, treatment/dose metadata, and chip, experiment, and biological-replicate IDs. Freeze a chip/experiment-held-out evaluation before any tuning; report calibration and uncertainty at the independent experiment level.
5. **Treat GSE255606 as a new question, not a shortcut.** If pursued, define a separate ALS-versus-control omics task, preregister it before analysis, and account for donor, chip, experiment, and library-preparation structure. Do not describe that sidecar as validation of image-based drug response.

## Competition-facing claim

NeuroChip Twin is an interpretable, reproducible **end-to-end research prototype** with distinct synthetic, microscopy-quality, segmentation, and electrophysiology evidence modules. It is not yet a biologically validated neural organ-on-chip drug-response predictor. This boundary is part of the reliability contribution, not a caveat to hide.

The published Kaggle rubric weights the five criteria as follows:

| Criterion | Weight | Strongest evidence to show | Gap to state plainly |
|---|---:|---|---|
| Problem importance and potential impact | 30% | Transparent workflow for longitudinal OoC image review and response-assay prioritization | Clinical or neural-chip response utility has not been established |
| Technical approach and innovation | 30% | Cell tracking, temporal reservoir, interpretable context fusion, explicit static/temporal/physics ablations | The response labels in the core benchmark are generated by the project itself |
| Results and validation | 20% | Separate held-out synthetic audits plus real OoC image-quality, rat DRG assay, and BBBC038 segmentation audits | None of the real-data audits validates neural OoC image-based toxicity or drug response |
| Reproducibility and implementation quality | 10% | Public source, commands, tests, provenance and now a clean GitHub Actions test matrix | Real-data reproduction requires users to obtain each source dataset under its own terms |
| Presentation quality | 10% | A short, judge-ready workflow demo with visible inputs, outputs, interpretation and limits | Keep the demo link and embedded summary synchronized with the latest audited results |

The strongest defensible submission should pair the clear system demonstration with exact evidence boundaries and a concrete path to paired, experiment-held-out neural OoC validation, rather than imply a rank or biological result unsupported by the data.

## Sources

- [Official AI4S Kaggle challenge and rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview/url)
- [Organ-on-a-Chip image dataset, Zenodo](https://zenodo.org/records/10203721); [associated dataset paper](https://doi.org/10.3390/data9020028)
- [EPA acute HSAB AOP neurotoxicity dataset](https://catalog.data.gov/dataset/kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319)
- [BBBC038 dataset](https://bbbc.broadinstitute.org/BBBC038)
- [GEO GSE255606](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255606); [associated ALS spinal-cord-chip study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12233189/)
- [Monzel et al. human midbrain-organoid neurotoxicity study](https://doi.org/10.1016/j.parkreldis.2020.05.011); [public HCS data directory](https://webdav.lcsb.uni.lu/public/data/machine-learning-assisted-neurotoxicity-prediction-in-human-midbrain-organoid/); [analysis code](https://github.com/LCSB-DVB/Monzel_2020)
- [Han et al. human sensory-neuron MPS and morphological deep-learning study](https://doi.org/10.3390/toxics12110809)
