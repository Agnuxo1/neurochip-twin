**Submission category: End-to-End System**

## Project Summary

NeuroChip Twin is an interpretable research prototype for time-lapse microscopy in organ-on-chip workflows. It tracks cells, extracts morphology, intensity, motion and persistence features, summarizes temporal trajectories with a fixed reservoir, and combines them with dose, flow, shear and clearance context. Separate readout heads estimate toxicity, viability and IC50; a counterfactual tool explores how predicted response changes with perfusion. Reports expose predictions, phenotype traces and the factors behind each flag.

The main image-to-response benchmark remains repository-generated synthetic data; there is no paired neural OoC response dataset. In the deterministic compound-specific benchmark (seed 42, 180 sequences, 25% held out), static and physics-only baselines reach ROC-AUC 0.567 and 0.692; the temporal model reaches 0.968, and additive multimodal fusion reaches 0.990 (F1 0.958). Ten-seed validation averages 0.987 ROC-AUC and 0.936 F1; grouped synthetic splits reach 0.983 ROC-AUC. A compound-holdout audit reaches 0.991 ± 0.007 ROC-AUC and 0.954 ± 0.028 F1. The benchmark deliberately includes latent susceptibility encoded in temporal features, so these stress tests do not estimate biological transfer. Separately, real rat DRG MEA tabular data (not OoC or image-paired) gave leave-one-treatment-label-out R² 0.335 / MAE 0.391 using dose/pre-rate only (12 labels, 155 rows), versus Ridge 0.171 / 0.449; paired label-level MAE difference -0.056 (95% cluster-bootstrap interval -0.084 to -0.028).

The image front end was separately audited on BBBC038v1: after calibration on 12 images, segmentation on 24 held-out images yields pixel IoU 0.520 and Dice 0.575. BBBC038 is a segmentation-portability audit only, not OoC response validation. This research-assistance prototype is not a clinical or dosing tool. Its decisive next test requires authorized neural OoC data, chip- or experiment-held-out evaluation, biological replicates and external validation. The public repository includes source, tests, technical report and reproducible commands.

## Technical Method

The pipeline performs morphology and intensity extraction, connected-component segmentation, Hungarian frame-to-frame association, trajectory aggregation and a fixed temporal reservoir. A physics feature block represents administered dose, flow, shear proxy, clearance and effective exposure. Cross-modal interaction terms are explicit and auditable. The final readout is a regularized multi-task model with toxicity classification, viability regression and IC50 regression heads. Reports include predictions, phenotype tables, confusion matrices, metric JSON, counterfactual CSV/PNG and an HTML overview.

As a separate exploratory assay branch, fixed tabular baselines were tested on
the [public EPA acute embryonic-rat DRG data](https://catalog.data.gov/dataset/kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319).
For MEA firing, each of 12 treatment labels was held out in turn, vehicle
controls remained training-only, and the model used dose plus pre-dose firing
without chemical identity. A 300-tree Random Forest reached R² 0.335 / MAE
0.391 on 155 rows versus Ridge R² 0.171 / MAE 0.449. A paired bootstrap
over 12 labels estimates an equal-weight MAE improvement of 0.056 (95% interval
0.028 to 0.084). This is acute rat DRG data, not OoC,
not paired with our images, and not validation of the segmentation or temporal
image model; the other grouped assay endpoints are weak or negative. EPA's
[license/disclaimer](https://pasteur.epa.gov/license/sciencehub-license.html)
and [source paper](https://doi.org/10.1016/j.tiv.2020.104989) are documented in
the report. The source workbook is not included.

## Why This Matters

Organ-on-chip experiments are dynamic. A final image can miss delayed death, transient morphology, or motility changes. A transparent temporal layer can prioritize wells for review and make the reason for a flag visible to an experimental scientist. The system is a research-assistance tool, not a medical device.

## Reproducibility

The public repository contains source code, tests, report and generated demo artifacts. Run:

    python -m pip install -r requirements.txt
    python -m pytest -q
    python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific
    python -m src.external_assay_validation --input /path/to/KodavantiP_Acute_HSAB_AOP_Neurotox_Science_Hub.xlsx --out outputs/external_assay_validation/summary.json

Technical report: https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/TECHNICAL_REPORT.md

Competition code notebook: https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo

## Limitations and Ethics

The included benchmark is synthetic and contains no clinical data. The system must not be used to make patient or dosing decisions. Any real deployment requires data-use authorization, biological replicates, batch-aware splits, calibration, domain review, and external validation. Public datasets and third-party software must retain their original licences. BBBC038 is used only as a segmentation portability audit and is not biological response validation.

## Prior Work Disclosure

The temporal design was informed by the author's QESN_MABe project; scientific-audit ideas were informed by CAJAL; JEV was used as the local planning/control plane. The competition-specific implementation, benchmark, ablation, report, and limitations are included in this repository.
