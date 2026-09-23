**Submission category: End-to-End System**

## Project Summary

NeuroChip Twin is a transparent research prototype for time-lapse microscopy in organ-on-chip workflows. It segments and tracks cell-like objects, extracts morphology, intensity and motion, summarizes temporal trajectories with a fixed reservoir, and fuses them with dose, flow, shear and clearance context. Auditable readouts estimate toxicity, viability and IC50; reports include phenotype traces and flow counterfactuals.

The image-to-response benchmark is generated synthetically; no paired neural OoC response data are available. With seed 42 and 180 sequences, additive multimodal fusion reaches ROC-AUC 0.990 (F1 0.958), versus 0.968 for the temporal model and 0.692 physics-only. Ten-seed mean ROC-AUC/F1 is 0.987/0.936; compound-holdout mean is 0.991 ± 0.007/0.954 ± 0.028. The generator encodes latent susceptibility in temporal features, so these scores measure controlled pipeline behavior, not biology. Separately, public acute rat DRG MEA tabular data (not OoC or image-paired) yielded leave-treatment-label-out RF R² 0.335, MAE 0.391 over 12 labels/155 rows; this does not validate the image model or human/OoC transfer.

An exploratory BBBC038v1 audit calibrated on 40 images and evaluated on 80 separate images. Optional distance-watershed post-processing improved mean per-image instance F1 across IoU 0.50–0.95 from 0.3207 to 0.3325 (paired difference 0.0118; 95% image-bootstrap interval 0.0013–0.0226); pixel Dice stayed at 0.5242. Exact split IDs and parameters are public. BBBC038 tests nuclei segmentation only, not OoC response. This prototype is not a clinical or dosing tool; the decisive test requires authorized neural OoC data, experiment-held-out evaluation, biological replicates and external validation.

## Technical Method

The pipeline performs morphology and intensity extraction, threshold/connected-component segmentation (with optional distance-watershed instance splitting), Hungarian frame-to-frame association, trajectory aggregation and a fixed temporal reservoir. A physics feature block represents administered dose, flow, shear proxy, clearance and effective exposure. Cross-modal interaction terms are explicit and auditable. The final readout is a regularized multi-task model with toxicity classification, viability regression and IC50 regression heads. Reports include predictions, phenotype tables, confusion matrices, metric JSON, counterfactual CSV/PNG and an HTML overview. The exact BBBC038 image IDs, checksum and audit parameters are recorded in the [split manifest](https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/bbbc038_watershed_split_seed20260924.json).

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
