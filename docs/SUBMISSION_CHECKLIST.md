# Kaggle submission checklist

This checklist records the state of the AI4S Open Innovation: AI for Life Science
writeup and its reproducibility evidence.

## Submission fields

- Category: **End-to-End System**
- Title: `NeuroChip Twin: interpretable temporal OoC digital twin`
- Subtitle: `Physics-informed temporal microscopy analysis for toxicity, viability and dose-response screening`
- Competition: [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien)
- Writeup editor: [NeuroChip Twin writeup](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/new-writeup-1789975521348)
- Video: [YouTube demo](https://www.youtube.com/watch?v=U0zyU-AvXo8)
- Code: [public GitHub repository](https://github.com/Agnuxo1/neurochip-twin)
- Technical report: [technical report](https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/TECHNICAL_REPORT.md)
- Video file mirror: `outputs/demo/neurochip_twin_demo.mp4`

## Evidence to report

The current public repository is reproducible from commit `f20470b` or later.
The primary benchmark is explicitly labelled synthetic:

- seed 42, 180 sequences, 25% held out;
- multimodal ROC-AUC 0.990, F1 0.958;
- viability R² 0.957 and IC50 R² 0.957;
- ten-seed random audit: ROC-AUC 0.987, F1 0.936;
- ten-seed grouped audit: ROC-AUC 0.983, F1 0.944;
- ten-seed compound-holdout audit: additive multimodal ROC-AUC 0.991 ± 0.007, F1 0.954 ± 0.028; explicit-interaction ablation ROC-AUC 0.986, F1 0.938;
- multimodal Brier/ECE: 0.041/0.064 on seed 42, 0.053/0.069 random-audit mean, and 0.048/0.071 grouped-audit mean;
- BBBC038 front-end audit: 12 calibration and 24 frozen evaluation images, IoU 0.520 and Dice 0.575.

## Compliance gates

- No clinical or patient claim is made.
- Synthetic labels and latent variables are disclosed.
- BBBC038 is used only for the segmentation portability audit; its source,
  download URL and CC0/public-domain statement are recorded in the generated
  JSON artifact.
- No external evaluation images are committed to the repository.
- Grouped validation is described as a synthetic leakage diagnostic, not as a
  substitute for measured chip/experiment IDs.
- The public writeup uses the cleaned, metrics-aligned content represented by
  `docs/KAGGLE_WRITEUP.md`.

## Final state

The Kaggle editor was complete (7/7 required items), and the video plus links
were present. The cleaned public description was submitted and visibly
verified after pressing **Update Submission**. Future public edits require a
fresh action-time confirmation before changing the Kaggle writeup.
