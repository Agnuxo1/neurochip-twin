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
- Kaggle code mirror: [CPU-only synthetic reproduction notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo)
- Technical report: [technical report](https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/TECHNICAL_REPORT.md)
- Video file mirror: `outputs/demo/neurochip_twin_demo.mp4`

## Live Kaggle audit — 2026-09-23

- The public writeup displays **Submitted!** and is dated Sep 21, 2026. This
  confirms a writeup is present, not that all eligibility gates are complete.
- Its visible Project Summary is 516 whitespace-delimited words. Kaggle's
  recommended structure asks for 200–300 words. The current writeup also does
  not declare **End-to-End System** at the beginning of its body.
- Kaggle's overview reports 11 teams and 11 submissions. The event uses expert
  review after the Oct 10 deadline (Oct 10–20), then selects 20 finalists; no
  official first/second-place ranking is published yet. Kaggle's notebook
  settings for this event show no scored submissions; the code notebook is a
  reproducibility/code-sharing artifact, not a leaderboard entry.
- The judging rubric weights impact 30%, technical approach 30%, results and
  validation 20%, reproducibility 10%, and presentation 10%. The project's
  largest scientific limitation remains the lack of paired neural OoC response
  validation; synthetic benchmark metrics do not fill that gap.
- The embedded YouTube demo is 38 seconds. The separately linked GitHub MP4 is
  4.5 seconds. The 132-second draft noted in an earlier checkpoint is not
  present in this recovered checkout and must not be described as available.
  The public video's runtime/access were checked, but its content has not yet
  been reviewed end-to-end against the required workflow/results narrative.
- Kaggle's foundational code-sharing rule requires code shared publicly during
  the competition to also be shared in its Kaggle discussion or code page.
  NeuroChip Twin now has a public, competition-linked Kaggle notebook that
  embeds the core implementation and uses no Internet, GPU, external dataset
  or paid service. Kaggle displays the notebook under Apache 2.0; the GitHub
  repository remains MIT. Version 2 completed a successful 49-second Kaggle
  run with synthetic ROC-AUC 0.990 and F1 0.958; this is not biological
  validation. The current hotness-sorted listing shows BioFluidNet-OoC at 15
  votes and NeuroChip Twin at 0 votes; these are votes, not judged placements.
- Kaggle separately requires the team leader to submit its registration form.
  The prior local audit recorded required form fields as blank on Sep 21;
  current completion is unverified. Do not treat the writeup's Submitted state
  as proof of form completion.
- The current repository main is commit `9924768` (Sep 21). The earlier dirty
  checkout and its Quris audit scripts/results were not present at the recorded
  path on Sep 23. Do not publish those prior Quris metrics from memory; recover
  their provenance or rerun an explicitly exploratory audit before reuse.

The replacement summary prepared in this local checkout starts with the
category declaration and is 244 words. It is not uploaded to Kaggle.

## Eligibility gate

Kaggle's official competition page requires the team leader to complete the
[official registration form](https://docs.google.com/forms/d/e/1FAIpQLSdRAat5jIunRaFNh_NntsVeJUnekEJDrbuokLZ32LFgCwPtiA/viewform)
before the Writeup is eligible for evaluation. On 2026-09-21 the signed-in
form still showed all required fields empty, so registration is **pending**.
The automated workflow has not entered or transmitted personal data such as
WhatsApp details, travel preference or contact information.

## Evidence to report

The current public repository is reproducible from commit `3349f89` or later.
The primary benchmark is explicitly labelled synthetic:

- seed 42, 180 sequences, 25% held out;
- multimodal ROC-AUC 0.990, F1 0.958;
- viability R² 0.957 and IC50 R² 0.957;
- ten-seed random audit: ROC-AUC 0.987, F1 0.936;
- ten-seed grouped audit: ROC-AUC 0.983, F1 0.944;
- ten-seed compound-holdout audit: additive multimodal ROC-AUC 0.991 ± 0.007, F1 0.954 ± 0.028; explicit-interaction ablation ROC-AUC 0.986, F1 0.938;
- multimodal Brier/ECE: 0.041/0.064 on seed 42, 0.053/0.069 random-audit mean, and 0.048/0.071 grouped-audit mean;
- BBBC038 front-end audit: 12 calibration and 24 frozen evaluation images, IoU 0.520 and Dice 0.575.
- Public OOC metadata audit: 3,072 non-empty rows across six cell-line
  categories; this is domain-coverage evidence only, not biological validation.

## Compliance gates

- No clinical or patient claim is made.
- Synthetic labels and latent variables are disclosed.
- BBBC038 is used only for the segmentation portability audit; its source,
  download URL and CC0/public-domain statement are recorded in the generated
  JSON artifact.
- No external evaluation images are committed to the repository.
- Grouped validation is described as a synthetic leakage diagnostic, not as a
  substitute for measured chip/experiment IDs.
- The Zenodo OOC spreadsheet is used only for provenance, coverage and
  missingness planning; its good/bad sample-quality labels are not response
  labels, and the 6.7 GB image archive is not committed.
- The public writeup uses the cleaned, metrics-aligned content represented by
  `docs/KAGGLE_WRITEUP.md` only in this local recovery checkout. The currently
  published writeup still has the longer, older summary described above.

## Final state

The public Kaggle writeup is present but needs a compliant summary/category
refresh. The Kaggle code-sharing gate is now covered by the public notebook;
the separate registration status is still unverified. The recovered source
checkout is at `work/neurochip-twin-recovery-20260923`. Its pending project
changes include the notebook source/metadata, an exact-source-parity test,
README links, and updated competition/validation documentation; `.cognition/`
contains local reference and run-output evidence. Fifteen tests pass,
`compileall` passes, and `git diff --check` passes. No GitHub push or writeup
edit was made. The public writeup must not be edited or submitted until the
required immediate action-time confirmation is obtained.
