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

- The public writeup still displays **Submitted!** and is dated Sep 21, 2026.
  This confirms a writeup is present, not that all eligibility gates are
  complete. A fresh authenticated read-only check on Sep 23 found the same
  older content; no editor or submission action was opened.
- Its visible Project Summary is the old, long synthetic/BBBC038 summary and
  does not declare **End-to-End System** at the beginning of the body, although
  Kaggle's rules require the category first and recommend a 200–300-word
  summary. The replacement in `docs/KAGGLE_WRITEUP.md` now has the category
  first and a 235-word Project Summary; it remains local pending a fresh,
  immediate confirmation before public editing and **Update Submission**.
- Kaggle's overview reports 11 teams and 11 submissions. The event uses expert
  review after the Oct 10 deadline (Oct 10–20), then selects 20 finalists; no
  official first/second-place ranking is published yet. Kaggle's notebook
  settings for this event show no scored submissions; the code notebook is a
  reproducibility/code-sharing artifact, not a leaderboard entry.
- The judging rubric weights impact 30%, technical approach 30%, results and
  validation 20%, reproducibility 10%, and presentation 10%. The project's
  largest scientific limitation remains the lack of paired neural OoC response
  validation; synthetic benchmark metrics do not fill that gap.
- An unauthenticated browser could load the public YouTube video; it is 38
  seconds and has no subtitles. A playback sample visibly presents a synthetic
  dose-response frame, cell count, dose, Hungarian tracking, fixed-reservoir
  readout and a no-clinical-claim notice. It is a genuine project output, but
  the short clip does not clearly walk through input → run → report/tests, so a
  fuller captioned workflow demo remains worthwhile. The GitHub MP4 is only
  4.5 seconds (36 frames at 8 fps) and is not that public 38-second video. The
  older 132-second draft mentioned in an earlier checkpoint is absent and must
  not be described as available.
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
category declaration and is 235 words. It is not uploaded to Kaggle.

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
- Historical BBBC038 pixel audit: 12 calibration and 24 evaluation images,
  IoU 0.520 and Dice 0.575.
- Exploratory BBBC038 instance audit: 40 calibration / 80 held-out images;
  watershed mean instance F1 (IoU 0.50–0.95) 0.3325 vs connected components
  0.3207; paired delta +0.0118, image-bootstrap 95% interval +0.0013 to
  +0.0226. Pixel Dice is unchanged at 0.5242. This is segmentation portability
  only, not OoC response validation.
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
- The Zenodo OOC image archive is used only in a separate exploratory
  good/bad sample-quality audit; those labels are not response labels. The
  6.7 GB image archive, per-image predictions and model weights are not
  committed. The source record/paper license discrepancy is disclosed in the
  technical report and Kaggle writeup draft.
- The public writeup uses the cleaned, metrics-aligned content represented by
  `docs/KAGGLE_WRITEUP.md` only in this local recovery checkout. The currently
  published writeup still has the longer, older summary described above.

## Final state

The public Kaggle writeup is present but needs a compliant summary/category
refresh. The Kaggle code-sharing gate is now covered by the public notebook;
the separate registration status is still unverified. The recovered source
checkout is at `work/neurochip-twin-recovery-20260923`. Commit `20973b9`
publishes the competition-linked notebook source/metadata, exact-source-parity
test, README links, and updated competition/validation documentation to
GitHub. `.cognition/` contains local reference and run-output evidence and is
not published. The test suite passes, `compileall` passes, and
`git diff --check` passes. No Kaggle writeup edit was made. The public writeup
must not be edited or submitted until the required immediate action-time
confirmation is obtained.
