# Kaggle submission checklist

This checklist records the state of the AI4S Open Innovation: AI for Life Science
writeup and its reproducibility evidence.

## Submission fields

- Category: **End-to-End System**
- Title: `NeuroChip Twin: interpretable temporal OoC digital twin`
- Subtitle: `Physics-informed temporal microscopy analysis for toxicity, viability and dose-response screening`
- Competition: [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien)
- Writeup editor: [NeuroChip Twin writeup](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/new-writeup-1789975521348)
- Current YouTube demo (38 seconds): [public video](https://www.youtube.com/watch?v=U0zyU-AvXo8)
- New judge demo included in this repository release: `outputs/demo/neurochip_twin_judges_demo.mp4` plus SRT captions
- Code: [public GitHub repository](https://github.com/Agnuxo1/neurochip-twin)
- Kaggle code mirror: [CPU-only synthetic reproduction notebook](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo)
- Technical report: [technical report](https://github.com/Agnuxo1/neurochip-twin/blob/main/docs/TECHNICAL_REPORT.md)
- Video file mirror: `outputs/demo/neurochip_twin_demo.mp4`

## Live Kaggle audit — 2026-09-23

- The authenticated writeup page displays **Submitted!** (Sep 21, 2026) and
  remains editable until Oct 10. Its body still has the earlier long Project
  Summary, does not begin with the required **End-to-End System** category,
  and does not include the latest OoC image-quality audit. The corrected
  `docs/KAGGLE_WRITEUP.md` remains a local draft; the public page was not edited.
- Its visible Project Summary is the old, long synthetic/BBBC038 summary and
  does not declare **End-to-End System** at the beginning of the body, although
  Kaggle's rules require the category first and recommend a 200–300-word
  summary. The replacement in `docs/KAGGLE_WRITEUP.md` now has the category
  first and a 294-word Project Summary; it remains local pending a fresh,
  immediate confirmation before public editing and **Update Submission**.
- Kaggle's Writeups page lists ten submitted team entries as of this check,
  but the other teams' contents are marked **Viewable at Hackathon close**.
  There is no public first/second-place ranking to analyze. The Kaggle CLI
  reports no scored submissions; this is an expert-reviewed writeup challenge,
  not a prediction leaderboard. Do not infer placement from votes or team count.
- The judging rubric weights impact 30%, technical approach 30%, results and
  validation 20%, reproducibility 10%, and presentation 10%. The project's
  largest scientific limitation remains the lack of paired neural OoC response
  validation; synthetic benchmark metrics do not fill that gap.
- The current public YouTube demo is 38 seconds and has no subtitles; the old
  GitHub MP4 is only 4.5 seconds. A new 107-second, English, captioned
  judge demo and SRT have now been rendered and are included in this repository
  release. It shows the synthetic workflow, the separate OoC image-quality
  audit, and BBBC038 segmentation with scopes explicitly labelled. It has not
  replaced the YouTube video embedded in the Kaggle writeup.
- Kaggle's foundational code-sharing rule requires code shared publicly during
  the competition to also be shared in its Kaggle discussion or code page.
  NeuroChip Twin now has a public, competition-linked Kaggle notebook that
  embeds the core implementation and uses no Internet, GPU, external dataset
  or paid service. Kaggle displays the notebook under Apache 2.0; the GitHub
  repository remains MIT. Version 2 completed a successful 49-second Kaggle
  run with synthetic ROC-AUC 0.990 and F1 0.958; this is not biological
  validation. The current hotness-sorted listing shows BioFluidNet-OoC at 15
  votes and NeuroChip Twin at 0 votes; these are votes, not judged placements.
- Kaggle requires the team leader to submit the registration form. A read-only
  check on Sep 23 shows all required fields are still blank, including WhatsApp
  and Guangzhou travel preference. No personal data was entered or transmitted;
  **Submitted!** on the writeup is not proof of eligibility.
- Before this repository release, remote GitHub `main` was commit `bd90a3b`.
  Quris audit scripts/results are not present in this checkout. Do not publish
  any prior Quris metrics from memory; recover their provenance or rerun an
  explicitly exploratory audit before reuse.

The replacement summary prepared in this local checkout starts with the
category declaration. It is not uploaded to Kaggle; editing the public writeup
and pressing **Update Submission** require immediate action-time confirmation.

## Eligibility gate

Kaggle's official competition page requires the team leader to complete the
[official registration form](https://docs.google.com/forms/d/e/1FAIpQLSdRAat5jIunRaFNh_NntsVeJUnekEJDrbuokLZ32LFgCwPtiA/viewform)
before the Writeup is eligible for evaluation. On 2026-09-23 the signed-in
form still showed all required fields empty, so registration is **pending**.
No personal data, WhatsApp details or travel preference have been entered or
transmitted.

## Evidence to report

The public repository's previous base commit was `bd90a3b`; this release adds
the demo, aggregate audit artifact, exporter and integrity checks.
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
- `outputs/ooc_quality_public_summary.json` contains aggregate-only audited
  scores; images, per-image predictions and model weights are omitted.
- The new judge demo is included in this GitHub release; the writeup still
  embeds the earlier 38-second YouTube video and has not been updated.
- The public writeup uses the cleaned, metrics-aligned content represented by
  `docs/KAGGLE_WRITEUP.md` only in this local recovery checkout. The currently
  published writeup still has the longer, older summary described above.

## Final state

The public Kaggle writeup is present but needs a compliant summary/video
refresh, and registration remains pending. Competitor writeups are hidden until
the event closes, so no verified project ranking is available. Local source,
tests and public-safe aggregate results are included in this repository
release; `.cognition/` contains private audit files and is not published. The
new judge video is in GitHub, while the YouTube channel and Kaggle embed still
show the earlier video.
The Kaggle writeup was not modified; a public edit and **Update Submission**
require immediate action-time confirmation.
