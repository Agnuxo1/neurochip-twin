# R11 Early-exit triage — preregistered protocol

Registered before any R11 result is computed. The SHA-256 of this file is stored in
`results/r11_early_exit.json` (`protocol_sha256`) and in `colab/decisions.md`.

## Question
Can the NeuroTrajectory twin, given only the **early recordings (DIV 5 and DIV 7)** of a chemical, decide
which chemicals the complete assay (DIV 5-12) would call **inactive**, so that their culture and
recordings can stop at DIV 7 while (almost) every active chemical continues to DIV 12?

Conceptual antecedent (cited, not re-used numerically): the early-rejection filter of
Angulo de Lafuente, Veselov & Goodman, *Speaking to Silicon*, arXiv:2601.12032 (2026), which discards
work items from early partial signals before full processing. Here the idea is tested on real biology,
where early developmental recordings are expected to carry information about the endpoint.

## Data and split
- EPA NFA dose x DIV data, frozen chemical folds; **outer folds 1-4 only** (194 chemicals never used
  for model selection), same cohort as R2 (`trajectory_cv.json`).
- Primary models: v1 cross-validated NeuroTrajectory models (`load_fold_models(fold)`), ensemble mean,
  each chemical scored only by the models of its own held-out fold. No retraining.

## Reference ("what the complete assay would say")
Chemical-level activity from **all measured concentrations and all DIVs** with the fixed a-priori rule
of `neurotwin.models.potency.chemical_potency` (BMR = 3 vehicle robust SD on the developmental summary).
Secondary reference: EPA DNT reference labels on the R4 cohort (97 chemicals) — reported, not used to
choose anything.

## Early-exit inputs (identical information for every method)
For each chemical, every tested concentration, **DIV 5 and DIV 7 wells only**; DIV 9 and DIV 12 are
masked (`cm = 0`) and never read.

## Methods (scores; higher = more likely active)
1. **Twin (primary).** Context = all tested concentrations with DIV 9/12 masked; the twin forecasts the
   full DIV 5-12 trajectory on the R4 grid; score = max over features of |A_f| of the forecast
   developmental summary, divided by BMR (same summary as the reference rule).
2. **Twin + uncertainty (secondary).** Same, using the upper 90 % band of the forecast
   (mu + 1.645 sigma on |A|); conservative variant for triage.
3. **Early-raw baseline.** The reference rule applied to the measured DIV 5/7 data only (summary over
   the two available DIVs).
4. **Persistence baseline.** DIV 9 and DIV 12 filled with the DIV 7 measurement (last observation
   carried forward), then the reference rule.

## Triage rule and threshold (nested, no peeking)
For each outer fold f in 1-4, the threshold tau_f for each method is the **largest** threshold such
that, on the chemicals of the other three outer folds, sensitivity for reference-active chemicals is
>= 0.95. Chemicals of fold f with score < tau_f are "exit at DIV 7". (Threshold chosen on other folds'
held-out scores; the models of those folds never saw those chemicals either.)

## Endpoints
- **Primary:** fraction of reference-inactive chemicals that exit at DIV 7 (specificity at the nested
  95 %-sensitivity threshold), twin vs the best of the two baselines; paired chemical bootstrap
  (10 000 resamples) of the difference, 95 % CI.
- **Co-primary safety:** realised sensitivity (reference-active chemicals continuing), with a
  Clopper-Pearson 95 % CI. Success requires realised sensitivity CI lower bound >= 0.85.
- **Secondary:** AUROC of each score vs the reference (bootstrap CI); overall fraction of chemicals
  exiting; operational saving = exiting chemicals x (2 of 4 recording days, DIV 12 - DIV 7 = 5 of 12
  culture days), stated as assumptions, not measured costs; same analysis on the EPA DNT labels (R4
  cohort); twin + uncertainty variant.

## Decision rule (written in advance)
- Twin better than the best baseline if the paired-bootstrap 95 % CI of the primary difference
  excludes 0 and the safety condition holds. Otherwise the report says "no gain over the simple rule"
  and the simple rule is recommended for early exit.
- If the reference has fewer than 15 inactive chemicals among the 194, the primary endpoint is
  reported as underpowered and only AUROC is interpreted.
- All results reported, positive or negative. Nothing tuned after seeing R11 numbers.
