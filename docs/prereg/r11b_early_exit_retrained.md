# R11b Early-exit triage with a twin retrained for temporal masking — preregistered protocol

Registered before any R11b model is trained. The SHA-256 of this file is stored in
`results/r11b_early_exit.json` and in `colab/decisions.md`.

## Why (stated honestly)
R11 (`docs/prereg/r11_early_exit.md`, verified independently) found early exit feasible. However, the
v1 twin, which was trained only with full DIV 5-12 context, was worse than the persistence rule (inactive
exit rate 0.46 vs 0.92 at the nested 95 % sensitivity threshold). The v1 twin never saw
episodes with hidden late DIVs. R11b asks whether training on such episodes closes the gap. It is a
follow-up registered **after** seeing R11. R11 remains the primary early-exit result, and R11b is
reported as a second, registered look.

## Models (only one change from v1)
- Same data, same outer folds 1-4, same training set per fold (every chemical not in fold f), same
  per-fold configuration as the v1 checkpoint of that fold (read from `cnp_fold{f}_seed0.pt`), same
  seeds 0/1/2, same optimiser, steps and context sampling (`TrainConfig` defaults).
- **Change:** in each training episode, with probability p = 0.5 (fixed a priori), the context wells
  have DIV 9 and DIV 12 hidden (values and mask set to 0; interpolation features recomputed from the
  masked context). The targets are always the full DIV 5-12 trajectories of every well.
- Saved to `data/processed/models_r11b/`. GPU runs go through the shared queue
  (`D:/PROJECTS/.cognition/gpu_queue/gpuq.py`).

## Evaluation (identical to R11)
Same cohort (194 chemicals), same early-exit inputs (DIV 5/7 only), same reference, same four methods,
where "twin" and "twin_upper90" now use the R11b models, and the same nested thresholds, endpoints,
bootstrap and decision rule as R11.

## Additional endpoints
1. **Twin R11b vs twin v1:** paired bootstrap of the inactive exit rate (R11b minus v1), 95 % CI.
2. **Late-only reference (circularity control):** every method re-evaluated against the activity call
   computed from the DIV 9 and DIV 12 measurements only (same BMR rule), with nested thresholds
   recomputed on that reference. It is reported next to the main reference and does not replace it.
3. **No harm to dose interpolation (non-inferiority):** with full DIV context, the k = 3 curve MAE
   (same 5 designs per chemical as `trajectory_cv`) of the R11b ensemble vs the v1 ensemble on the
   194 chemicals, as a paired bootstrap of the relative change. R11b is acceptable as the delivered
   model only if the upper CI bound is <= +2 %. Otherwise v1 stays the delivered forecaster and R11b
   is used only for early exit.

## Decision rule (written in advance)
- If R11b meets the R11 success rule (CI of twin R11b vs best baseline excludes 0 and the safety
  condition holds), the report recommends the R11b twin for early exit. Otherwise the report
  recommends the persistence rule and states that temporal-masking training did not make the twin
  better than the simple rule.
- All results are reported, positive or negative. Nothing is tuned after seeing R11b numbers.
