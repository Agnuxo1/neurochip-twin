# Figure captions — NeuroChip Twin v2 technical report

Numbers use the double-brace template syntax of `scripts/render_docs.py` (file, colon, dotted path, optional
`| format`) and resolve against `repo/results/`. Values whose source keys contain `.` or `=` (e.g. `alpha_0.1`, `B=3`, `AUC.3hit`) are
mirrored verbatim by `scripts/make_figures.py` into `results/figures_derived.json`, together with the quantities
the figures themselves derive (example selection, counts). `python scripts/make_figures.py` regenerates every
figure and writes `captions_rendered.md` next to this file; it fails if any reference does not resolve.
Figure files: `report/figures/<stem>.pdf` (vector) and `<stem>.png` (200 dpi), 6.5 in wide.

## fig1_system

**Figure 1 | NeuroChip Twin v2 from data asset to decisions.** Public, licensed inputs (left) become one frozen,
provenance-tracked data asset: the EPA Network Formation Assay (rat cortical networks on 48-well MEA recorded at
DIV 5, 7, 9 and 12; 17 network features; {{ trajectory_cv.json : n_chemicals }} chemicals and
{{ figures_derived.json : fig1.n_wells }} exposed wells) and the Brewer four-compartment hippocampal MEA
({{ data_audit_brewer.json : n_recordings }} recordings, {{ data_audit_brewer.json : n_axons }} sorted tunnel axons,
{{ data_audit_brewer.json : license }}). Cohort, quality-control rules and all splits are frozen by SHA-256, and
evaluation protocols are fixed before they are run. NeuroTrajectory, a few-shot conditional neural process meta-trained
across chemicals, turns k measured concentrations of an unseen chemical into a forecast of its whole
dose × DIV × feature trajectory with a predictive scale. Cross-conformal Mondrian calibration (cytotoxicity regime × k)
makes the intervals valid and an epistemic score triggers abstention. Downstream, the twin gives a developmental
neurotoxicity (DNT) hazard call from three concentrations, potency (benchmark concentration, BMC, by interpolation
of the measured doses; the design chosen after the negative result in Fig. 6c) and DoseCompass, which proposes the
next concentration by expected information and closes the laboratory loop (dashed). The chip layer estimates
directed propagation per tunnel on the real four-compartment array; it is descriptive and is not calibrated across
platforms. Reference data (EPA DNT labels and hit-count rules, invitrodb cytotoxicity, Harrill imaging, Kosnik acute
MEA) are used for evaluation only. Box colour marks the block's role (legend): grey, data or reference; blue, twin
components; orange, decisions and outputs.

## fig2_data

**Figure 2 | The two data assets.** (a) Share of exposed wells in which each NFA feature is defined, by DIV
(unlabelled cells: defined in every well). Burst and network-spike descriptors are undefined in wells without bursts or network
events, most often at DIV 5 (minimum {{ figures_derived.json : fig2.availability_min_pct | .0f }}%); undefined
values are masked, never imputed. (b) Tested concentrations per chemical
({{ figures_derived.json : fig2.levels_per_chemical.7 }} of {{ figures_derived.json : fig2.n_chemicals }}
chemicals have seven). (c) Replicate wells per chemical × concentration (median
{{ figures_derived.json : fig2.median_replicate_wells_per_level | .0f }};
{{ figures_derived.json : fig2.n_wells }} exposed wells; vehicle wells are used only for normalisation).
(d) Electrode map of one Brewer array (NoStim recording {{ figures_derived.json : fig2.chip_recording_shown.fid }},
{{ figures_derived.json : fig2.chip_recording_shown.orientation }} layout; counterclockwise arrays are mirrored):
compartments EC, DG, CA3 and CA1 with {{ figures_derived.json : fig2.electrodes_per_compartment.EC }} recording
electrodes each (dots), joined by {{ data_audit_brewer.json : tunnels_per_recording }} microfluidic tunnels
({{ figures_derived.json : fig2.tunnels_per_pair.EC_DG }} per compartment pair; grey bars with an electrode at each
end) that hold single axons; blue arrows mark the feed-forward direction on each tunnel bundle,
EC → DG → CA3 → CA1 → EC. Source: Zenodo 10257483.

## fig3_forecast_main

**Figure 3 | Few-shot forecasting of unseen chemicals (primary result).** (a) Curve MAE (error of the forecast
against the mean of the held-out wells at each unmeasured concentration, in vehicle-SD units) as a function of the
number k of measured concentrations, for NeuroTrajectory and five baselines that receive exactly the same wells;
all {{ trajectory_cv.json : n_chemicals }} chemicals, nested five-fold chemical cross-validation, five seeded designs
per k; bars are 95% bootstrap CIs over chemicals. (b) Paired difference NeuroTrajectory minus the best baseline at
each k (the best baseline is re-chosen for every k; below zero favours the model), for the primary v1 run on all
folds, the same run on folds 1–4 that were never inspected during development, and the v2 grid that adds a
dose-smoothness hyperparameter (robustness). At k = 3 on folds 1–4 the curve MAE is
{{ trajectory_cv.json : by_k_folds_1to4.3.neurotrajectory | .3f }} against
{{ trajectory_cv.json : by_k_folds_1to4.3.baseline | .3f }} for log-linear interpolation, a difference of
{{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.mean_diff | .3f }} (95% CI
{{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.ci95.0 | .3f }} to
{{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.ci95.1 | .3f }};
{{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.rel_change_pct | abs .1f }}% lower), with the model better
for {{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.frac_chem_improved | pct .1f }}% of
{{ trajectory_cv.json : by_k_folds_1to4.3.paired_vs_best.n_chemicals }} chemicals; the v2 grid gives
{{ trajectory_cv_v2.json : by_k.3.curve_mae.paired_vs_best.mean_diff | .3f }}. With no measured concentration
(k = 0, population prior) the gain over analog kNN is negligible
({{ trajectory_cv.json : by_k.0.curve_mae.paired_vs_best.mean_diff | .3f }}, 95% CI
{{ trajectory_cv.json : by_k.0.curve_mae.paired_vs_best.ci95.0 | .3f }} to
{{ trajectory_cv.json : by_k.0.curve_mae.paired_vs_best.ci95.1 | .4f }}). (c) Per-chemical curve MAE at k = 3 on
folds 1–4 (log axes; triangles: model worse than interpolation). Circles mark the median and 10th-percentile
chemicals of this distribution (distribution reference only; Fig. 4 shows three chemicals chosen by its own,
separately stated percentile rule).

## fig4_examples

**Figure 4 | Three held-out chemicals chosen by a stated percentile rule.** Rule, fixed before plotting: on outer
folds 1–4 at k = 3, the per-chemical gain is the {{ figures_derived.json : fig4.best_baseline }} curve MAE minus the
NeuroTrajectory curve MAE (mean over the five seeded designs). For each of the 90th, 50th and 10th percentiles of
this gain we take the chemicals within two percentile points of that quantile and prefer, among them, one with a
reference EPA label (positive or negative, not unknown), breaking ties by distance to the quantile value: 90th
percentile {{ figures_derived.json : fig4.p90.chemical }} (gain {{ figures_derived.json : fig4.p90.improvement | +.3f }},
EPA label {{ figures_derived.json : fig4.p90.epa_label }}), median {{ figures_derived.json : fig4.median.chemical }}
(gain {{ figures_derived.json : fig4.median.improvement | +.3f }}, EPA label
{{ figures_derived.json : fig4.median.epa_label }}), 10th percentile
{{ figures_derived.json : fig4.p10.chemical }} (gain {{ figures_derived.json : fig4.p10.improvement | +.3f }}, EPA
label {{ figures_derived.json : fig4.p10.epa_label }}). All three preferred chemicals within their window carry a
reference label. Each is shown with its median design (by curve MAE among its five seeded designs) — never the
worst — so none of the three panels is cherry-picked for difficulty. Rows are three features fixed across all three
chemicals (mean firing rate, burst rate, network spikes), not re-selected per chemical. Black: wells at the three
concentrations given to the model; orange open circles: held-out wells, revealed after forecasting; blue: ensemble
forecast (mean) with its {{ figures_derived.json : fig4.band }} band
({{ figures_derived.json : fig4.conformal_halfwidth_vehicle_sd | .2f }} vehicle SD half-width, pooled across
Mondrian strata; a chemical- or regime-specific band is not available); dashed: log-linear interpolation of the
measured concentration means. Units: vehicle robust SD, clipped at ±10. The forecasts use the three models of each
chemical's cross-validation fold, which never saw it (CPU bundle); re-scoring all five designs reproduces the
cross-validation values within the reported tolerance
({{ figures_derived.json : fig4.p90.recomputed_mean5_neurotrajectory | .3f }},
{{ figures_derived.json : fig4.median.recomputed_mean5_neurotrajectory | .3f }} and
{{ figures_derived.json : fig4.p10.recomputed_mean5_neurotrajectory | .3f }}). The 10th-percentile case shows a
characteristic error: when the measured concentrations sit in a flat part of the curve, the forecast above the
highest one reverts toward the population's typical high-dose decline, whereas the held-out wells stay near vehicle.

## fig5_calibration

**Figure 5 | Calibrated uncertainty and abstention.** (a) Observed minus nominal coverage of held-out wells for the
ensemble's parametric intervals (open) and after cross-conformal Mondrian calibration (filled), at three nominal
levels and k = 1–3; 95% bootstrap CIs over chemicals. At 90% nominal and k = 3, coverage moves from
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.parametric_coverage | pct .1f }}% to
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.conformal_coverage | pct .1f }}% (95% CI
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.conformal_ci95.0 | pct .1f }}–{{ figures_derived.json : fig5.by_nominal.nominal90.k3.conformal_ci95.1 | pct .1f }}%).
(b) Conformal coverage per Mondrian stratum (chemical-level cytotoxicity regime from the AlamarBlue/LDH calls of the
NFA assay; N in the legend) at 90% nominal. (c) The price of validity: mean width of the 90% intervals, wider for
cytotoxic and indeterminate chemicals than for non-cytotoxic ones (k = 3:
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.by_regime.cytotoxic.width | .2f }},
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.by_regime.indeterminate.width | .2f }} and
{{ figures_derived.json : fig5.by_nominal.nominal90.k3.by_regime.non_cytotoxic.width | .2f }} vehicle SD); the
pooled parametric width is shown for reference. (d) Abstention: forecasts
whose epistemic score exceeds the 90th percentile of the calibration folds are flagged
({{ conformal_r7.json : abstention.abstention_rate | pct .1f }}% overall); their curve MAE is higher than that of
the retained forecasts ({{ conformal_r7.json : abstention.curve_mae_abstained | .2f }} against
{{ conformal_r7.json : abstention.curve_mae_retained | .2f }}).

## fig6_hazard

**Figure 6 | A hazard call from fewer wells, and a negative potency result.** (a) Sensitivity and specificity
(Clopper–Pearson 95% CIs) of the chemical-level DNT call on the {{ figures_derived.json : fig6.n_chemicals_mcnemar }}
reference chemicals shared with the reproduced EPA hit-count rules. The pre-specified NeuroTrajectory variant uses
trajectory summaries forecast from only three measured concentrations; the all-doses variant uses every tested
concentration; the EPA rules are reproduced exactly ({{ epa_baseline_repro.json : models.DIV12.published_delta_pp.balanced_accuracy_pp | .1f }} percentage-point difference in DIV12 balanced accuracy from the published table), but
EPA's Top-k features were selected in-sample. (b) Exact McNemar tests on the discordant chemicals: k = 3 against
AUC.3hit p = {{ figures_derived.json : fig6.mcnemar_k3.AUC3hit.exact_p | .3f }}, against DIV12
p = {{ figures_derived.json : fig6.mcnemar_k3.DIV12.exact_p | .3f }}, against Top2
p = {{ figures_derived.json : fig6.mcnemar_k3.Top2.exact_p | .3f }}. These are three uncorrected comparisons with
{{ dnt_r4.json : n_negative }} negative chemicals, so no general superiority is claimed. (c) Potency: paired
difference in composite potency error (|ΔBMC| in log10 units when both calls are active, 1 on activity
disagreement) between model-derived BMCs and interpolation of the measured doses, scored against the
replicate-split reference built from independent wells. Both the primary protocol (R3 v2, BMC from the model's dense
dose grid) and the registered secondary (R3b, measured doses completed by the model) are worse than interpolation at
every k (k = 3: {{ potency_r3.json : by_k.3.replicate_split_paired_vs_all.loglinear_interp.mean_diff | +.3f }},
95% CI {{ potency_r3.json : by_k.3.replicate_split_paired_vs_all.loglinear_interp.ci95.0 | .3f }} to
{{ potency_r3.json : by_k.3.replicate_split_paired_vs_all.loglinear_interp.ci95.1 | .3f }}; R3b
{{ potency_r3b.json : by_k.3.fill_paired_vs.loglinear_interp.mean_diff | +.3f }}), so the system reports BMCs by
interpolating measured doses. An earlier reference built from interpolated level means favoured interpolation by
construction; that analysis is superseded and reported only in the appendix. (d) AUROC of the two variants on all
{{ dnt_r4.json : n_positive }} positive and {{ dnt_r4.json : n_negative }} negative reference chemicals (bootstrap
95% CI): k = 3 {{ dnt_r4.json : variants.k3.auroc | .3f }}
({{ dnt_r4.json : variants.k3.auroc_ci95.0 | .3f }}–{{ dnt_r4.json : variants.k3.auroc_ci95.1 | .3f }}), all doses
{{ dnt_r4.json : variants.full.auroc | .3f }}.

## fig7_dosecompass

**Figure 7 | DoseCompass: which concentration to measure next.** (a) Example chosen by a fixed rule (the first
reference-active, DNT-positive chemical in alphabetical order with seven tested concentrations):
{{ r6_dosecompass.json : example_chemical.chemical }} after measuring its median concentration
({{ figures_derived.json : fig7.example_first_uM }} µM). Top: twin posterior of the developmental maximum |effect|
at each candidate concentration (bands) and the observed values, hidden from the planner; dotted line: benchmark
response. Bottom: expected information gain about potency per candidate; the planner proposes
{{ figures_derived.json : fig7.example_proposed_uM }} µM. (b) Composite potency error of the NeuroTrajectory
estimator by budget B for the two registered DoseCompass protocols (R6, median first, starts at the median
concentration; R6b, top first, is anchored at the highest), the fixed log-spaced design, random orders (band:
2.5–97.5% of 200 replicates) and the oracle subset (the best achievable within EPA's tested grid). (c) Paired
differences at B = 3, bootstrap 95% CIs over {{ r6_dosecompass.json : n_chemicals }} chemicals. Both primary
comparisons with the fixed log-spaced design are ties (R6, median first,
{{ r6_dosecompass.json : primary_endpoint.paired_bootstrap.mean_diff | +.3f }}, CI
{{ r6_dosecompass.json : primary_endpoint.paired_bootstrap.ci95.0 | .3f }} to
{{ r6_dosecompass.json : primary_endpoint.paired_bootstrap.ci95.1 | .3f }}; R6b, top first,
{{ r6b_dosecompass_anchored.json : primary_endpoint.paired_bootstrap.mean_diff | +.3f }}, CI
{{ r6b_dosecompass_anchored.json : primary_endpoint.paired_bootstrap.ci95.0 | .3f }} to
{{ r6b_dosecompass_anchored.json : primary_endpoint.paired_bootstrap.ci95.1 | .3f }}); R6 (median first) beats random
ordering ({{ figures_derived.json : fig7.B3_v1_paired.dosecompass_minus_random.mean_diff | +.3f }}, CI
{{ figures_derived.json : fig7.B3_v1_paired.dosecompass_minus_random.ci95.0 | .3f }} to
{{ figures_derived.json : fig7.B3_v1_paired.dosecompass_minus_random.ci95.1 | .3f }}). Two protocols were tested on
the same data without multiplicity correction, so a single interval excluding zero is weaker evidence than it
appears. (d) Activity detection at B = 3: R6b (top first) reaches accuracy
{{ r6b_dosecompass_anchored.json : secondary_endpoints.by_strategy.dosecompass_anchored.activity_accuracy | .3f }}
and kappa {{ r6b_dosecompass_anchored.json : secondary_endpoints.by_strategy.dosecompass_anchored.activity_kappa | .3f }},
against {{ r6b_dosecompass_anchored.json : secondary_endpoints.by_strategy.fixed_logspaced.activity_accuracy | .3f }}
and {{ r6b_dosecompass_anchored.json : secondary_endpoints.by_strategy.fixed_logspaced.activity_kappa | .3f }} for the
fixed design (point estimates; random designs show 2.5–97.5% of 200 replicates).

## fig8_chip

**Figure 8 | Chip layer on a real four-compartment MEA (Brewer).** (a) Feed-forward fraction of sorted tunnel axons
per compartment edge, for each recording (NoStim n = {{ r8_chiplayer.json : conditions.NoStim }}, HFS5
n = {{ r8_chiplayer.json : conditions.HFS5 }}, HFS40 n = {{ r8_chiplayer.json : conditions.HFS40 }}) and pooled
over the unstimulated arrays with Wilson 95% CIs (EC → DG
{{ figures_derived.json : fig8.pooled_nostim_ff_fraction_wilson.EC-DG.ff_fraction | .2f }}, DG → CA3
{{ figures_derived.json : fig8.pooled_nostim_ff_fraction_wilson.DG-CA3.ff_fraction | .2f }}); dashed line: no
directional bias. (b) Conduction times of the unstimulated arrays' axons per edge (quantised by the {{ data_audit_brewer.json : sample_rate_hz | , }} Hz sampling;
bar: median). (c) Lag of the peak cross-correlation between compartment population-burst rates:
{{ figures_derived.json : fig8.burst_lag_zero_n }} of {{ figures_derived.json : fig8.burst_edges_n }}
edge × recording pairs peak at zero lag (100 ms bins), i.e. synchronous activity, not evidence of directed
propagation. (d, e) Leave-one-recording-out over the {{ r8_chiplayer.json : conditions.NoStim }} unstimulated
arrays; each line joins the errors for one held-out array, bars are macro means. The empirical-Bayes shrinkage model
lowers the per-tunnel feed-forward-fraction error against pooled tunnel means by
{{ r8_chiplayer.json : cv.comparisons.tunnel_mean.ff_fraction_mae.difference_model_minus_baseline | abs .3f }}
(95% CI {{ r8_chiplayer.json : cv.comparisons.tunnel_mean.ff_fraction_mae.ci95.0 | .3f }} to
{{ r8_chiplayer.json : cv.comparisons.tunnel_mean.ff_fraction_mae.ci95.1 | .3f }}), a modest gain; against the
direction-permuted null the interval includes zero
({{ r8_chiplayer.json : cv.comparisons.permuted_direction.ff_fraction_mae.ci95.0 | .3f }} to
{{ r8_chiplayer.json : cv.comparisons.permuted_direction.ff_fraction_mae.ci95.1 | .3f }}), and there is no gain on
conduction time ({{ r8_chiplayer.json : cv.comparisons.tunnel_mean.conduction_mae_ms.difference_model_minus_baseline | +.3f }} ms,
95% CI {{ r8_chiplayer.json : cv.comparisons.tunnel_mean.conduction_mae_ms.ci95.0 | .3f }} to
{{ r8_chiplayer.json : cv.comparisons.tunnel_mean.conduction_mae_ms.ci95.1 | .3f }}).

## fig9_crossassay

**Figure 9 | Cross-assay corroboration (unpaired).** The assays differ in plates, cultures, exposure windows and
read-outs, and share no well with the NFA. (a) Harrill high-content imaging: for the
{{ x2_harrill.json : headline.n_both_active }} chemicals active in both assays (identity-level matches), the network
BMC from the NFA lies below the neurite or synapse morphology BMC, with a median offset of
{{ x2_harrill.json : headline.potency_offset_nfa_minus_morph_log10_median | +.2f }} log10 units (95% CI
{{ x2_harrill.json : headline.potency_offset_ci95.0 | +.2f }} to
{{ x2_harrill.json : headline.potency_offset_ci95.1 | +.2f }}): network function changes at lower concentrations
than morphology. (b) Kosnik acute MEA call on {{ x3_kosnik.json : headline.n_overlap_measured }} shared chemicals:
developmental NFA activity and the k = 3 twin both discriminate acute neuroactivity above chance but modestly (NFA
balanced accuracy {{ x3_kosnik.json : measured_acute_call.nfa_reference.balanced_accuracy | .2f }}, specificity
{{ x3_kosnik.json : measured_acute_call.nfa_reference.specificity | .2f }}; permutation p-values above each
estimate); acute exposure of mature networks is a different biological question from developmental exposure.
(c) Concordance statistics with bootstrap 95% CIs over chemicals: activity agreement with Harrill morphology
(Cohen's κ {{ x2_harrill.json : headline.kappa | .2f }}), potency rank correlation pooled
({{ x2_harrill.json : headline.spearman_potency_both_active | .2f }}) and per imaging family, and the acute potency
rank correlation with Kosnik ({{ x3_kosnik.json : measured_acute_call.potency_spearman_both_active.spearman | .2f }},
95% CI {{ x3_kosnik.json : measured_acute_call.potency_spearman_both_active.ci95.0 | .2f }} to
{{ x3_kosnik.json : measured_acute_call.potency_spearman_both_active.ci95.1 | .2f }}; not significant). The twin's
k = 3 activity call agrees less with Harrill morphology
(κ {{ x2_harrill.json : twin_k3_vs_harrill_primary_overlap.activity.kappa | .2f }}) because it calls more chemicals
active ({{ x2_harrill.json : twin_k3_vs_harrill_primary_overlap.activity.twin_active }} of
{{ x2_harrill.json : twin_k3_vs_harrill_primary_overlap.activity.n }}).

## fig10_ablations_failures

**Figure 10 | Ablations, failures and a negative result.** (a) One component removed at a time (v2 configuration,
same outer folds and designs, seed 0; paired bootstrap 95% CIs; above zero means the component helps). The
interpolation-informed decoder matters most, and more as k grows (k = 3:
{{ r9_ablations.json : by_k.3.no_interp_decoder_minus_full.mean_diff | +.3f }}, 95% CI
{{ r9_ablations.json : by_k.3.no_interp_decoder_minus_full.ci95.0 | .3f }} to
{{ r9_ablations.json : by_k.3.no_interp_decoder_minus_full.ci95.1 | .3f }}); dose-local attention helps at k = 3
({{ r9_ablations.json : by_k.3.no_dose_attention_minus_full.mean_diff | +.3f }}, CI
{{ r9_ablations.json : by_k.3.no_dose_attention_minus_full.ci95.0 | .3f }} to
{{ r9_ablations.json : by_k.3.no_dose_attention_minus_full.ci95.1 | .3f }}) but not detectably at k = 1–2; the
three-seed ensemble helps at k = 1–2. (b) Per-chemical gain over log-linear interpolation at k = 3 on folds 1–4:
the model is worse for {{ figures_derived.json : fig4.frac_model_worse | pct .1f }}% of chemicals, with a long tail of
large losses; dotted lines mark the median and 10th-percentile chemicals of this distribution (distribution
reference only; not necessarily the same chemicals as Fig. 4, which uses its own percentile rule). (c) Mean
curve-MAE difference by annotated chemical class
at k = 3 (annotation of the EPA reference list as distributed; classes with at least
{{ figures_derived.json : fig10.class_min_n }} chemicals; marker area grows with n; descriptive, no CI). On
average the model loses for pharmaceuticals and food additives and gains most for organochlorines, pyrethroids and
acetylcholinesterase inhibitors; {{ r10_failures.json : n_no_estimable_total }} chemicals have no estimable network
potency and are counted rather than dropped. (d) A selectivity index contrasting network potency with chemical-level
cytotoxicity does not separate DNT-positive from DNT-negative chemicals (AUROC
{{ r5_selectivity.json : auroc_selectivity_index | .2f }}, 95% CI {{ r5_selectivity.json : auroc_si_ci95.0 | .2f }}
to {{ r5_selectivity.json : auroc_si_ci95.1 | .2f }}); cytotoxic potency alone does better
({{ r5_selectivity.json : auroc_cytotoxic_potency_only | .2f }}). Reported as a negative result.

## figA2_darwin

**Figure A2 | Darwin-Cage honest-ceiling audit: searching for residual structure the twin left on the table.**
(a) Number of programs (feature combinations of the forecast residual) confirmed on unseen chemicals, for each of
the four confirmation folds (fold 4 is the pre-specified primary fold); the original z/√n significance rule alone
against the rule used, which additionally requires a chemical-level sign-flip permutation test, to control the
family-wise false-positive rate. (b) Statistical power of the search itself: with five known effects planted in the
real residuals (plate|date shift and feature × DIV × class interactions), the fraction of the {{ r10_residual_audit.json : protocol.plant_reps }} plants recovered
exactly and by an oracle given the true program, against the planted effect size (vehicle SD, log axis); the dotted
line marks 80% power, the audit's own minimum detectable effect. (c) Curve-MAE change from subtracting the
confirmed Darwin program's fitted correction, against a generic learner (histogram gradient boosting on the same
prediction-time covariates, nested-selected), each vs. the uncorrected forecast, by k (paired bootstrap 95% CIs).
Both corrections make the forecast worse rather than better at every k, which is the audit's honest-ceiling
conclusion: the residual has no exploitable structure beyond what NeuroTrajectory already captures, and the search
that found "significant" residual programs does not reproduce as forecast improvement out of sample.

## figA3_viability_voi

**Figure A3 | Value of viability information for the forecast (value-of-information audit, with a stated leakage
caveat).** Curve-MAE change from adding chemical-level viability information to the nested residual corrector,
against the uncorrected forecast, by k (paired bootstrap 95% CIs); open markers mark scenario arms whose viability
read is not available strictly before the forecast (see caveat below). (a) All
{{ figures_derived.json : figA3.all_chemicals_n }} chemicals: V0, no viability (closed circles), against V1, PubChem
AB/LDH cytotoxicity calls aggregated over all tested concentrations including the ones the forecast hides (open
squares, scenario). (b) The {{ figures_derived.json : figA3.ntp_subset_n }} chemicals with an independent NTP
per-well viability read: V0 (no viability), V1 (PubChem AB/LDH, scenario), V2 (NTP chemical-level summary, scenario)
and V3 (NTP viability from only the wells at the measured context concentrations — the one leakage-free arm).
Caveat: {{ figures_derived.json : figA3.leakage_caveat }} V1 and V2 are therefore not evidence that viability
improves the forecast; V3, which shows no reliable gain either
({{ figures_derived.json : figA3.v1_vs_v0_k3.mean_diff | +.3f }} for V1 vs V0 at k = 3, for reference), is the only
arm free of that caveat.

## Reserved (not yet generated)

- `figA1_own_arch` — own prior architectures evaluated on real data (source `results/appendix_own_arch.json`, being computed).
