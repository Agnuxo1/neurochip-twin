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
DIV 5, 7, 9 and 12; 17 network features; 243 chemicals and
6902 exposed wells) and the Brewer four-compartment hippocampal MEA
(21 recordings, 440 sorted tunnel axons,
CC0-1.0). Cohort, quality-control rules and all splits are frozen by SHA-256, and
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
events, most often at DIV 5 (minimum 20%); undefined
values are masked, never imputed. (b) Tested concentrations per chemical
(212 of 243
chemicals have seven). (c) Replicate wells per chemical × concentration (median
3;
6902 exposed wells; vehicle wells are used only for normalisation).
(d) Electrode map of one Brewer array (NoStim recording 1,
clockwise layout; counterclockwise arrays are mirrored):
compartments EC, DG, CA3 and CA1 with 19 recording
electrodes each (dots), joined by 20 microfluidic tunnels
(5 per compartment pair; grey bars with an electrode at each
end) that hold single axons; blue arrows mark the feed-forward direction on each tunnel bundle,
EC → DG → CA3 → CA1 → EC. Source: Zenodo 10257483.

## fig3_forecast_main

**Figure 3 | Few-shot forecasting of unseen chemicals (primary result).** (a) Curve MAE (error of the forecast
against the mean of the held-out wells at each unmeasured concentration, in vehicle-SD units) as a function of the
number k of measured concentrations, for NeuroTrajectory and five baselines that receive exactly the same wells;
all 243 chemicals, nested five-fold chemical cross-validation, five seeded designs
per k; bars are 95% bootstrap CIs over chemicals. (b) Paired difference NeuroTrajectory minus the best baseline at
each k (the best baseline is re-chosen for every k; below zero favours the model), for the primary v1 run on all
folds, the same run on folds 1–4 that were never inspected during development, and the v2 grid that adds a
dose-smoothness hyperparameter (robustness). At k = 3 on folds 1–4 the curve MAE is
1.194 against
1.345 for log-linear interpolation, a difference of
-0.151 (95% CI
-0.199 to
-0.100;
11.2% lower), with the model better
for 77.3% of
194 chemicals; the v2 grid gives
-0.144. With no measured concentration
(k = 0, population prior) the gain over analog kNN is negligible
(-0.013, 95% CI
-0.027 to
0.0002). (c) Per-chemical curve MAE at k = 3 on
folds 1–4 (log axes; triangles: model worse than interpolation). Circles mark the median and 10th-percentile
chemicals of this distribution (distribution reference only; Fig. 4 shows three chemicals chosen by its own,
separately stated percentile rule).

## fig4_examples

**Figure 4 | Three held-out chemicals chosen by a stated percentile rule.** Rule, fixed before plotting: on outer
folds 1–4 at k = 3, the per-chemical gain is the loglinear_interp curve MAE minus the
NeuroTrajectory curve MAE (mean over the five seeded designs). For each of the 90th, 50th and 10th percentiles of
this gain we take the chemicals within two percentile points of that quantile and prefer, among them, one with a
reference EPA label (positive or negative, not unknown), breaking ties by distance to the quantile value: 90th
percentile Heptachlor epoxide B (gain +0.472,
EPA label positive), median DDT
(gain +0.148, EPA label
positive), 10th percentile
Retinoic acid (gain -0.120, EPA
label positive). All three preferred chemicals within their window carry a
reference label. Each is shown with its median design (by curve MAE among its five seeded designs) — never the
worst — so none of the three panels is cherry-picked for difficulty. Rows are three features fixed across all three
chemicals (mean firing rate, burst rate, network spikes), not re-selected per chemical. Black: wells at the three
concentrations given to the model; orange open circles: held-out wells, revealed after forecasting; blue: ensemble
forecast (mean) with its conformal 90% (pooled across Mondrian strata, k = 3) band
(3.71 vehicle SD half-width, pooled across
Mondrian strata; a chemical- or regime-specific band is not available); dashed: log-linear interpolation of the
measured concentration means. Units: vehicle robust SD, clipped at ±10. The forecasts use the three models of each
chemical's cross-validation fold, which never saw it (CPU bundle); re-scoring all five designs reproduces the
cross-validation values within the reported tolerance
(1.741,
1.242 and
1.914). The 10th-percentile case shows a
characteristic error: when the measured concentrations sit in a flat part of the curve, the forecast above the
highest one reverts toward the population's typical high-dose decline, whereas the held-out wells stay near vehicle.

## fig5_calibration

**Figure 5 | Calibrated uncertainty and abstention.** (a) Observed minus nominal coverage of held-out wells for the
ensemble's parametric intervals (open) and after cross-conformal Mondrian calibration (filled), at three nominal
levels and k = 1–3; 95% bootstrap CIs over chemicals. At 90% nominal and k = 3, coverage moves from
88.2% to
89.9% (95% CI
89.1–90.6%).
(b) Conformal coverage per Mondrian stratum (chemical-level cytotoxicity regime from the AlamarBlue/LDH calls of the
NFA assay; N in the legend) at 90% nominal. (c) The price of validity: mean width of the 90% intervals, wider for
cytotoxic and indeterminate chemicals than for non-cytotoxic ones (k = 3:
8.05,
8.44 and
6.66 vehicle SD); the
pooled parametric width is shown for reference. (d) Abstention: forecasts
whose epistemic score exceeds the 90th percentile of the calibration folds are flagged
(10.7% overall); their curve MAE is higher than that of
the retained forecasts (1.67 against
1.25).

## fig6_hazard

**Figure 6 | A hazard call from fewer wells, and a negative potency result.** (a) Sensitivity and specificity
(Clopper–Pearson 95% CIs) of the chemical-level DNT call on the 97
reference chemicals shared with the reproduced EPA hit-count rules. The pre-specified NeuroTrajectory variant uses
trajectory summaries forecast from only three measured concentrations; the all-doses variant uses every tested
concentration; the EPA rules are reproduced exactly (0.0 percentage-point difference in DIV12 balanced accuracy from the published table), but
EPA's Top-k features were selected in-sample. (b) Exact McNemar tests on the discordant chemicals: k = 3 against
AUC.3hit p = 0.013, against DIV12
p = 0.064, against Top2
p = 0.118. These are three uncorrected comparisons with
19 negative chemicals, so no general superiority is claimed. (c) Potency: paired
difference in composite potency error (|ΔBMC| in log10 units when both calls are active, 1 on activity
disagreement) between model-derived BMCs and interpolation of the measured doses, scored against the
replicate-split reference built from independent wells. Both the primary protocol (R3 v2, BMC from the model's dense
dose grid) and the registered secondary (R3b, measured doses completed by the model) are worse than interpolation at
every k (k = 3: +0.106,
95% CI 0.050 to
0.163; R3b
+0.069), so the system reports BMCs by
interpolating measured doses. An earlier reference built from interpolated level means favoured interpolation by
construction; that analysis is superseded and reported only in the appendix. (d) AUROC of the two variants on all
79 positive and 19 negative reference chemicals (bootstrap
95% CI): k = 3 0.915
(0.853–0.965), all doses
0.871.

## fig7_dosecompass

**Figure 7 | DoseCompass: which concentration to measure next.** (a) Example chosen by a fixed rule (the first
reference-active, DNT-positive chemical in alphabetical order with seven tested concentrations):
(-)-Nicotine after measuring its median concentration
(1.0 µM). Top: twin posterior of the developmental maximum |effect|
at each candidate concentration (bands) and the observed values, hidden from the planner; dotted line: benchmark
response. Bottom: expected information gain about potency per candidate; the planner proposes
0.3 µM. (b) Composite potency error of the NeuroTrajectory
estimator by budget B for the two registered DoseCompass protocols (R6, median first, starts at the median
concentration; R6b, top first, is anchored at the highest), the fixed log-spaced design, random orders (band:
2.5–97.5% of 200 replicates) and the oracle subset (the best achievable within EPA's tested grid). (c) Paired
differences at B = 3, bootstrap 95% CIs over 243 chemicals. Both primary
comparisons with the fixed log-spaced design are ties (R6, median first,
-0.006, CI
-0.060 to
0.053; R6b, top first,
-0.019, CI
-0.073 to
0.033); R6 (median first) beats random
ordering (-0.045, CI
-0.088 to
-0.002). Two protocols were tested on
the same data without multiplicity correction, so a single interval excluding zero is weaker evidence than it
appears. (d) Activity detection at B = 3: R6b (top first) reaches accuracy
0.909
and kappa 0.757,
against 0.881
and 0.677 for the
fixed design (point estimates; random designs show 2.5–97.5% of 200 replicates).

## fig8_chip

**Figure 8 | Chip layer on a real four-compartment MEA (Brewer).** (a) Feed-forward fraction of sorted tunnel axons
per compartment edge, for each recording (NoStim n = 9, HFS5
n = 6, HFS40 n = 6) and pooled
over the unstimulated arrays with Wilson 95% CIs (EC → DG
0.68, DG → CA3
0.29); dashed line: no
directional bias. (b) Conduction times of the unstimulated arrays' axons per edge (quantised by the 25,000 Hz sampling;
bar: median). (c) Lag of the peak cross-correlation between compartment population-burst rates:
72 of 84
edge × recording pairs peak at zero lag (100 ms bins), i.e. synchronous activity, not evidence of directed
propagation. (d, e) Leave-one-recording-out over the 9 unstimulated
arrays; each line joins the errors for one held-out array, bars are macro means. The empirical-Bayes shrinkage model
lowers the per-tunnel feed-forward-fraction error against pooled tunnel means by
0.036
(95% CI -0.056 to
-0.018), a modest gain; against the
direction-permuted null the interval includes zero
(-0.096 to
0.014), and there is no gain on
conduction time (-0.007 ms,
95% CI -0.015 to
0.001).

## fig9_crossassay

**Figure 9 | Cross-assay corroboration (unpaired).** The assays differ in plates, cultures, exposure windows and
read-outs, and share no well with the NFA. (a) Harrill high-content imaging: for the
23 chemicals active in both assays (identity-level matches), the network
BMC from the NFA lies below the neurite or synapse morphology BMC, with a median offset of
-0.56 log10 units (95% CI
-0.98 to
-0.37): network function changes at lower concentrations
than morphology. (b) Kosnik acute MEA call on 99 shared chemicals:
developmental NFA activity and the k = 3 twin both discriminate acute neuroactivity above chance but modestly (NFA
balanced accuracy 0.65, specificity
0.38; permutation p-values above each
estimate); acute exposure of mature networks is a different biological question from developmental exposure.
(c) Concordance statistics with bootstrap 95% CIs over chemicals: activity agreement with Harrill morphology
(Cohen's κ 0.29), potency rank correlation pooled
(0.50) and per imaging family, and the acute potency
rank correlation with Kosnik (0.17,
95% CI -0.08 to
0.42; not significant). The twin's
k = 3 activity call agrees less with Harrill morphology
(κ 0.08) because it calls more chemicals
active (50 of
55).

## fig10_ablations_failures

**Figure 10 | Ablations, failures and a negative result.** (a) One component removed at a time (v2 configuration,
same outer folds and designs, seed 0; paired bootstrap 95% CIs; above zero means the component helps). The
interpolation-informed decoder matters most, and more as k grows (k = 3:
+0.082, 95% CI
0.062 to
0.104); dose-local attention helps at k = 3
(+0.021, CI
0.008 to
0.035) but not detectably at k = 1–2; the
three-seed ensemble helps at k = 1–2. (b) Per-chemical gain over log-linear interpolation at k = 3 on folds 1–4:
the model is worse for 22.7% of chemicals, with a long tail of
large losses; dotted lines mark the median and 10th-percentile chemicals of this distribution (distribution
reference only; not necessarily the same chemicals as Fig. 4, which uses its own percentile rule). (c) Mean
curve-MAE difference by annotated chemical class
at k = 3 (annotation of the EPA reference list as distributed; classes with at least
5 chemicals; marker area grows with n; descriptive, no CI). On
average the model loses for pharmaceuticals and food additives and gains most for organochlorines, pyrethroids and
acetylcholinesterase inhibitors; 60 chemicals have no estimable network
potency and are counted rather than dropped. (d) A selectivity index contrasting network potency with chemical-level
cytotoxicity does not separate DNT-positive from DNT-negative chemicals (AUROC
0.51, 95% CI 0.35
to 0.68); cytotoxic potency alone does better
(0.72). Reported as a negative result.

## figA2_darwin

**Figure A2 | Darwin-Cage honest-ceiling audit: searching for residual structure the twin left on the table.**
(a) Number of programs (feature combinations of the forecast residual) confirmed on unseen chemicals, for each of
the four confirmation folds (fold 4 is the pre-specified primary fold); the original z/√n significance rule alone
against the rule used, which additionally requires a chemical-level sign-flip permutation test, to control the
family-wise false-positive rate. (b) Statistical power of the search itself: with five known effects planted in the
real residuals (plate|date shift and feature × DIV × class interactions), the fraction of the 5 plants recovered
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
194 chemicals: V0, no viability (closed circles), against V1, PubChem
AB/LDH cytotoxicity calls aggregated over all tested concentrations including the ones the forecast hides (open
squares, scenario). (b) The 51 chemicals with an independent NTP
per-well viability read: V0 (no viability), V1 (PubChem AB/LDH, scenario), V2 (NTP chemical-level summary, scenario)
and V3 (NTP viability from only the wells at the measured context concentrations — the one leakage-free arm).
Caveat: PubChem AB/LDH calls (AIDs 2284083, 2284068) are measured at DIV12 on the same NFA plates and aggregated over all concentrations, including the concentrations the forecast hides. Arms V1 and V2 therefore emulate a prior viability screen and are NOT evidence that viability improves the forecast. Arm V3 uses only the wells at the measured context concentrations. V1 and V2 are therefore not evidence that viability
improves the forecast; V3, which shows no reliable gain either
(-0.046 for V1 vs V0 at k = 3, for reference), is the only
arm free of that caveat.

## Reserved (not yet generated)

- `figA1_own_arch` — own prior architectures evaluated on real data (source `results/appendix_own_arch.json`, being computed).
