# MEA features from spike lists

`neurotwin.features.mea.extract_features(electrode_spikes, duration_s)` returns
the 17 names used by the EPA NFA trajectory model. The implementation was
written from published method descriptions and uses no vendored R code.
**These are operational analogs, not a validated reproduction of the EPA
well-level values.** NFA modeling retains EPA's published features.

## Primary method sources

- [Cotterill et al. 2016](https://journals.sagepub.com/doi/10.1177/1087057116640520), Methods and Table 1: MaxInterval burst detection; 3 ms, five-electrode network-spike bins; STTC correlation with ±50 ms window; active and bursting electrode summaries.
- [Ball et al. 2017](https://doi.org/10.1016/j.neunet.2017.07.009), Sections 2 and 3: normalized **multivariate** total correlation, divided by `N−1`, with 3 ms binary spike bins; electrodes with under 0.01% occupied bins excluded. The [author-provided full text](https://www.researchgate.net/publication/318674687_A_multivariate_extension_of_mutual_information_for_growing_neural_networks) states the 3 ms choice and inclusion rule.
- [Frank et al. 2017](https://doi.org/10.1093/toxsci/kfx169), Data analysis: 16 established measures from `meadq`/`sjemea`, with normalized multiinformation added as the 17th.
- [EPA NFA refinement data](https://github.com/USEPA/CompTox-DNT-NFA-Refinement): source well-level values used by our trajectory model.

Cotterill's published minimum is **six** spikes per electrode burst, not five.
Its 50 ms parameter belongs to the STTC correlation window, while network
spikes and Ball's normalized multiinformation use **3 ms bins**. These primary
sources supersede the preliminary task shorthand. The NFA columns do not
publish every endpoint's exact aggregation convention, so the mappings below
state our chosen convention explicitly.

## Detection and aggregation

Inputs are sorted spike times in seconds for each electrode and the analyzed
recording duration in seconds. A spike at the exact duration is invalid.
An electrode is active at ≥5 spikes/minute. A MaxInterval burst starts with
an ISI ≤0.1 s, extends over ISIs ≤0.25 s, and contains ≥6 spikes over ≥0.05 s.
Candidates separated by <0.8 s are merged before final filtering. A bursting
electrode has ≥1 accepted burst/minute. Per-electrode values are aggregated
using the median over eligible active/bursting electrodes, following Cotterill's
well summary pattern. Durations and intervals are seconds; firing rates are Hz;
burst rates are per minute; percentages are 0–100.

Network events are maximal runs of adjacent 3 ms bins where ≥5 distinct
electrodes spike in each bin. Per-event peak is the greatest concurrent
electrode count. Event spike count includes every spike in qualifying bins.
The network event summaries use arithmetic means, as an operational reading of
EPA's `ns.*.m` columns. Event intervals are from previous event end to next
event start. A measured absence is zero for counts; an undefined duration,
interval, fraction, or correlation is JSON `null`.

| NFA feature name | Operational definition |
|---|---|
| `firing_rate_mean` | Median spikes/s over active electrodes |
| `burst_rate` | Median accepted bursts/min over electrodes with bursts |
| `per_burst_interspike_interval` | Median electrode mean within-burst ISI, s |
| `per_burst_spike_percent` | Median electrode fraction of spikes inside bursts ×100 |
| `burst_duration_mean` | Median electrode mean burst duration, s |
| `interburst_interval_mean` | Median electrode mean end-to-start IBI, s |
| `active_electrodes_number` | Count of electrodes with ≥5 spikes/min |
| `bursting_electrodes_number` | Count of active electrodes with ≥1 burst/min |
| `network_spike_number` | Count of contiguous qualifying network events |
| `network_spike_peak` | Mean event peak concurrent-electrode count |
| `spike_duration_mean` | Mean network-event duration, s; source column `ns.durn.m` |
| `per_network_spike_spike_percent` | Percent of all spikes in network events |
| `inter_network_spike_interval_mean` | Mean event end-to-next-start interval, s; source column `ns.mean.insis`, whose name is ambiguous |
| `network_spike_duration_std` | Sample SD of network-event durations, s |
| `per_network_spike_spike_number_mean` | Mean spike count per network event |
| `correlation_coefficient_mean` | Mean pairwise STTC over active electrodes, ±50 ms; no pair gives null |
| `mutual_information_norm` | Ball normalized multiinformation rate, bits/s: `(Σ H(X_i) − H(X_1,…,X_N)) / (N−1) / 0.003` |

For the information feature, `X_i` is 1 when channel `i` has ≥1 spike in a
3 ms bin; otherwise 0. Marginal and joint entropies use empirical bin
frequencies and base-2 logarithms. Fewer than two retained electrodes give
null. Ball cautions that joint-entropy estimation becomes unreliable for
networks much larger than 16 electrodes at fixed recording length; Brewer
subregions contain 19 electrodes, so their NMI is exploratory.

## Cross-platform scope and R1

`python repo/scripts/run_brewer_features.py` computes the named analogs for
each of Brewer's 21 recordings × 4 subregions from the 300 s well spike lists.
These are not cortical 48-well NFA cultures and must not be pooled with NFA
readouts as if they were paired observations.

`python repo/scripts/audit_r1_sources.py` compares the public
[Brown/Cotterill spike archive](https://github.com/sje30/EPAmeadev) metadata
with the local NFA plate/date/DIV keys. That archive's README explicitly permits
use of its data/resources with attribution. Its HDF5 recordings have no
overlapping plate with the local NFA endpoint table, so a three-plate matched
Pearson/ICC/relative-error fidelity test is not estimable. The script does not
download 307 MB of unmatched spikes. The audit JSON records the remote Git
tree SHA and metadata-body SHA256. Neither is a checksum for undownloaded
HDF5 payloads. If a matched raw-spike archive later appears, R1 must be run
per physical well and only then may these analogs be described as EPA-faithful.
