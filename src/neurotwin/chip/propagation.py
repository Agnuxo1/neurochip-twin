"""Brewer four-compartment graph, descriptive activity, and recording-level CV.

FF means EC→DG→CA3→CA1→EC in the source tables. FID is condition-local.
NoStim is the only fitting/evaluation cohort; HFS is descriptive only.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REGIONS = ("EC", "DG", "CA3", "CA1")
PAIRS = ("EC-DG", "DG-CA3", "CA3-CA1", "CA1-EC")
DURATION_S = 300.0
TUNNELS = tuple([f"{c}6-{c}7" for c in "ABCDE"] +
                [f"F{i}-G{i}" for i in range(1, 6)] +
                [f"{c}6-{c}7" for c in "HJKLM"] +
                [f"F{i}-G{i}" for i in range(8, 13)])
DELAY_VALUES_MS = (0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88)


def pair_for_tunnel(orientation: str, tunnel: str) -> str:
    """Map physical tunnel to biological edge, including tunnels with zero axons."""
    if orientation not in {"clockwise", "counterclockwise"} or tunnel not in TUNNELS:
        raise ValueError((orientation, tunnel))
    vertical_left = tunnel[0] in "ABCDE"
    vertical_right = tunnel[0] in "HJKLM"
    bottom = tunnel.startswith("F") and int(tunnel.split("-")[0][1:]) <= 5
    if orientation == "clockwise":
        return ("EC-DG" if vertical_left else "CA3-CA1" if vertical_right
                else "CA1-EC" if bottom else "DG-CA3")
    return ("CA1-EC" if vertical_left else "DG-CA3" if vertical_right
            else "EC-DG" if bottom else "CA3-CA1")


def read_data(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    import pyarrow.parquet as pq

    with (data_dir / "recordings.csv").open(newline="", encoding="utf-8") as handle:
        recordings = list(csv.DictReader(handle))
    for record in recordings:
        record["fid"] = int(record["fid"])
    wells = pq.read_table(data_dir / "wells.parquet").to_pylist()
    axons = pq.read_table(data_dir / "axons.parquet").to_pylist()
    return recordings, wells, axons


def key(row: dict) -> tuple[str, int]:
    return row["condition"], int(row["fid"])


def _delay_distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p10_ms": None,
                "p90_ms": None, "values_ms": []}
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)), "p10_ms": float(np.quantile(a, .1)),
            "p90_ms": float(np.quantile(a, .9)),
            "values_ms": [float(x) for x in sorted(values)]}


def _burst_features(times: np.ndarray, bin_s: float = .1) -> tuple[dict, np.ndarray]:
    counts = np.histogram(times, bins=int(DURATION_S / bin_s), range=(0, DURATION_S))[0]
    # High-activity bins; a 300 ms refractory window avoids counting one burst repeatedly.
    threshold = max(3, int(np.ceil(np.quantile(counts, .95))))
    candidates = np.flatnonzero((counts >= threshold) &
                                (counts >= np.r_[0, counts[:-1]]) &
                                (counts >= np.r_[counts[1:], 0]))
    peaks = []
    for i in candidates:
        if not peaks or i - peaks[-1] >= 3:
            peaks.append(int(i))
        elif counts[i] > counts[peaks[-1]]:
            peaks[-1] = int(i)
    return {"n_bursts": len(peaks), "burst_rate_per_min": len(peaks) / 5,
            "threshold_spikes_per_100ms": threshold,
            "burst_peak_times_s": [float((i + .5) * bin_s) for i in peaks]}, counts


def _lag_summary(source: np.ndarray, target: np.ndarray, seed: int,
                 max_lag_bins: int = 20, n_null: int = 99) -> dict:
    """Population cross-correlation, max-lag corrected by circular-shift null."""
    x = np.log1p(source.astype(float))
    y = np.log1p(target.astype(float))
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return {"lag_ms": None, "correlation": None, "p_shift": None,
                "n_null": n_null, "reason": "silent population"}
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)

    def curve(z: np.ndarray) -> np.ndarray:
        return np.asarray([np.dot(x[:len(x)-lag], z[lag:]) if lag >= 0
                           else np.dot(x[-lag:], z[:len(z)+lag]) for lag in lags]) / denom

    observed = curve(y)
    peak = int(np.argmax(observed))
    rng = np.random.default_rng(seed)
    # Shifts exclude the tested ±2 s window and its periodic counterpart.
    null = [float(np.max(curve(np.roll(y, int(shift))))) for shift in
            rng.integers(max_lag_bins + 1, len(y) - max_lag_bins, size=n_null)]
    return {"lag_ms": int(lags[peak] * 100), "correlation": float(observed[peak]),
            "p_shift": (1 + sum(v >= observed[peak] for v in null)) / (n_null + 1),
            "n_null": n_null, "lag_window_ms": [-2000, 2000],
            "interpretation": "population association; not axonal conduction or causality"}


def summarize(recordings: list[dict], wells: list[dict], axons: list[dict]) -> list[dict]:
    by_well = defaultdict(list)
    by_axon = defaultdict(list)
    for row in wells:
        by_well[key(row)].append(row)
    for row in axons:
        by_axon[(key(row), row["tunnel"])].append(row)
    out = []
    for record in recordings:
        rec_key = key(record)
        if len(by_well[rec_key]) != 76:
            raise ValueError(f"Expected 76 wells: {rec_key}")
        region_summary, histograms = {}, {}
        well_summary = []
        for well in by_well[rec_key]:
            bursts, _ = _burst_features(np.asarray(well["spike_times_s"], dtype=float))
            well_summary.append({"electrode": well["electrode"], "subregion": well["subregion"],
                                 "n_spikes": well["n_spikes"],
                                 "spike_rate_hz": well["n_spikes"] / DURATION_S,
                                 **bursts})
        for region in REGIONS:
            subset = [w for w in by_well[rec_key] if w["subregion"] == region]
            times = np.concatenate([np.asarray(w["spike_times_s"], dtype=float) for w in subset])
            bursts, counts = _burst_features(times)
            region_summary[region] = {"n_wells": len(subset), "n_spikes": len(times),
                                      "spike_rate_hz": len(times) / DURATION_S,
                                      **bursts}
            histograms[region] = counts
        tunnels = []
        for tunnel in TUNNELS:
            subset = by_axon[(rec_key, tunnel)]
            pair = pair_for_tunnel(record["orientation"], tunnel)
            if any(row["subregion_pair"] != pair for row in subset):
                raise ValueError(f"Tunnel topology mismatch: {rec_key}, {tunnel}")
            ff = [row for row in subset if row["direction"] == "ff"]
            fb = [row for row in subset if row["direction"] == "fb"]
            tunnels.append({"tunnel": tunnel, "subregion_pair": pair,
                            "n_axons_ff": len(ff), "n_axons_fb": len(fb),
                            "ff_fraction": len(ff) / len(subset) if subset else None,
                            "ff_spike_rate_hz": sum(len(r["upstream_spike_times_s"]) for r in ff) / DURATION_S,
                            "fb_spike_rate_hz": sum(len(r["upstream_spike_times_s"]) for r in fb) / DURATION_S,
                            "ff_conduction": _delay_distribution([r["conduction_ms"] for r in ff]),
                            "fb_conduction": _delay_distribution([r["conduction_ms"] for r in fb])})
        edges = []
        for i, pair in enumerate(PAIRS):
            source, target = pair.split("-")
            selected = [t for t in tunnels if t["subregion_pair"] == pair]
            n_ff = sum(t["n_axons_ff"] for t in selected)
            n_fb = sum(t["n_axons_fb"] for t in selected)
            edges.append({"pair": pair, "source": source, "target": target,
                          "n_axons_ff": n_ff, "n_axons_fb": n_fb,
                          "ff_fraction": n_ff / (n_ff + n_fb) if n_ff + n_fb else None,
                          "directionality_d": (n_ff - n_fb) / (n_ff + n_fb) if n_ff + n_fb else None,
                          "burst_propagation": _lag_summary(histograms[source], histograms[target],
                                                            seed=1000 * (i + 1) + 100 *
                                                            (0 if record["condition"] == "NoStim" else
                                                             1 if record["condition"] == "HFS5" else 2) + record["fid"])})
        out.append({"condition": record["condition"], "fid": record["fid"],
                    "orientation": record["orientation"], "wells": well_summary,
                    "regions": region_summary,
                    "tunnels": tunnels, "edges": edges})
    return out


def _fit(rows: list[dict], strength: float = 8.0) -> dict:
    """Training-only empirical Bayes prior by anatomical pair and tunnel."""
    global_counts = Counter(r["direction"] for r in rows)
    global_p = (global_counts["ff"] + 1) / (len(rows) + 2)
    pair_counts = defaultdict(Counter)
    tunnel_counts = defaultdict(Counter)
    delays = defaultdict(list)
    for r in rows:
        pair_counts[r["subregion_pair"]][r["direction"]] += 1
        tunnel_counts[(r["tunnel"], r["subregion_pair"])][r["direction"]] += 1
        delays[(r["tunnel"], r["subregion_pair"])].append(r["conduction_ms"])
    pair_p = {pair: (counter["ff"] + strength * global_p) / (sum(counter.values()) + strength)
              for pair, counter in pair_counts.items()}
    pair_delays = {pair: [r["conduction_ms"] for r in rows if r["subregion_pair"] == pair]
                   for pair in PAIRS}
    return {"global_p": global_p, "pair_p": pair_p, "counts": tunnel_counts,
            "delays": delays, "pair_delays": pair_delays,
            "global_delays": [r["conduction_ms"] for r in rows], "strength": strength}


def _predict(fit: dict, tunnel: str, pair: str, baseline: bool = False) -> tuple[float, float, list[float]]:
    if baseline:
        # Physical tunnel pooled across orientations; train-only global fallback.
        counts = sum((v for (t, _), v in fit["counts"].items() if t == tunnel), Counter())
        values = [v for (t, _), x in fit["delays"].items() if t == tunnel for v in x]
        p = counts["ff"] / sum(counts.values()) if sum(counts.values()) else fit["global_p"]
        values = values or fit["global_delays"]
    else:
        counts = fit["counts"][(tunnel, pair)]
        prior = fit["pair_p"].get(pair, fit["global_p"])
        p = (counts["ff"] + fit["strength"] * prior) / (sum(counts.values()) + fit["strength"])
        local = fit["delays"][(tunnel, pair)]
        pair_values = fit["pair_delays"][pair] or fit["global_delays"]
        local_counts = Counter(round(v, 2) for v in local)
        prior_counts = Counter(round(v, 2) for v in pair_values)
        n = len(local)
        pmf = [(local_counts[v] + fit["strength"] * prior_counts[v] / len(pair_values)) /
               (n + fit["strength"]) for v in DELAY_VALUES_MS]
        return float(p), float(sum(v * q for v, q in zip(DELAY_VALUES_MS, pmf))), pmf
    counts_delay = Counter(round(v, 2) for v in values)
    # Laplace floor allows finite held-out log likelihood.
    pmf = [(counts_delay[v] + .1) / (len(values) + .1 * len(DELAY_VALUES_MS))
           for v in DELAY_VALUES_MS]
    return float(p), float(np.mean(values)), pmf


def _recording_scores(truth: list[dict], fit: dict, fit_null: dict) -> dict:
    by_tunnel = defaultdict(list)
    for row in truth:
        by_tunnel[row["tunnel"]].append(row)
    errors = defaultdict(list)
    predictions = []
    for tunnel, rows in sorted(by_tunnel.items()):
        pair = rows[0]["subregion_pair"]
        assert all(r["subregion_pair"] == pair for r in rows)
        actual_p = sum(r["direction"] == "ff" for r in rows) / len(rows)
        actual_delays = [r["conduction_ms"] for r in rows]
        entry = {"tunnel": tunnel, "subregion_pair": pair, "n_axons": len(rows),
                 "observed_ff_fraction": actual_p,
                 "observed_mean_conduction_ms": float(np.mean(actual_delays))}
        for name, model in (("model", fit), ("tunnel_mean", fit), ("permuted_direction", fit_null)):
            p, delay, pmf = _predict(model, tunnel, pair, baseline=name == "tunnel_mean")
            entry[name] = {"ff_fraction": p, "conduction_mean_ms": delay}
            errors[(name, "ff")].append(abs(p - actual_p))
            errors[(name, "delay")].extend(abs(delay - x) for x in actual_delays)
            if name != "permuted_direction":
                for observed in actual_delays:
                    index = min(range(len(DELAY_VALUES_MS)), key=lambda i: abs(DELAY_VALUES_MS[i] - observed))
                    errors[(name, "nll")].append(-np.log(max(pmf[index], 1e-12)))
        predictions.append(entry)
    scores = {name: {"ff_fraction_mae": float(np.mean(errors[(name, "ff")])),
                     "conduction_mae_ms": float(np.mean(errors[(name, "delay")])),
                     **({"conduction_nll_nats": float(np.mean(errors[(name, "nll")]))}
                        if name != "permuted_direction" else {})}
              for name in ("model", "tunnel_mean", "permuted_direction")}
    return {"n_observed_tunnels": len(by_tunnel), "n_axons": len(truth),
            "scores": scores, "predictions": predictions}


def _bootstrap(folds: list[dict], metric: str, baseline: str, n: int = 10000) -> dict:
    diff = np.asarray([f["scores"]["model"][metric] - f["scores"][baseline][metric]
                       for f in folds])
    rng = np.random.default_rng(1200 + len(metric) + len(baseline))
    boot = np.mean(diff[rng.integers(0, len(diff), size=(n, len(diff)))], axis=1)
    return {"difference_model_minus_baseline": float(diff.mean()),
            "ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
            "p_bootstrap_no_improvement": float(np.mean(boot >= 0)),
            "unit": "recording", "n_resamples": n}


def cross_validate(axons: list[dict], seed: int = 17) -> dict:
    control = [r for r in axons if r["condition"] == "NoStim"]
    fids = sorted({r["fid"] for r in control})
    if len(fids) != 9:
        raise ValueError(f"Expected nine NoStim recordings; got {fids}")
    folds = []
    for fid in fids:
        train = [r for r in control if r["fid"] != fid]
        test = [r for r in control if r["fid"] == fid]
        if not test:
            raise ValueError(f"No axons for held-out FID {fid}")
        rng = np.random.default_rng(seed + fid)
        permuted = rng.permutation([r["direction"] for r in train])
        null_train = [{**r, "direction": str(direction)} for r, direction in zip(train, permuted)]
        fold = _recording_scores(test, _fit(train), _fit(null_train))
        fold.update({"condition": "NoStim", "fid": fid, "n_train_axons": len(train),
                     "train_recordings": [x for x in fids if x != fid]})
        folds.append(fold)
    metrics = ("ff_fraction_mae", "conduction_mae_ms", "conduction_nll_nats")
    summary = {name: {metric: float(np.mean([f["scores"][name][metric] for f in folds]))
                      for metric in metrics if metric in folds[0]["scores"][name]}
               for name in ("model", "tunnel_mean", "permuted_direction")}
    comparisons = {baseline: {metric: _bootstrap(folds, metric, baseline)
                              for metric in (metrics if baseline == "tunnel_mean" else metrics[:2])}
                   for baseline in ("tunnel_mean", "permuted_direction")}
    return {"protocol": "leave-one-recording-out over nine NoStim recordings",
            "seed": seed, "model": "pair/tunnel empirical Bayes shrinkage, strength=8 axons",
            "baseline_tunnel": "pooled physical tunnel mean over eight training recordings",
            "baseline_direction_null": "permutation of training axon direction labels across the eight recordings; fixed per fold",
            "direction_null_delay_note": "direction permutation leaves conduction times unchanged, so its delay MAE equals the model by construction",
            "aggregation": "macro mean of nine recording MAEs; FF MAE per observed tunnel, delay MAE per held-out axon",
            "summary": summary, "comparisons": comparisons, "folds": folds}


def export_parameters(axons: list[dict], graphs: list[dict]) -> dict:
    control = [r for r in axons if r["condition"] == "NoStim"]
    fit = _fit(control)
    pair_params = []
    for pair in PAIRS:
        selected = [r for r in control if r["subregion_pair"] == pair]
        n_ff = sum(r["direction"] == "ff" for r in selected)
        n_fb = len(selected) - n_ff
        delays = _delay_distribution([r["conduction_ms"] for r in selected])
        pair_params.append({"pair": pair, "n_axons_ff": n_ff, "n_axons_fb": n_fb,
                            "observed_directionality_d": (n_ff - n_fb) / len(selected) if selected else None,
                            "ff_weight": fit["pair_p"].get(pair, fit["global_p"]),
                            "fb_weight": 1 - fit["pair_p"].get(pair, fit["global_p"]),
                            "conduction": delays})
    tunnel_params = [{"orientation": orientation, "tunnel": tunnel,
                      "pair": pair_for_tunnel(orientation, tunnel),
                      "predicted_ff_fraction": _predict(fit, tunnel, pair_for_tunnel(orientation, tunnel))[0],
                      "predicted_mean_conduction_ms": _predict(fit, tunnel, pair_for_tunnel(orientation, tunnel))[1]}
                     for orientation in ("clockwise", "counterclockwise") for tunnel in TUNNELS]
    return {"source_cohort": "NoStim only, nine recordings; HFS excluded",
            "units": {"conduction": "ms", "weight": "fraction of sorted axons"},
            "scope": "descriptive Brewer parameters; no causal or cross-platform calibration",
            "fit_pair_edges": pair_params, "fit_tunnels": tunnel_params,
            "observed_per_recording": [{"condition": g["condition"], "fid": g["fid"],
                                        "edges": g["edges"]} for g in graphs]}


def write_outputs(data_dir: Path, result_dir: Path) -> dict:
    recordings, wells, axons = read_data(data_dir)
    graphs = summarize(recordings, wells, axons)
    cv = cross_validate(axons)
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {"title": "Propagation fitted and evaluated on Brewer",
              "source": "https://zenodo.org/records/10257483", "license": "CC0-1.0",
              "fid_semantics": "condition-local ordinal; HFS recordings are not paired to NoStim by fid",
              "n_recordings": len(graphs), "n_axons": len(axons),
              "conditions": dict(Counter(g["condition"] for g in graphs)),
              "graph_description": "four directed pairs; FF follows EC-DG-CA3-CA1-EC; feedback is reverse",
              "burst_method": "100 ms population bins, 95th-percentile peak threshold, 300 ms separation; ±2 s lag scan with 99 circular shifts",
              "burst_interpretation": "zero-lag correlation is synchronous activity, not evidence of directed propagation; lag associations are exploratory",
              "graphs": graphs, "cv": cv,
              "integrity_prior_check": {"source": "colab/board.json task #4 review note by Claude",
                                        "sign_agreement": "437/437 measurable axons",
                                        "corr_absolute_lag_conduction": 0.984,
                                        "status": "previously verified; not recalculated here"}}
    (result_dir / "r8_chiplayer.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Concise companion result for consumers that need only R8 metrics.
    compact = {k: result[k] for k in ("title", "source", "n_recordings", "n_axons", "conditions", "fid_semantics")}
    compact["cv"] = {k: cv[k] for k in ("protocol", "seed", "model", "aggregation", "summary", "comparisons")}
    (result_dir / "chiplayer_r8.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    (result_dir / "chiplayer_params.json").write_text(json.dumps(export_parameters(axons, graphs), indent=2) + "\n", encoding="utf-8")
    return compact
