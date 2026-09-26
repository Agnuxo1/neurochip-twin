"""Independent spike-list to MEA feature extractor.

Operational definitions follow Cotterill et al. (2016) and Ball et al. (2017)
where published. The EPA 17-endpoint pipeline has not been validated against
matched raw spike lists; these outputs must not replace the published NFA values.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

FEATURES = (
    "firing_rate_mean", "burst_rate", "per_burst_interspike_interval",
    "per_burst_spike_percent", "burst_duration_mean", "interburst_interval_mean",
    "active_electrodes_number", "bursting_electrodes_number", "network_spike_number",
    "network_spike_peak", "spike_duration_mean", "per_network_spike_spike_percent",
    "inter_network_spike_interval_mean", "network_spike_duration_std",
    "per_network_spike_spike_number_mean", "correlation_coefficient_mean",
    "mutual_information_norm",
)

BURST_START_ISI_S = .1
BURST_MAX_ISI_S = .25
BURST_MIN_INTERVAL_S = .8
BURST_MIN_DURATION_S = .05
BURST_MIN_SPIKES = 6
NETWORK_BIN_S = .003
NETWORK_MIN_ELECTRODES = 5
CORRELATION_WINDOW_S = .05
NMI_MIN_OCCUPANCY = .0001


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _trains(spikes_by_electrode: Mapping[str, Sequence[float]], duration_s: float) -> dict[str, np.ndarray]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not spikes_by_electrode:
        raise ValueError("At least one electrode is required")
    result = {}
    for electrode, times in spikes_by_electrode.items():
        train = np.asarray(times, dtype=float)
        if train.ndim != 1 or np.any(~np.isfinite(train)) or np.any(train < 0) or np.any(train >= duration_s):
            raise ValueError(f"Invalid spike times for {electrode}")
        if len(train) > 1 and np.any(np.diff(train) < 0):
            raise ValueError(f"Unsorted spike times for {electrode}")
        result[str(electrode)] = train
    return result


def detect_bursts(times: Sequence[float], *, min_spikes: int = BURST_MIN_SPIKES) -> list[tuple[int, int]]:
    """MaxInterval bursts as inclusive index pairs, with close candidates merged."""
    t = np.asarray(times, dtype=float)
    if len(t) < min_spikes:
        return []
    isi = np.diff(t)
    candidates: list[tuple[int, int]] = []
    i = 0
    while i < len(isi):
        if isi[i] > BURST_START_ISI_S:
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(isi) and isi[j] <= BURST_MAX_ISI_S:
            j += 1
        candidates.append((start, j))
        i = j
    merged: list[tuple[int, int]] = []
    for start, end in candidates:
        if merged and t[start] - t[merged[-1][1]] < BURST_MIN_INTERVAL_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged
            if end - start + 1 >= min_spikes and t[end] - t[start] >= BURST_MIN_DURATION_S]


def detect_network_spikes(trains: Mapping[str, np.ndarray], duration_s: float,
                          *, bin_s: float = NETWORK_BIN_S,
                          min_electrodes: int = NETWORK_MIN_ELECTRODES) -> list[dict]:
    """Contiguous 3 ms bins with at least five firing electrodes."""
    n_bins = int(np.ceil(duration_s / bin_s))
    participation = np.zeros(n_bins, dtype=np.uint16)
    all_indices = []
    for train in trains.values():
        indices = np.minimum((train / bin_s).astype(np.int64), n_bins - 1)
        participation[np.unique(indices)] += 1
        all_indices.append(indices)
    active = participation >= min_electrodes
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    all_spikes = np.concatenate(all_indices) if all_indices else np.empty(0, dtype=int)
    return [{"start_s": float(start * bin_s),
             "end_s": float(min(end * bin_s, duration_s)),
             "duration_s": float(min(end * bin_s, duration_s) - start * bin_s),
             "peak_electrodes": int(participation[start:end].max()),
             "n_spikes": int(np.count_nonzero((all_spikes >= start) & (all_spikes < end)))}
            for start, end in zip(starts, ends)]


def _covered_fraction(t: np.ndarray, duration_s: float, window_s: float) -> float:
    if not len(t):
        return 0.0
    starts = np.maximum(0, t - window_s)
    ends = np.minimum(duration_s, t + window_s)
    total = 0.0
    edge = 0.0
    for start, end in zip(starts, ends):
        if start > edge:
            total += end - start
            edge = end
        elif end > edge:
            total += end - edge
            edge = end
    return total / duration_s


def spike_time_tiling_coefficient(a: Sequence[float], b: Sequence[float], duration_s: float,
                                  window_s: float = CORRELATION_WINDOW_S) -> float | None:
    """Cutts/Eglen STTC, with Cotterill's ±50 ms coincidence window."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(x) or not len(y):
        return None
    pa = np.mean(np.searchsorted(y, x + window_s, side="right") >
                 np.searchsorted(y, x - window_s, side="left"))
    pb = np.mean(np.searchsorted(x, y + window_s, side="right") >
                 np.searchsorted(x, y - window_s, side="left"))
    ta, tb = _covered_fraction(x, duration_s, window_s), _covered_fraction(y, duration_s, window_s)
    first = (pa - tb) / (1 - pa * tb) if 1 - pa * tb > 1e-12 else 1.0
    second = (pb - ta) / (1 - pb * ta) if 1 - pb * ta > 1e-12 else 1.0
    return float(np.clip((first + second) / 2, -1, 1))


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def normalized_multiinformation(trains: Mapping[str, np.ndarray], duration_s: float,
                                bin_s: float = NETWORK_BIN_S) -> float | None:
    """Ball's multivariate total correlation/(n-1), as a bits/s rate.

    Electrode inclusion uses ≥0.01% occupied 3 ms bins. More than 31 included
    electrodes is rejected rather than overflowing the uint32 state word.
    """
    n_bins = int(np.ceil(duration_s / bin_s))
    occupied = []
    for train in trains.values():
        bins = np.unique(np.minimum((np.asarray(train, dtype=float) / bin_s).astype(np.int64), n_bins - 1))
        if len(bins) / n_bins >= NMI_MIN_OCCUPANCY:
            occupied.append(bins)
    if len(occupied) < 2:
        return None
    if len(occupied) > 31:
        return None
    states = np.zeros(n_bins, dtype=np.uint32)
    marginal = 0.0
    for index, bins in enumerate(occupied):
        states[bins] |= np.uint32(1 << index)
        p = len(bins) / n_bins
        marginal += _entropy(np.array([p, 1 - p]))
    _, counts = np.unique(states, return_counts=True)
    total_correlation = max(0.0, marginal - _entropy(counts))
    return float(total_correlation / (len(occupied) - 1) / bin_s)


def extract_features(spikes_by_electrode: Mapping[str, Sequence[float]], duration_s: float) -> dict[str, float | None]:
    """Return exactly the 17 NFA-named endpoints for one recorded electrode group.

    Undefined means ``None``; a measured absence (e.g. zero network spikes)
    is zero. This is a documented analog, not an EPA-fidelity claim.
    """
    trains = _trains(spikes_by_electrode, duration_s)
    active = {e: t for e, t in trains.items() if len(t) * 60 / duration_s >= 5}
    firing = [len(t) / duration_s for t in active.values()]
    burst_rates, isi_means, burst_pcts, durations, ibi_means = [], [], [], [], []
    bursting_count = 0
    for t in active.values():
        events = detect_bursts(t)
        rate = len(events) * 60 / duration_s
        if rate >= 1:
            bursting_count += 1
        if not events:
            continue
        burst_rates.append(rate)
        isi_means.append(float(np.mean(np.concatenate([np.diff(t[s:e+1]) for s, e in events]))))
        burst_pcts.append(100 * sum(e-s+1 for s, e in events) / len(t))
        durations.append(float(np.mean([t[e] - t[s] for s, e in events])))
        if len(events) >= 2:
            ibi_means.append(float(np.mean([t[events[i][0]] - t[events[i-1][1]]
                                             for i in range(1, len(events))])))
    events = detect_network_spikes(trains, duration_s)
    total_spikes = sum(map(len, trains.values()))
    intervals = [events[i]["start_s"] - events[i-1]["end_s"] for i in range(1, len(events))]
    correlations = [value for a, b in combinations(active.values(), 2)
                    if (value := spike_time_tiling_coefficient(a, b, duration_s)) is not None]
    values = (
        _median(firing), _median(burst_rates), _median(isi_means), _median(burst_pcts),
        _median(durations), _median(ibi_means), float(len(active)), float(bursting_count),
        float(len(events)), _mean([event["peak_electrodes"] for event in events]),
        _mean([event["duration_s"] for event in events]),
        100 * sum(event["n_spikes"] for event in events) / total_spikes if total_spikes else None,
        _mean(intervals), float(np.std([e["duration_s"] for e in events], ddof=1)) if len(events) >= 2 else None,
        _mean([event["n_spikes"] for event in events]), _mean(correlations),
        normalized_multiinformation(trains, duration_s),
    )
    return dict(zip(FEATURES, values))
