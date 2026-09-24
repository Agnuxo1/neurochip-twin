"""Deterministic summaries for already-corrected neural calcium traces.

This is a standalone feature branch, not a trained response model. It accepts
per-cell ΔF/F traces and a noise floor measured in cell-free regions using the
same scale. The caller must preserve chip/experiment identifiers and perform
inference at that independent experimental-unit level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _pairwise_synchrony(
    traces: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Return mean Pearson r and fraction of pairs above a fixed threshold."""
    variable = np.std(traces, axis=1) > np.finfo(float).eps
    usable = traces[variable]
    if len(usable) < 2:
        return float("nan"), float("nan")

    correlations = np.corrcoef(usable)
    upper = correlations[np.triu_indices(len(usable), k=1)]
    upper = upper[np.isfinite(upper)]
    if not len(upper):
        return float("nan"), float("nan")
    return float(np.mean(upper)), float(np.mean(upper > threshold))


def _keep_peaks_separated_within_window(
    peaks: np.ndarray,
    peak_heights: np.ndarray,
    min_distance_frames: int,
    n_frames: int,
) -> np.ndarray:
    """Return a mask that applies peak spacing without crossing a window edge."""
    keep = np.zeros(len(peaks), dtype=bool)
    blocked_frames = np.zeros(n_frames, dtype=bool)
    for position in np.argsort(-peak_heights, kind="stable"):
        peak = int(peaks[position])
        if not blocked_frames[peak]:
            keep[position] = True
            left = max(0, peak - min_distance_frames + 1)
            right = min(n_frames, peak + min_distance_frames)
            blocked_frames[left:right] = True
    return keep


def calcium_activity_features(
    delta_f_over_f: np.ndarray,
    times_s: np.ndarray,
    chamber_ids: np.ndarray,
    *,
    perturbation_time_s: float,
    background_noise_dff: float,
    prominence_sigma: float = 2.0,
    min_peak_distance_s: float = 1.0,
    synchrony_threshold: float = 0.6,
) -> pd.DataFrame:
    """Summarize event dynamics separately by chamber and pre/post window.

    Parameters
    ----------
    delta_f_over_f:
        Finite matrix with shape ``(cells, frames)``. Traces must already be
        background-corrected and normalized to ΔF/F by a documented pipeline.
    times_s:
        Strictly increasing, approximately uniform frame timestamps in seconds.
    chamber_ids:
        One compartment label per cell; labels are kept separate in the output.
    perturbation_time_s:
        Split point between baseline frames (``time < split``) and post frames.
    background_noise_dff:
        Positive noise scale, measured from cell-free regions in ΔF/F units.
    prominence_sigma:
        Peak-prominence threshold as a multiplier of the supplied noise scale.
    min_peak_distance_s:
        Minimum temporal distance between detected peaks.
    synchrony_threshold:
        Correlation cutoff used only for the descriptive pair-fraction feature.

    Notes
    -----
    The output has one row per chamber and time window. Cells are nested
    measurements, not independent experimental replicates. A missing peak set
    yields NaN for event prominence; a window with fewer than two varying cell
    traces yields NaN synchrony rather than an artificial zero.

    Event width is measured at 10% of peak prominence (90% relative height)
    and converted from samples to seconds. It is an event-shape descriptor,
    not an inferred action-potential width or a direct reproduction of a
    specific assay's peak-width definition.

    Peak detection, prominence, width, and minimum-distance filtering are
    computed independently on each period's trace segment. As with
    ``scipy.signal.find_peaks``, the first and last sample of each segment are
    not counted as peaks because they lack two-sided local-neighbor context.
    This prevents measurements in one period from changing the other period's
    event features.
    """
    traces = np.asarray(delta_f_over_f, dtype=float)
    times = np.asarray(times_s, dtype=float)
    chambers = np.asarray(chamber_ids)

    if traces.ndim != 2 or min(traces.shape) < 1:
        raise ValueError("delta_f_over_f must be a non-empty cells-by-frames matrix")
    if times.ndim != 1 or len(times) != traces.shape[1] or len(times) < 4:
        raise ValueError("times_s must contain one timestamp per frame and at least 4 frames")
    if chambers.ndim != 1 or len(chambers) != traces.shape[0]:
        raise ValueError("chamber_ids must contain one label per cell")
    if not np.isfinite(traces).all() or not np.isfinite(times).all():
        raise ValueError("traces and timestamps must be finite; resolve missing frames upstream")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be strictly increasing")
    intervals = np.diff(times)
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=0.05, atol=1e-9):
        raise ValueError("times_s must be approximately uniformly sampled for peak-distance filtering")
    if not np.isfinite(perturbation_time_s) or not times[0] < perturbation_time_s < times[-1]:
        raise ValueError("perturbation_time_s must fall inside the recording")
    if not np.isfinite(background_noise_dff) or background_noise_dff <= 0:
        raise ValueError("background_noise_dff must be finite and positive")
    if not np.isfinite(prominence_sigma) or prominence_sigma <= 0:
        raise ValueError("prominence_sigma must be finite and positive")
    if not np.isfinite(min_peak_distance_s) or min_peak_distance_s <= 0:
        raise ValueError("min_peak_distance_s must be finite and positive")
    if not np.isfinite(synchrony_threshold) or not -1 <= synchrony_threshold <= 1:
        raise ValueError("synchrony_threshold must be between -1 and 1")

    masks = {
        "baseline": times < perturbation_time_s,
        "post": times >= perturbation_time_s,
    }
    if any(np.count_nonzero(mask) < 2 for mask in masks.values()):
        raise ValueError("baseline and post windows must each contain at least two frames")

    min_distance_frames = max(1, int(np.ceil(min_peak_distance_s / dt)))
    prominence = prominence_sigma * background_noise_dff
    rows: list[dict[str, object]] = []

    for chamber in pd.unique(chambers):
        cell_mask = chambers == chamber
        chamber_traces = traces[cell_mask]
        for period, time_mask in masks.items():
            window = chamber_traces[:, time_mask]
            window_times = times[time_mask]
            duration_min = (window_times[-1] - window_times[0] + dt) / 60.0
            event_rates: list[float] = []
            event_prominences: list[float] = []
            event_widths_s: list[float] = []
            for trace in chamber_traces:
                window_trace = trace[time_mask]
                peaks, properties = find_peaks(
                    window_trace,
                    prominence=prominence,
                    width=(None, None),
                    rel_height=0.9,
                )
                selected = np.flatnonzero(
                    _keep_peaks_separated_within_window(
                        peaks,
                        window_trace[peaks],
                        min_distance_frames,
                        len(window_trace),
                    )
                )
                event_rates.append(len(selected) / duration_min)
                event_prominences.extend(properties["prominences"][selected].tolist())
                event_widths_s.extend((properties["widths"][selected] * dt).tolist())

            mean_r, connected_fraction = _pairwise_synchrony(
                window, synchrony_threshold
            )
            rows.append(
                {
                    "chamber_id": chamber,
                    "period": period,
                    "n_cells": int(len(window)),
                    "event_rate_per_min_mean": float(np.mean(event_rates)),
                    "event_rate_per_min_median": float(np.median(event_rates)),
                    "active_cell_fraction": float(np.mean(np.asarray(event_rates) > 0)),
                    "mean_event_prominence_dff": (
                        float(np.mean(event_prominences))
                        if event_prominences
                        else float("nan")
                    ),
                    "mean_event_width_at_10pct_prominence_s": (
                        float(np.mean(event_widths_s)) if event_widths_s else float("nan")
                    ),
                    "median_event_width_at_10pct_prominence_s": (
                        float(np.median(event_widths_s)) if event_widths_s else float("nan")
                    ),
                    "mean_pairwise_correlation": mean_r,
                    "fraction_pairs_above_sync_threshold": connected_fraction,
                    "prominence_threshold_dff": float(prominence),
                }
            )

    return pd.DataFrame(rows)
