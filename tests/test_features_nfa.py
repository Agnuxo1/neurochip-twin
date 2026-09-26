"""Known spike trains and EPA-source limits for the independent feature engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))

from neurotwin.features.mea import (  # noqa: E402
    FEATURES, detect_bursts, detect_network_spikes, extract_features,
    normalized_multiinformation, spike_time_tiling_coefficient,
)


def test_max_interval_six_spikes_and_separation():
    first = np.arange(6) * .05
    second = 2 + first
    train = np.r_[first, second]
    assert detect_bursts(train) == [(0, 5), (6, 11)]
    assert detect_bursts(first[:5]) == []
    assert detect_bursts(np.array([0, .11, .16, .21, .26, .31])) == []


def test_network_events_require_five_electrodes_in_three_ms_bin():
    five = {str(i): np.array([.1005, .1035]) for i in range(5)}
    events = detect_network_spikes(five, 1.0)
    assert len(events) == 1  # adjacent bins form one event
    assert sum(e["n_spikes"] for e in events) == 10
    assert all(e["peak_electrodes"] == 5 for e in events)
    assert detect_network_spikes(dict(list(five.items())[:4]), 1.0) == []


def test_sttc_and_multivariate_information():
    train = np.array([.1, .3, .5, .7, .9])
    assert spike_time_tiling_coefficient(train, train, 1.0) == pytest.approx(1.0)
    assert spike_time_tiling_coefficient(train, [], 1.0) is None
    assert normalized_multiinformation({"a": train, "b": train}, 1.0) > 0
    assert normalized_multiinformation({"a": train, "b": []}, 1.0) is None


def test_seventeen_features_on_known_synchronous_bursts():
    train = np.arange(6) * .05 + .1005
    x = extract_features({f"e{i}": train for i in range(5)}, 10.0)
    assert tuple(x) == FEATURES
    assert len(x) == 17
    assert x["firing_rate_mean"] == pytest.approx(.6)
    assert x["burst_rate"] == pytest.approx(6.0)
    assert x["bursting_electrodes_number"] == 5
    assert x["per_burst_spike_percent"] == pytest.approx(100.0)
    assert x["network_spike_number"] == 6
    assert x["network_spike_peak"] == 5
    assert x["correlation_coefficient_mean"] == pytest.approx(1.0)
    assert x["mutual_information_norm"] > 0


def test_undefined_vs_zero_and_input_validation():
    x = extract_features({"silent": []}, 10)
    assert x["network_spike_number"] == 0
    assert x["firing_rate_mean"] is None
    assert x["burst_duration_mean"] is None
    assert x["mutual_information_norm"] is None
    with pytest.raises(ValueError, match="Unsorted"):
        extract_features({"a": [.2, .1]}, 1)


def test_brewer_and_r1_audits():
    brewer = ROOT / "repo/results/brewer_features.json"
    r1 = ROOT / "repo/results/r1_feature_fidelity.json"
    if not brewer.exists() or not r1.exists():
        pytest.skip("Run run_brewer_features.py and audit_r1_sources.py")
    features = json.loads(brewer.read_text(encoding="utf-8"))
    audit = json.loads(r1.read_text(encoding="utf-8"))
    assert features["n_groups"] == 84
    assert len(features["rows"]) == 84
    assert all(row["n_electrodes"] == 19 for row in features["rows"])
    assert audit["plate_overlap"] == 0
    assert audit["plate_date_div_overlap"] == 0
    assert audit["metrics"] is None
