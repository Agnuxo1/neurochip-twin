"""Integration checks for the pinned Brewer 4-compartment MEA dataset."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))
from neurotwin.data.brewer import (  # noqa: E402
    CONDITIONS,
    DURATION_S,
    N_SAMPLES,
    SAMPLE_RATE_HZ,
    SUBREGIONS,
    TUNNELS,
    process,
)

RAW = ROOT / "data" / "raw" / "brewer_hfs"
OUT = ROOT / "data" / "processed" / "brewer"
AUDIT = ROOT / "repo" / "results" / "data_audit_brewer.json"


@pytest.fixture(scope="module")
def outputs():
    if not AUDIT.exists() or not (OUT / "recordings.csv").exists():
        process(RAW, OUT, AUDIT)
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def _table_rows(name: str):
    parquet = OUT / f"{name}.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq

        return pq.read_table(parquet).to_pylist()
    data = np.load(OUT / f"{name}.npz", allow_pickle=False)
    return [json.loads(item) for item in data["rows"]]


def test_sample_geometry_and_layout():
    assert SAMPLE_RATE_HZ == 25_000
    assert N_SAMPLES == 7_500_000
    assert DURATION_S == 300
    assert len(TUNNELS) == 20
    assert SUBREGIONS == {"EC", "DG", "CA3", "CA1"}
    for condition in CONDITIONS:
        with h5py.File(RAW / f"{condition}WellSpikes.mat") as file:
            root = next(value for key, value in file.items() if not key.startswith("#"))
            assert root.shape == (9 if condition == "NoStim" else 6, 1)
        with h5py.File(RAW / f"{condition}SortedAxons.mat") as file:
            root = next(value for key, value in file.items() if not key.startswith("#"))
            assert root.shape == (9 if condition == "NoStim" else 6, 1)


def test_well_counts_and_time_bounds(outputs):
    wells = _table_rows("wells")
    assert len(wells) == outputs["n_well_rows"] == 21 * 76
    counts = Counter((row["condition"], row["fid"]) for row in wells)
    assert set(counts.values()) == {76}
    for row in wells:
        assert row["subregion"] in SUBREGIONS
        assert row["n_spikes"] == len(row["spike_times_s"])
        times = row["spike_times_s"]
        assert all(0 <= time < DURATION_S for time in times)
        assert times == sorted(times)
    with (OUT / "recordings.csv").open(newline="", encoding="utf-8") as file:
        recordings = list(csv.DictReader(file))
    assert len(recordings) == outputs["n_recordings"] == 21
    for record in recordings:
        key = (record["condition"], int(record["fid"]))
        subset = [row for row in wells if (row["condition"], row["fid"]) == key]
        assert record["orientation"] in {"clockwise", "counterclockwise"}
        for region in SUBREGIONS:
            assert sum(row["n_spikes"] for row in subset if row["subregion"] == region) == int(record[f"n_spikes_{region}"])


def test_axon_pairs_and_directions(outputs):
    axons = _table_rows("axons")
    assert len(axons) == outputs["n_axons"]
    assert Counter(row["direction"] for row in axons) == {
        "ff": outputs["n_axons_ff"], "fb": outputs["n_axons_fb"]
    }
    seen = set()
    for row in axons:
        assert row["tunnel"] in TUNNELS
        assert row["direction"] in {"ff", "fb"}
        assert row["conduction_ms"] > 0
        assert row["axon_idx"] >= 1
        for key in ("upstream_spike_times_s", "downstream_spike_times_s"):
            assert all(0 <= time < DURATION_S for time in row[key])
        identity = (row["condition"], row["fid"], row["tunnel"], row["direction"], row["axon_idx"])
        assert identity not in seen
        seen.add(identity)
    assert sum(item["n_axons_ff"] + item["n_axons_fb"] for item in outputs["recordings"]) == len(axons)
