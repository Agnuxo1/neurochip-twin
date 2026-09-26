"""Topology, isolation and exported-evidence checks for Brewer ChipLayer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))

from neurotwin.chip.propagation import (  # noqa: E402
    PAIRS, TUNNELS, _fit, _predict, cross_validate, pair_for_tunnel, read_data,
)


def test_topology_changes_with_orientation():
    assert len(TUNNELS) == 20
    assert pair_for_tunnel("clockwise", "A6-A7") == "EC-DG"
    assert pair_for_tunnel("counterclockwise", "A6-A7") == "CA1-EC"
    assert pair_for_tunnel("clockwise", "F8-G8") == "DG-CA3"
    assert pair_for_tunnel("counterclockwise", "F8-G8") == "CA3-CA1"
    for orientation in ("clockwise", "counterclockwise"):
        assert {pair_for_tunnel(orientation, tunnel) for tunnel in TUNNELS} == set(PAIRS)


def test_fit_uses_only_passed_training_rows():
    row = {"tunnel": "A6-A7", "subregion_pair": "EC-DG", "direction": "ff", "conduction_ms": .24}
    fit = _fit([row] * 5)
    p1, d1, _ = _predict(fit, "A6-A7", "EC-DG")
    held_out = {**row, "direction": "fb", "conduction_ms": .88}
    _ = held_out  # construct a conflicting observation without fitting it
    p2, d2, _ = _predict(fit, "A6-A7", "EC-DG")
    assert p1 == p2 and d1 == d2
    assert _predict(_fit([held_out] * 5), "A6-A7", "EC-DG")[0] < p1


@pytest.mark.parametrize("condition", ["NoStim", "HFS5", "HFS40"])
def test_source_topology(condition):
    recordings, _, axons = read_data(ROOT / "data" / "processed" / "brewer")
    orientations = {(r["condition"], r["fid"]): r["orientation"] for r in recordings}
    for row in (r for r in axons if r["condition"] == condition):
        assert row["subregion_pair"] == pair_for_tunnel(orientations[(condition, row["fid"])], row["tunnel"])


def test_r8_outputs_and_split():
    path = ROOT / "repo" / "results" / "r8_chiplayer.json"
    if not path.exists():
        pytest.skip("Run repo/scripts/run_chiplayer.py to create R8 evidence")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["n_recordings"] == 21 and result["n_axons"] == 440
    assert len(result["graphs"]) == 21
    assert all(len(g["wells"]) == 76 and len(g["tunnels"]) == 20 and
               len(g["edges"]) == 4 for g in result["graphs"])
    folds = result["cv"]["folds"]
    assert len(folds) == 9
    assert sum(f["n_axons"] for f in folds) == sum(
        sum(t["n_axons_ff"] + t["n_axons_fb"] for t in g["tunnels"])
        for g in result["graphs"] if g["condition"] == "NoStim")
    for fold in folds:
        assert fold["condition"] == "NoStim"
        assert fold["fid"] not in fold["train_recordings"]
        assert len(fold["train_recordings"]) == 8
        assert len(fold["predictions"]) == fold["n_observed_tunnels"]
        assert all(0 <= p["model"]["ff_fraction"] <= 1 for p in fold["predictions"])
