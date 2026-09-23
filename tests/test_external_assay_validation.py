import numpy as np
import pandas as pd
import pytest

from src.external_assay_validation import (
    _bootstrap_mean_ci,
    _melt_dose_blocks,
    _paired_group_mae_difference,
    make_group_splits,
)


def test_dose_block_reader_unpivots_and_maps_vehicle_to_zero():
    grid = [
        ["Doses are in uM"],
        ["", "Test compound"],
        ["", "Ctrl", "1", "5", "10", "50", "100"],
        ["run-1", "10", "11", "12", "13", "14", "15"],
    ]

    frame = _melt_dose_blocks(grid, "NLPN")

    assert frame.columns.tolist() == ["assay", "date", "chemical", "dose_uM", "target"]
    assert len(frame) == 6
    assert frame.iloc[0].to_dict() == {
        "assay": "NLPN",
        "date": "run-1",
        "chemical": "Test compound",
        "dose_uM": 0.0,
        "target": 10.0,
    }
    assert frame.dose_uM.tolist() == [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]


def test_date_and_unseen_chemical_splits_do_not_leak_groups():
    frame = pd.DataFrame(
        [
            {"date": date, "chemical": chemical, "pre_rate": 4.0, "target": 0.8}
            for date in ("d1", "d2")
            for chemical in ("A", "B", "CTR")
        ]
    )

    for train, test in make_group_splits(frame, "date"):
        assert set(frame.iloc[train].date).isdisjoint(frame.iloc[test].date)

    chemical_splits = make_group_splits(frame, "chemical")
    assert len(chemical_splits) == 2
    for train, test in chemical_splits:
        held_out = set(frame.iloc[test].chemical)
        assert len(held_out) == 1
        assert "CTR" not in held_out
        assert "CTR" in set(frame.iloc[train].chemical)
        assert held_out.isdisjoint(set(frame.iloc[train].chemical))


def test_chemical_holdout_never_scores_vehicle_controls():
    frame = pd.DataFrame(
        [
            {"date": "d1", "chemical": chemical, "pre_rate": 1.0, "target": 1.0}
            for chemical in ("A", "B", "CTR")
        ]
    )

    covered = np.concatenate([test for _, test in make_group_splits(frame, "chemical")])

    assert sorted(frame.iloc[covered].chemical.tolist()) == ["A", "B"]


def test_bootstrap_interval_is_deterministic_and_contains_constant_mean():
    assert _bootstrap_mean_ci([2.0, 2.0, 2.0]) == [2.0, 2.0]


def test_paired_group_comparison_requires_same_groups_and_reports_direction():
    forest = {
        "fold_metrics": [
            {"group": "A", "mae": 1.0},
            {"group": "B", "mae": 3.0},
        ]
    }
    ridge = {
        "fold_metrics": [
            {"group": "A", "mae": 2.0},
            {"group": "B", "mae": 4.0},
        ]
    }

    result = _paired_group_mae_difference(forest, ridge)

    assert result["n_groups"] == 2
    assert result["forest_minus_ridge_mae"] == -1.0
    assert result["bootstrap_95ci"] == [-1.0, -1.0]

    ridge["fold_metrics"].pop()
    with pytest.raises(ValueError, match="identical held-out groups"):
        _paired_group_mae_difference(forest, ridge)
