"""G0 cohort and split invariants, including the content freeze."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "scripts"))
from check_freeze_g0 import verify  # noqa: E402

RESULTS = ROOT / "repo" / "results"


def _read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_freeze_hashes_and_unmodified_chemical_split():
    checked = verify(ROOT)
    assert checked["status"] == "verified"
    assert checked["checked_files"] >= 29
    manifest = _read("freeze_g0.json")
    existing = ROOT / "repo" / "results" / "splits_epa_nfa.json"
    assert manifest["chemical_split_unchanged_sha256"] == manifest["files"]["repo/results/splits_epa_nfa.json"]
    assert existing.is_file()
    assert manifest["x1_targets"].startswith("deferred")


def test_cohort_aliases_and_qc():
    cohort = _read("cohort_freeze.json")
    assert len(cohort["cohort"]) == 267
    assert len({row["dtxsid"] for row in cohort["cohort"]}) == 243
    assert cohort["qc"]["exclude_wllq_zero"]
    assert cohort["qc"]["n_rows_wllq_zero"] == 0
    assert cohort["normalization"]["group_by"] == ["plate", "date", "div"]
    aliases = {row["group"]: row for row in cohort["aliases"]}
    assert not aliases["valproate_active_moiety"]["collapse_identity"]
    assert not aliases["heptachlor_metabolite_pair"]["collapse_identity"]
    assert len({member["dtxsid"] for member in aliases["heptachlor_metabolite_pair"]["members"]}) == 2


def test_dose_levels_are_wholly_held_out():
    dose = _read("splits_dose.json")
    wells = pd.read_parquet(ROOT / "data" / "processed" / "epa_nfa" / "wells.parquet",
                            columns=["dtxsid", "concentration_uM"])
    expected = {(row.dtxsid, float(row.concentration_uM))
                for row in wells[wells.concentration_uM.gt(0)].drop_duplicates().itertuples()}
    actual = [(row["dtxsid"], row["concentration_uM"]) for row in dose["assignments"]]
    assert len(actual) == len(set(actual)) == dose["n_levels"] == 1783
    assert set(actual) == expected
    assert all(concentration > 0 for _, concentration in actual)
    folds = Counter(row["test_fold"] for row in dose["assignments"])
    assert set(folds) == set(range(5))
    assert max(folds.values()) - min(folds.values()) < 50


def test_plate_and_family_holdouts():
    plate = _read("splits_plate.json")
    family = _read("splits_family.json")
    by_plate = defaultdict(set)
    for row in plate["assignments"]:
        by_plate[row["plate"]].add(row["test_fold"])
    assert len(by_plate) == plate["n_plates"] == 163
    assert len(plate["assignments"]) == plate["n_plate_dates"] == 169
    assert all(len(folds) == 1 for folds in by_plate.values())
    assert sum(plate["fold_row_counts"]) == 31_756
    assert max(plate["fold_row_counts"]) - min(plate["fold_row_counts"]) < 100
    assert family["n_eligible_chemicals"] == 206
    assert family["n_excluded_chemicals"] == 37
    by_family = defaultdict(set)
    for row in family["assignments"]:
        if row["family"] is not None:
            by_family[row["family"]].add(row["test_fold"])
        else:
            assert row["test_fold"] is None
    assert len(by_family) == family["n_families"] == 42
    assert all(len(folds) == 1 for folds in by_family.values())


def test_reference_revision_sensitivity():
    report = _read("epa_baseline_repro.json")
    assert report["reference_revision"] == "dnt_ref_tbl2"
    revised = report["reference_sensitivity"]
    assert (revised["n_positive"], revised["n_negative"]) == (102, 18)
    assert (report["models"]["DIV12"]["balanced_accuracy_pct"],
            revised["models"]["DIV12"]["balanced_accuracy_pct"]) == (84.0, 76.6)
    assert (report["models"]["Top2"]["balanced_accuracy_pct"],
            revised["models"]["Top2"]["balanced_accuracy_pct"]) == (87.2, 81.4)
    assert all(not value for item in revised["models"].values()
               for value in item["published_delta_pp"].values())
