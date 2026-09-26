"""Integration checks for the EPA NFA trajectory and baseline conversion."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))
sys.path.insert(0, str(ROOT / "data" / ".pylibs"))
from neurotwin.data.epa_nfa import FEATURES, frozen_splits, process  # noqa: E402

RAW = ROOT / "data" / "neurotox_probe" / "nfa_refine"
OUT = ROOT / "data" / "processed" / "epa_nfa"
RESULTS = ROOT / "repo" / "results"


@pytest.fixture(scope="module")
def generated():
    if not (OUT / "wells.parquet").exists() or not (RESULTS / "epa_baseline_repro.json").exists():
        process(RAW, OUT, RESULTS)
    return (
        pd.read_parquet(OUT / "wells.parquet"),
        pd.read_parquet(OUT / "vehicle_stats.parquet"),
        pd.read_parquet(OUT / "dnt_labels.parquet"),
        json.loads((RESULTS / "epa_baseline_repro.json").read_text()),
        json.loads((RESULTS / "splits_epa_nfa.json").read_text()),
    )


def test_well_schema_and_vehicle_normalization(generated):
    wells, stats, _, _, _ = generated
    assert len(wells) == 31_756
    assert len(FEATURES) == 17
    assert set(wells["div"]) == {5, 7, 9, 12}
    assert set(wells.well_type) == {"vehicle", "treated"}
    assert set(wells.concentration_uM[wells.well_type.eq("vehicle")]) == {0}
    assert sum(stats.n_vehicle_wells) == 4_112
    assert wells.dtxsid.notna().all()
    group = stats.iloc[0]
    mask = (wells.plate == group.plate) & (wells.date == group.date) & (wells["div"] == group["div"])
    controls = wells[mask & wells.well_type.eq("vehicle")].drop_duplicates("well")
    feature = "firing_rate_mean"
    median = controls[feature].median()
    mad = 1.4826 * np.median(np.abs(controls[feature].dropna() - median))
    assert median == pytest.approx(group[f"vehicle_median_{feature}"])
    assert mad == pytest.approx(group[f"vehicle_mad_{feature}"])
    row = wells[mask & wells[feature].notna()].iloc[0]
    if mad > 0:
        assert row[f"z_vehicle_{feature}"] == pytest.approx((row[feature] - median) / mad)


def test_reference_provenance(generated):
    _, _, labels, baseline, _ = generated
    assert len(labels) == 267
    assert labels.spid.is_unique
    assert labels.dtxsid.nunique() == 243
    assert labels.dtxsid.notna().all()
    assert labels.label.dropna().isin({"Positive", "Negative"}).all()
    for source in ("epa_ref_tbl2", "shafer_ref_tbl", "oecd_appendix_a", "sept2025_reference", "efsa_listed"):
        assert source in labels
    assert baseline["n_positive"] == 86
    assert baseline["n_negative"] == 19
    assert baseline["n_reference"] == 105
    assert baseline["n_fitted"] == 255


def test_reproduced_endpoint_hit_calls_and_metrics(generated):
    _, _, _, report, _ = generated
    assert report["models"]["DIV12"]["tp"] == 63
    assert report["models"]["DIV12"]["tn"] == 18
    assert report["models"]["DIV12"]["balanced_accuracy_pct"] == 84.0
    assert report["models"]["Top2"]["tp"] == 64
    assert report["models"]["Top2"]["tn"] == 19
    assert report["models"]["Top2"]["balanced_accuracy_pct"] == 87.2
    assert all(not value for value in report["hit_sum_mismatches"].values())
    assert all(not delta for result in report["models"].values() for delta in result["published_delta_pp"].values())
    cache = OUT / "upstream"
    div12 = pd.read_excel(cache / "DIV12_analysis_tcplfit2_res_02May2025.xlsx")
    auc = pd.read_excel(cache / "AUC_analysis_tcplfit2_res_30Apr2025.xlsx")
    importance = pd.read_csv(cache / "RF_importance_AUC_29feb26.csv")
    published = pd.read_csv(cache / "Bioactivity_bin_tbl_comp_methods.csv").set_index("spid")
    div_hits = div12.groupby("name").hitc.sum()
    top2 = importance.Feature.head(2).str.replace("AUC", "CCTE_Shafer_MEA_dev", regex=False)
    top_hits = auc[auc.assay.isin(top2)].groupby("name").hitc.sum()
    assert div_hits.equals(published.loc[div_hits.index, "div12.hitsum"].rename("hitc"))
    assert top_hits.equals(published.loc[top_hits.index, "top2.feature.hitsum"].rename("hitc"))


def test_folds_keep_every_chemical_together(generated):
    _, _, labels, _, split = generated
    assert split["seed"] == 20260926
    assert split["n_folds"] == 5
    assert len(split["assignments"]) == 267
    by_id = defaultdict(set)
    by_name = defaultdict(set)
    for row in split["assignments"]:
        by_id[row["dtxsid"]].add(row["fold"])
        by_name[" ".join(row["chemical"].casefold().split())].add(row["fold"])
    assert all(len(folds) == 1 for folds in by_id.values())
    assert all(len(folds) == 1 for folds in by_name.values())
    assert set(row["fold"] for row in split["assignments"]) == set(range(5))
    assert Counter(row["spid"] for row in split["assignments"]).most_common(1)[0][1] == 1
    assert frozen_splits(labels) == split
