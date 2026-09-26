"""Tests for the G5 cross-assay corroboration (X2 Harrill imaging, X3 Kosnik acute MEA).

Synthetic tests run everywhere (CI has only repo/); tests that need the downloaded datasets or the
generated results skip when those are absent.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))
sys.path.insert(0, str(ROOT / "repo" / "scripts"))
DATA = ROOT / "data"
RESULTS = ROOT / "repo" / "results"
HARRILL_XLSX = DATA / "raw" / "epa_dnt" / "Harrill_DNT_Assay_Dataset.xlsx"
KOSNIK_DIR = DATA / "neurotox_probe" / "kosnik2020"

from neurotwin.data import harrill as H  # noqa: E402
from neurotwin.data import kosnik as K  # noqa: E402
from neurotwin.models.potency import bmc_from_curve  # noqa: E402
from neurotwin.models.trajectory import NF  # noqa: E402


# ------------------------------------------------------------------ synthetic
def test_first_crossing_matches_nfa_potency_rule():
    rng = np.random.default_rng(0)
    grid = np.linspace(-2, 2, 120)
    for _ in range(20):
        A = np.cumsum(rng.normal(0, 0.4, (120, NF)), 0)
        ref, _ = bmc_from_curve(grid, A)
        ours = np.array([H.first_crossing(grid, np.abs(A[:, f])) for f in range(NF)])
        assert np.allclose(np.nan_to_num(ref, nan=99), np.nan_to_num(ours, nan=99))


def _raw_plate():
    rows, dbid = [], 0

    def add(aid, sample, wtype, wq, conc, raw, row, col, plate="P1"):
        nonlocal dbid
        dbid += 1
        rows.append((dbid, aid, sample, plate, row, col, wtype, wq, conc, raw))
    for aid in (2, 3):
        for i, v in enumerate([100.0, 104.0, 96.0]):
            add(aid, "DMSO " if i else "DMSO", "n", 1, 0.1, v, i + 1, 1)
        add(aid, "Valproate", "t", 1, 1.0, 50.0, 1, 2)
        add(aid, "Valproic Acid", "t", 1, 1.0, 50.0, 2, 2)
        add(aid, "Valproic Acid", "t", 0, 3.0, 1.0, 3, 2)          # rejected well
        add(aid, "Blank", "b", 0, 0.0, 0.0, 4, 2)
    return pd.DataFrame(rows, columns=H.RAW_COLUMNS)


def test_curate_quality_alias_and_vehicle_normalisation():
    t, noise, audit = H.curate(_raw_plate())
    assert audit["n_rows_quality0"] == 4 and audit["n_test_rows_quality0"] == 2
    assert audit["n_test_wells_quality0"] == 1
    assert set(t["sample"]) == {"Valproic Acid"} and (t.conc == 1.0).all()
    assert np.allclose(t.pct, -50.0)                               # 50 / median(100, 104, 96) - 1
    # leave-one-out vehicle scale: deviations of each vehicle well from the median of the other two
    assert noise.loc[2, "n_vehicle_wells"] == 3 and noise.loc[2, "sd_vehicle_pct"] > 0


def _levels(chem, aid, resp, logc=(-2, -1, 0, 1, 2)):
    return pd.DataFrame({"sample": chem, "aid": aid, "logc": list(logc), "resp": list(resp), "n_wells": 3})


def test_window_selective_and_censored():
    # morphology (neurite length, aid 7) drops at 0.5 log uM while neuron count (aid 6) never drops
    lv = pd.concat([_levels("A", 7, [0, 0, -1, -5, -8]), _levels("A", 6, [0, 0, 0, -1, -1]),
                    _levels("B", 7, [0, 0, 0, -1, -5]), _levels("B", 6, [0, -6, -8, -9, -9])])
    ct = H.chemical_table(lv).set_index("harrill_name")
    a, b = ct.loc["A"], ct.loc["B"]
    assert a.morph_active and not a.count_active and a.selective_morph and a.window_censored
    assert np.isclose(a.morph_bmc_log10, 0.5, atol=0.03) and np.isclose(a.window_log10, 2 - a.morph_bmc_log10)
    assert b.morph_active and b.count_active and not b.selective_morph and b.window_log10 < 0


def test_direction_and_bidirectional_rule():
    lv = _levels("C", 3, [0, 1, 4, 6, 8])                         # neurite length INCREASES
    assert not H.chemical_table(lv).morph_active.iloc[0]             # declared direction is loss
    bi = H.chemical_table(lv, bidirectional=True).iloc[0]
    assert bi.morph_active and bi.morph_direction == 1


def test_harrill_name_matching():
    nfa = ["Chlorpyrifos oxon", "Sodium valproate", "(-)-Nicotine", "Manganese dichloride"]
    m = H.match_nfa(["Chlorpyrifos Oxon", "Valproic Acid", "Nicotine", "Manganese"], nfa).set_index("harrill_name")
    assert m.loc["Chlorpyrifos Oxon", "match_type"] == "exact"
    assert m.loc["Valproic Acid", "match_type"] == "salt_or_form"
    assert m.loc["Nicotine", "nfa_chem"] == "(-)-Nicotine"
    assert m.loc["Manganese", "match_type"] == "unmatched"


def test_kosnik_matching_by_cas_then_name():
    calls = pd.DataFrame({"spid": ["S1", "S2"], "casn": ["50-29-3", "999-99-9"], "chnm": ["p,p'-DDT", "Foo"],
                          "final_active": [True, False], "n_final_endpoints": [4, 0],
                          "acute_bmc_log10": [0.5, np.nan], "any_hit_active": [True, True]})
    calls["cas"] = calls.casn.map(K.clean_cas)
    calls["nk"] = calls.chnm.map(K.name_key)
    cw = pd.DataFrame({"dtxsid": ["DTXSID1"], "casrn": ["50-29-3"]})
    nfa = pd.DataFrame({"chem": ["DDT", "foo", "Bar"], "dtxsid": ["DTXSID1", "DTXSID2", "DTXSID3"]})
    m = K.match_nfa(nfa, calls, cw).set_index("chem")
    assert m.loc["DDT", "match"] == "dtxsid_via_casrn" and m.loc["DDT", "acute_final_active"]
    assert m.loc["foo", "match"] == "name" and "Bar" not in m.index
    assert K.clean_cas("1/7/8018") is None and K.clean_cas(" 58-89-9 ") == "58-89-9"


def test_classification_statistics():
    from run_crossassay import auroc, bal_acc, kappa, spearman
    rng = np.random.default_rng(1)
    y = rng.random(60) < 0.4
    s = rng.normal(size=60) + y
    pairs = (s[y][:, None] > s[~y][None, :]).mean() + 0.5 * (s[y][:, None] == s[~y][None, :]).mean()
    assert np.isclose(auroc(y, s), pairs)                          # Mann-Whitney definition
    assert np.isclose(bal_acc(y, y), 1.0) and np.isnan(bal_acc(np.ones(5, bool), np.ones(5, bool)))
    assert np.isclose(kappa(y, y), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)


# ------------------------------------------------------------------ real data (skip in CI)
def _need(path):
    if not Path(path).exists():
        pytest.skip(f"data not available: {path}")


def test_harrill_curation_audit_on_real_file():
    _need(HARRILL_XLSX)
    cache = DATA / "processed" / "crossassay" / "harrill_raw.parquet"
    raw = H.load_raw(HARRILL_XLSX, cache if cache.exists() else None)
    assert len(raw) == H.N_ROWS_EXPECTED
    _, noise, audit = H.curate(raw)
    assert audit["n_test_rows_quality0"] == 11 and audit["n_test_wells_quality0"] == 5
    assert audit["n_endpoints"] == 21 and len(audit["concentrations_uM"]) == 14
    assert audit["concentrations_uM"][0] == 0.001 and audit["concentrations_uM"][-1] == 3000.0
    assert audit["n_sample_names_test_raw"] == 75 and audit["n_chemicals_test_after_alias"] == 74
    assert H.check_endpoint_sheet(HARRILL_XLSX)["mismatches"] == []
    assert (noise.sd_vehicle_pct > 0).all()


def test_kosnik_parse_audit_on_real_files():
    _need(KOSNIK_DIR / "MEA_Data" / "MEA_Truth_Chemicals.csv")
    k = K.load(KOSNIK_DIR)
    a = K.audit(k)
    assert a["checks_pass"] and a["n_spids"] == 384
    assert a["truth_counts"] == {"Neuroactive": 41, "Negative": 32}
    calls = K.acute_calls(k)
    assert calls.final_active.sum() == a["n_final_active_spids"]
    assert (calls.loc[calls.final_active, "n_final_endpoints"] >= 3).all()


def test_results_json_consistency():
    x2p, x3p = RESULTS / "x2_harrill.json", RESULTS / "x3_kosnik.json"
    if not (x2p.exists() and x3p.exists()):
        pytest.skip("cross-assay results not generated")
    x2, x3 = json.loads(x2p.read_text(encoding="utf-8")), json.loads(x3p.read_text(encoding="utf-8"))
    assert "UNPAIRED" in x2["label"] and "UNPAIRED" in x3["label"]
    per = pd.DataFrame(x2["per_chemical"])
    prim = per[per.match_type.isin(["exact", "synonym"])]
    po = x2["primary_overlap"]
    assert po["n_chemicals"] == len(prim) == x2["headline"]["n_overlap_primary"]
    assert po["activity_nfa_vs_harrill_morph"]["nfa_active"] == int(prim.nfa_active.sum())
    assert po["activity_nfa_vs_harrill_morph"]["harrill_morph_active"] == int(prim.morph_active.sum())
    both = prim[prim.nfa_active & prim.morph_active]
    assert po["potency_spearman_both_active"]["n"] == len(both)
    assert x2["headline"]["kappa"] == po["activity_nfa_vs_harrill_morph"]["kappa"]
    r = x3["measured_acute_call"]["nfa_reference"]
    pc = pd.DataFrame(x3["per_chemical"])
    assert r["n"] == len(pc) and r["n_positive"] == int(pc.acute_final_active.sum())
    for key in ("balanced_accuracy", "balanced_accuracy_ci95", "auroc", "auroc_ci95"):
        assert x3["headline"][key] == r[key]
    assert x3["headline"]["n_overlap_measured"] == r["n"]
    assert x2["headline"]["spearman_potency_both_active"] == po["potency_spearman_both_active"]["spearman"]
    assert "NEUROACTIVITY" in x3["label_semantics"] and "not a molecular-target" in x3["label_semantics"]
    assert x3["truth_set"]["status"] in {"ok", "appendix_only"}
    for blk in (r, x3["truth_set"]["nfa_reference"]):
        if blk.get("balanced_accuracy") is not None:
            lo, hi = blk["balanced_accuracy_ci95"]
            assert lo <= blk["balanced_accuracy"] <= hi
    assert (RESULTS / "figures" / "crossassay.png").exists()
