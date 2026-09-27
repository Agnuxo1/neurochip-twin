"""Tests for R11 early-exit triage (scripts/run_early_exit.py -> results/r11_early_exit.json).

Synthetic tests run everywhere; the JSON coherence test skips when the result file is absent.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))
sys.path.insert(0, str(ROOT / "repo" / "scripts"))
OUT = ROOT / "repo" / "results" / "r11_early_exit.json"

import run_early_exit as EE  # noqa: E402


def test_protocol_hash_matches_registration():
    assert hashlib.sha256(EE.PROTOCOL.read_bytes()).hexdigest() == EE.PROTOCOL_SHA


def test_nested_threshold_keeps_95pct_of_other_folds_actives_and_ignores_own_fold():
    rng = np.random.default_rng(1)
    s = rng.random(400)
    active = rng.random(400) < 0.7
    folds = np.repeat([1, 2, 3, 4], 100)
    for f in (1, 2, 3, 4):
        tau = EE.nested_threshold(s, active, folds, f)
        other = (folds != f) & active
        assert (s[other] >= tau).mean() >= EE.SENS_TARGET
        s2 = s.copy(); s2[folds == f] = 0.0          # own fold must not matter
        assert EE.nested_threshold(s2, active, folds, f) == tau


def test_early_task_hides_div9_and_div12():
    from neurotwin.models.trajectory import ChemTask, ND, NF
    n = 6
    t = ChemTask("x", 1, np.repeat([0.0, 1.0, 2.0], 2).astype(np.float32),
                 np.ones((n, ND, NF), np.float32), np.ones((n, ND, NF), bool), np.array(["p"] * n))
    e = EE.early_task(t)
    assert e.m[:, EE.EARLY].all() and not e.m[:, 2:].any() and not e.y[:, 2:].any()


def test_result_json_coherent():
    if not OUT.exists():
        pytest.skip("r11_early_exit.json not generated")
    d = json.loads(OUT.read_text(encoding="utf-8"))
    assert d["protocol_sha256"] == EE.PROTOCOL_SHA
    assert d["n_reference_active"] + d["n_reference_inactive"] == d["n_chemicals"]
    for m in EE.METHODS:
        e = d["methods"][m]
        assert e["tp"] + e["fn"] == d["n_reference_active"]
        assert e["tn"] + e["fp"] == d["n_reference_inactive"]
        assert 0 <= e["auroc"] <= 1 and e["sensitivity_ci95"][0] <= e["sensitivity"] <= e["sensitivity_ci95"][1]
    p = d["primary"]
    assert p["ci95"][0] <= p["diff_inactive_exit_rate"] <= p["ci95"][1]


def test_r11b_json_coherent():
    p = ROOT / "repo" / "results" / "r11b_early_exit.json"
    if not p.exists():
        pytest.skip("r11b_early_exit.json not generated")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["protocol_sha256"] == hashlib.sha256((ROOT / "repo/docs/prereg/r11b_early_exit_retrained.md").read_bytes()).hexdigest()
    for ref, blk in d["references"].items():
        assert blk["n_active"] + blk["n_inactive"] == d["n_chemicals"]
        for m, e in blk["methods"].items():
            assert e["tp"] + e["fn"] == blk["n_active"] and e["tn"] + e["fp"] == blk["n_inactive"]
    # twin_v1 in R11b must reproduce the R11 twin exactly
    r11 = json.loads(OUT.read_text(encoding="utf-8"))
    assert d["references"]["reference_active"]["methods"]["twin_v1"]["auroc"] == r11["methods"]["twin"]["auroc"]
    ni = d["noninferiority_dose_interpolation"]
    assert ni["noninferior_upper_le_2pct"] == (ni["rel_change_ci95_pct"][1] <= 2.0)
