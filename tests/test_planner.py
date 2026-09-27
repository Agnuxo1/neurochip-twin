"""DoseCompass planner (R6). Synthetic tests run in CI; tests on the EPA NFA data skip when data/ is absent."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neurotwin.models.potency import BMR, chemical_potency
from neurotwin.models.trajectory import DIVS, ND, NF, ChemTask
from neurotwin.planner import (bmc_batch, mask_divs, mutual_info_bits, plan_step, potency_codes,
                               recommend_next, sequential_design)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LEVELS = np.log10([0.03, 0.1, 0.3, 1, 3, 10, 30]).astype(np.float32)


def hill(x, ac50=-0.8, top=6.0, h=2.0):
    return top / (1 + 10 ** (h * (ac50 - np.asarray(x, np.float64))))


def make_task(reps=3, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    logc = np.repeat(LEVELS, reps)
    y = np.repeat(hill(logc)[:, None, None], ND, 1).repeat(NF, 2).astype(np.float32)
    y += noise * rng.standard_normal(y.shape).astype(np.float32)
    return ChemTask("synthetic", 0, logc, y, np.ones_like(y, bool), np.array(["p|d"] * len(logc)))


class FakeCNP(torch.nn.Module):
    """Stand-in with the NeuroTrajectoryCNP call signature: predicts a known Hill curve, shifted by the mean
    context residual so that the output depends on the context wells (and only on them)."""
    likelihood = "gaussian"

    def __init__(self, sd=1.0):
        super().__init__()
        self.sd = sd

    def forward(self, cx, cy, cm, cmask, qx, qi=None):
        w = cmask.float().unsqueeze(-1)
        base_c = torch.as_tensor(hill(cx.numpy()), dtype=torch.float32)
        resid = ((cy - base_c) * cm * w).sum(1) / (cm * w).sum(1).clamp(min=1)
        mu = torch.as_tensor(hill(qx.numpy()), dtype=torch.float32).expand(-1, -1, ND * NF) + resid.unsqueeze(1)
        return mu, torch.full_like(mu, self.sd)


def test_bmc_batch_matches_potency_rule():
    rng = np.random.default_rng(0)
    g = np.linspace(-2, 2, 120).astype(np.float32)
    for _ in range(200):
        A = (rng.standard_normal((120, NF)) * 2 + rng.standard_normal(NF) * np.linspace(0, 3, 120)[:, None]).astype(np.float32)
        ref = chemical_potency(g, A)
        act, bmc = bmc_batch(g, A[None])
        assert bool(act[0]) == ref["active"]
        if ref["active"]:
            assert abs(bmc[0] - ref["bmc_log10"]) < 1e-5


def test_mutual_information_limits():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 4, 20000)
    assert mutual_info_bits(x, x)[0] == pytest.approx(2.0, abs=0.01)
    assert mutual_info_bits(x, rng.integers(0, 4, 20000))[0] < 0.01


def test_potency_codes():
    codes, n = potency_codes(np.array([True, True, False]), np.array([-1.0, 0.6, np.nan]), -1.0, 1.0)
    assert n == 10 and list(codes) == [0, 6, 9]


def test_recommend_next_targets_the_bmr_crossing():
    t = make_task()
    first = float(LEVELS[3])                    # strongly active: BMC lies below the median level
    chosen, scores = recommend_next([FakeCNP()], t, [first], device="cpu", tau=0.5, ell=0.25, n_samples=2000)
    assert set(round(k, 4) for k in scores) == set(round(float(v), 4) for v in LEVELS if v != LEVELS[3])
    assert chosen in [float(v) for v in LEVELS[:3]]
    assert max(scores[float(v)] for v in LEVELS[4:]) < 0.05       # levels above an active one carry ~no information
    assert chosen == max(scores, key=scores.get)


def test_recommend_next_restricts_to_given_candidates_and_needs_a_measured_level():
    t = make_task()
    cands = [float(LEVELS[0]), float(LEVELS[5]), float(LEVELS[3])]      # LEVELS[3] is already measured
    chosen, scores = recommend_next([FakeCNP()], t, [float(LEVELS[3])], cands, device="cpu", n_samples=500)
    assert set(scores) == {float(LEVELS[0]), float(LEVELS[5])} and chosen in scores
    with pytest.raises(ValueError):
        recommend_next([FakeCNP()], t, [], device="cpu", n_samples=100)


def test_planner_never_reads_unmeasured_wells():
    t = make_task(noise=0.5)
    first, second = float(LEVELS[3]), float(LEVELS[1])
    a = plan_step([FakeCNP()], t, [first, second], device="cpu", tau=0.5, ell=0.5, n_samples=1000)
    y = t.y.copy()
    hidden = ~np.isin(t.logc, [LEVELS[3], LEVELS[1]])
    y[hidden] = 9.0                              # corrupt every unmeasured well
    t2 = ChemTask(t.chem, t.fold, t.logc, y, t.m, t.plate)
    b = plan_step([FakeCNP()], t2, [first, second], device="cpu", tau=0.5, ell=0.5, n_samples=1000)
    assert a.chosen == b.chosen and a.eig_bits == b.eig_bits and a.p_active == b.p_active


def test_sequential_design_and_div_mask():
    t = make_task(noise=0.3)
    seq, steps = sequential_design([FakeCNP()], t, float(LEVELS[3]), 4, device="cpu", tau=0.5, ell=0.5, n_samples=500)
    assert len(seq) == 4 and len(set(seq)) == 4 and seq[0] == float(LEVELS[3]) and len(steps) == 4
    v = mask_divs(t)
    assert not v.m[:, DIVS.index(12)].any() and v.m[:, :3].all() and (v.y[:, 3] == 0).all()
    seq_d, _ = sequential_design([FakeCNP()], t, float(LEVELS[3]), 3, device="cpu", planner_task=v, tau=0.5, ell=0.5, n_samples=500)
    assert len(seq_d) == 3 and seq_d[0] == seq[0]


def test_fixed_log_spaced_rule():
    sys.path.insert(0, str(ROOT / "repo" / "scripts"))
    from run_dosecompass import first_index, fixed_design
    assert first_index(7) == 3 and first_index(8) == 3
    assert fixed_design(7, 3) == (0, 3, 6)
    assert fixed_design(7, 2) == (3, 6)
    assert fixed_design(7, 4) == (0, 3, 4, 6)
    assert fixed_design(8, 3) == (0, 3, 7)


# ---------------------------------------------------------------- real data (skipped in CI)
def _need_data():
    if not (DATA / "processed/epa_nfa/tasks_cache.pkl").exists() or not (DATA / "processed/models").exists():
        pytest.skip("EPA NFA data / fold models not present (CI has repo/ only)")


def test_real_chemical_no_peeking():
    _need_data()
    from neurotwin.planner import load_fold_models
    tasks, _ = pickle.load(open(DATA / "processed/epa_nfa/tasks_cache.pkl", "rb"))
    t = sorted([x for x in tasks if x.fold == 1], key=lambda x: x.chem)[0]
    models = load_fold_models(DATA / "processed/models", t.fold, "cpu")
    first = float(t.levels[(len(t.levels) - 1) // 2])
    c1, s1 = recommend_next(models, t, [first], device="cpu", tau=0.25, ell=0.5, n_samples=1000)
    y = t.y.copy()
    y[t.logc != t.levels[(len(t.levels) - 1) // 2]] *= -3.0
    t2 = ChemTask(t.chem, t.fold, t.logc, y, t.m, t.plate, t.label)
    c2, s2 = recommend_next(models, t2, [first], device="cpu", tau=0.25, ell=0.5, n_samples=1000)
    assert c1 == c2 and s1 == s2 and c1 in [float(v) for v in t.levels] and c1 != first


def test_results_json_consistent():
    p = ROOT / "repo/results/r6_dosecompass.json"
    if not p.exists():
        pytest.skip("run repo/scripts/run_dosecompass.py to create R6 evidence")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["status"] == "complete" and "protocol" in d and len(d["protocol_sha256"]) == 64
    prim = d["primary_endpoint"]
    r3 = d["results"]["B=3"]["neurotrajectory"]
    assert prim["dosecompass"] == r3["strategies"]["dosecompass"]["composite_error_mean"]
    assert prim["paired_bootstrap"] == r3["paired_bootstrap_composite"]["dosecompass_minus_fixed_logspaced"]
    for B in ("B=2", "B=3", "B=4"):
        for est in ("neurotrajectory", "loglinear_interp"):
            s = d["results"][B][est]["strategies"]
            assert s["oracle_upper_bound"]["composite_error_mean"] <= s["dosecompass"]["composite_error_mean"]
            assert s["oracle_upper_bound"]["composite_error_mean"] <= s["fixed_logspaced"]["composite_error_mean"]
    for row in d["per_chemical"]:
        assert row["dosecompass_order_idx"][0] == (row["n_levels"] - 1) // 2
        assert len(set(row["dosecompass_order_idx"])) == len(row["dosecompass_order_idx"]) == 4


# ---------------------------------------------------------------- R6b DoseCompass-anchored
def _r6b():
    sys.path.insert(0, str(ROOT / "repo" / "scripts"))
    import run_dosecompass_anchored as r6b
    return r6b


def test_r6b_anchor_always_included():
    r6b = _r6b()
    from run_dosecompass import fixed_design
    for L in range(4, 14):
        for B in (2, 3, 4):
            assert L - 1 in fixed_design(L, B)                       # comparator (a) already contains the top
            subs = r6b.anchored_subsets(L, B)
            assert subs and all(L - 1 in s and len(set(s)) == B for s in subs)
        orders = r6b.random_anchored_orders("chem-x", L, n=50)
        assert all(o[0] == L - 1 and sorted(o) == list(range(L)) for o in orders)
        v1o = r6b.random_v1_orders("chem-x", L, n=5)
        assert all(o[0] == (L - 1) // 2 for o in v1o)                # comparator (c) = v1 random (median first)
    t = make_task(noise=0.3)
    seq, steps = r6b.anchored_design([FakeCNP()], t, 4, "cpu", tau=0.5, ell=0.5, n_samples=500)
    assert seq[0] == float(LEVELS[-1]) and len(set(seq)) == 4 and len(steps) == 4
    assert all(float(LEVELS[-1]) in seq[:B] for B in (2, 3, 4))


def test_r6b_anchored_design_never_reads_hidden_wells():
    r6b = _r6b()
    t = make_task(noise=0.5)
    seq, st = r6b.anchored_design([FakeCNP()], t, 3, "cpu", tau=0.5, ell=0.5, n_samples=800)
    y = t.y.copy()
    y[~np.isin(t.logc, np.asarray(seq, np.float32))] = 9.0          # corrupt every well never measured
    t2 = ChemTask(t.chem, t.fold, t.logc, y, t.m, t.plate)
    seq2, st2 = r6b.anchored_design([FakeCNP()], t2, 3, "cpu", tau=0.5, ell=0.5, n_samples=800)
    assert seq == seq2
    assert all(a.eig_bits == b.eig_bits and a.p_active == b.p_active for a, b in zip(st, st2))


def test_r6b_real_chemical_no_hidden_wells():
    _need_data()
    r6b = _r6b()
    from neurotwin.planner import load_fold_models
    tasks, _ = pickle.load(open(DATA / "processed/epa_nfa/tasks_cache.pkl", "rb"))
    t = sorted([x for x in tasks if x.fold == 1], key=lambda x: x.chem)[0]
    models = load_fold_models(DATA / "processed/models", t.fold, "cpu")
    seq, _ = r6b.anchored_design(models, t, 3, "cpu", tau=0.5, ell=1.0, n_samples=500)
    y = t.y.copy()
    y[~np.isin(t.logc, np.asarray(seq, np.float32))] *= -3.0
    t2 = ChemTask(t.chem, t.fold, t.logc, y, t.m, t.plate, t.label)
    seq2, _ = r6b.anchored_design(models, t2, 3, "cpu", tau=0.5, ell=1.0, n_samples=500)
    assert seq == seq2 and seq[0] == float(t.levels[-1])


def test_r6b_results_json_consistent():
    p = ROOT / "repo/results/r6b_dosecompass_anchored.json"
    if not p.exists():
        pytest.skip("run repo/scripts/run_dosecompass_anchored.py to create R6b evidence")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "complete":
        pytest.skip("R6b registered but results pending")
    import hashlib
    r6b = _r6b()
    assert d["protocol"] == r6b.PROTOCOL
    assert hashlib.sha256(d["protocol"].encode()).hexdigest() == d["protocol_sha256"]
    assert d["protocol_registered_utc"] <= d.get("run_started_utc", d["protocol_registered_utc"])
    assert "second DoseCompass protocol" in d["multiplicity"] and "v1" in d["multiplicity"]
    assert d["v1_reference"]["unchanged_during_run"]
    v1p = ROOT / "repo/results/r6_dosecompass.json"
    if v1p.exists():
        assert hashlib.sha256(v1p.read_bytes()).hexdigest() == d["v1_reference"]["file_sha256_at_start"]
    prim = d["primary_endpoint"]
    r3 = d["results"]["B=3"]["neurotrajectory"]
    assert prim["dosecompass_anchored"] == r3["strategies"]["dosecompass_anchored"]["composite_error_mean"]
    assert prim["fixed_logspaced"] == r3["strategies"]["fixed_logspaced"]["composite_error_mean"]
    assert prim["paired_bootstrap"] == r3["paired_bootstrap_composite"]["dosecompass_anchored_minus_fixed_logspaced"]
    assert (d["secondary_endpoints"]["dosecompass_anchored_minus_random_anchored"]
            == r3["paired_bootstrap_composite"]["dosecompass_anchored_minus_random_anchored"])
    for B in ("B=2", "B=3", "B=4"):
        for est in ("neurotrajectory", "loglinear_interp"):
            s = d["results"][B][est]["strategies"]
            orc = s["oracle_anchored_upper_bound"]["composite_error_mean"]
            assert orc <= s["dosecompass_anchored"]["composite_error_mean"]
            assert orc <= s["fixed_logspaced"]["composite_error_mean"]
            assert orc <= s["random_anchored"]["expected_composite_error_mean"] + 1e-4
    for row in d["per_chemical"]:
        o = row["dosecompass_anchored_order_idx"]
        assert o[0] == row["anchor_idx"] == row["n_levels"] - 1
        assert len(set(o)) == len(o) == 4
    assert len(d["per_chemical"]) == d["n_chemicals"]
    means = np.mean([[row["composite_B3"]["neurotrajectory"][k] for k in ("dosecompass_anchored", "fixed_logspaced")]
                     for row in d["per_chemical"]], 0)
    assert abs(means[0] - prim["dosecompass_anchored"]) < 1e-3 and abs(means[1] - prim["fixed_logspaced"]) < 1e-3
