"""Tests of the Darwin-Cage port (synthetic, always run) and of the R10 audit outputs (skip when data are absent)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurotwin.audit.darwin import (Alphabet, AtomCache, DarwinSearch, atom_values, crossover, integer_view,
                                    mutate, plant, plant_residual, program_key, random_atom, uses_atoms)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent


def original_program_key(prog, D):
    """The mixed-radix key exactly as in the original honest_ceiling/darwin.py."""
    key = np.zeros(len(next(iter(D.values()))), dtype=np.int64)
    for atom in prog:
        codes = pd.factorize(atom_values(atom, D))[0].astype(np.int64)
        key = key * (int(codes.max()) + 1) + codes
        if key.max() > 2 ** 62:
            key = pd.factorize(key)[0].astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def synth(n_chem=60, rows=200, seed=0, chem_sd=1.0, noise_sd=1.0):
    """Residual table clustered by chemical: r = chemical offset + noise; two categorical columns, one numeric."""
    rng = np.random.default_rng(seed)
    chem = np.repeat(np.arange(n_chem), rows)
    df = pd.DataFrame({"chem": chem, "feat": rng.integers(0, 8, len(chem)), "div": rng.integers(0, 4, len(chem)),
                       "x": rng.normal(size=len(chem))})
    r = rng.normal(0, chem_sd, n_chem)[chem] + rng.normal(0, noise_sd, len(chem))
    fold = chem % 4 + 1
    D = integer_view(df, ["feat", "div"], {"x": 10})
    return df, D, r, fold, chem


ALPHA = Alphabet(cats=("feat", "div"), nums=("x",), ops=("raw", "div", "qbin"), divs=(2, 5, 10), qbins=(2, 4, 8))


def same_partition(a, b):
    pairs = pd.DataFrame({"a": a, "b": b}).drop_duplicates()
    return pairs.a.is_unique and pairs.b.is_unique


def test_program_key_matches_original():
    _, D, _, _, _ = synth(n_chem=10, rows=50)
    cache = AtomCache(D, max_items=2)
    for prog in ([("cat", "feat")], [("cat", "feat"), ("cat", "div")], [("div", "x", 5), ("cat", "div")],
                 [("qbin", "x", 4), ("cat", "feat"), ("raw", "x")]):
        k0 = original_program_key(prog, D)
        for k in (program_key(prog, D, cache), program_key(prog, D)):
            assert k.min() == 0 and k.max() == k0.max() and same_partition(k, k0)


def test_fitness_and_confirm_equal_original_masked_statistics():
    df, D, r, fold, chem = synth(n_chem=24, rows=60, seed=4)
    s = DarwinSearch(D=D, r=r, fold=fold, alphabet=ALPHA, confirm_fold=4, groups=chem, n_perm=10)
    key = program_key([("cat", "feat"), ("qbin", "x", 4)], D)
    nk = int(key.max()) + 1

    def est(fit, ho):     # the original _fitness / confirm arithmetic
        sm = np.bincount(key[fit], weights=r[fit], minlength=nk)
        c = np.bincount(key[fit], minlength=nk)
        return (sm / (c + 5.0))[key[ho]]
    conf = fold == 4
    cors = [np.corrcoef(est((fold != k) & ~conf, fold == k), r[fold == k])[0, 1] for k in (1, 2, 3)]
    f = s.fitness(key)
    assert np.isclose(f[0], np.mean(cors)) and np.isclose(f[1], np.min(cors))
    assert np.isclose(s.confirm(key), np.corrcoef(est(~conf, conf), r[conf])[0, 1])


def test_qbin_and_random_atoms_respect_alphabet():
    _, D, _, _, _ = synth(n_chem=10, rows=100)
    v = atom_values(("qbin", "x", 4), D)
    assert v.min() == 0 and v.max() == 3 and abs(np.bincount(v).std() / np.bincount(v).mean()) < 0.1
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = random_atom(rng, ALPHA)
        assert a[0] in ("cat", "raw", "div", "qbin") and a[1] in ("feat", "div", "x")
        if a[0] in ("div", "qbin"):
            assert a[2] in ALPHA.params(a[0])
    p = [random_atom(rng, ALPHA) for _ in range(3)]
    for _ in range(100):
        p = mutate(p, rng, ALPHA)
        assert 1 <= len(p) <= 4
        p = crossover(p, [random_atom(rng, ALPHA)], rng)
        assert 1 <= len(p) <= 4


def test_plant_residual_adds_exact_structure():
    df, D, r, _, _ = synth(n_chem=8, rows=40)
    rp, shift = plant_residual(r, D, [("cat", "feat")], 0.2, seed=3)
    assert np.allclose(rp - r, shift) and np.allclose(np.abs(shift), 0.2)
    # constant within each key value
    assert pd.Series(shift).groupby(df.feat.to_numpy()).nunique().max() == 1
    y, sh = plant(np.full(len(r), 0.5), D, [("cat", "div")], 0.1, seed=0)   # original Bernoulli plant kept
    assert set(np.unique(y)) <= {0, 1} and np.allclose(np.abs(sh), 0.1)


def test_uses_atoms():
    assert uses_atoms([["cat", "feat"], ["cat", "div"]], [("cat", "div")])
    assert not uses_atoms([["cat", "feat"]], [("cat", "feat"), ("cat", "div")])


def test_search_finds_and_confirms_a_planted_structure():
    df, D, r, fold, chem = synth(n_chem=80, rows=150, chem_sd=0.3, noise_sd=1.0, seed=1)
    rp, _ = plant_residual(r, D, [("cat", "feat"), ("cat", "div")], 0.5, seed=0)
    s = DarwinSearch(D=D, r=rp, fold=fold, alphabet=ALPHA, confirm_fold=4, groups=chem, n_perm=2000)
    res = s.run(max_evals=150, pop=30, seed=0, log_every=5)
    assert res["confirmed"], "a 0.5 SD interaction on 12k rows must be confirmed"
    assert any(uses_atoms(h["program"], [("cat", "feat")]) for h in res["confirmed"])
    assert res["confirmed"][0]["p_signflip"] < s.p_one and "p_fwer_maxT" in res["confirmed"][0]


def test_cluster_signflip_blocks_chemical_level_false_positives():
    """Pure chemical-level noise that lines up with a numeric column by chance within chemicals: the original
    z/sqrt(n) rule is anti-conservative under clustering; the sign-flip test must control it."""
    rng = np.random.default_rng(5)
    n_chem, rows = 40, 400
    chem = np.repeat(np.arange(n_chem), rows)
    # a per-chemical covariate (constant within chemical) and a residual dominated by chemical offsets
    df = pd.DataFrame({"feat": rng.integers(0, 4, len(chem)), "div": rng.integers(0, 4, len(chem)),
                       "x": np.repeat(rng.normal(size=n_chem), rows)})
    D = integer_view(df, ["feat", "div"], {"x": 10})
    fold = chem % 4 + 1
    false_orig, false_new = 0, 0
    for rep in range(12):
        r = np.random.default_rng(100 + rep).normal(0, 1.0, n_chem)[chem] + rng.normal(0, 0.3, len(chem))
        s = DarwinSearch(D=D, r=r, fold=fold, alphabet=ALPHA, confirm_fold=4, groups=chem, n_perm=2000)
        det = s.confirm_detail(program_key([("div", "x", 2)], D))
        false_orig += int(det["confirm"] > s.threshold)
        false_new += int(det["confirm"] > s.threshold and det["p_signflip"] < s.p_one)
    assert false_new == 0
    assert false_orig >= 3          # 5/12 with these seeds: the row-level rule alone would "discover" noise


def test_signflip_null_matches_bruteforce():
    df, D, r, fold, chem = synth(n_chem=20, rows=30, seed=2)
    s = DarwinSearch(D=D, r=r, fold=fold, alphabet=ALPHA, confirm_fold=4, groups=chem, n_perm=50, perm_seed=1)
    key = program_key([("cat", "feat")], D)
    det = s.confirm_detail(key)
    ho = fold == 4
    g = pd.factorize(chem[ho])[0]
    brute = [np.corrcoef(det["est"], r[ho] * s._S[b][g])[0, 1] for b in range(50)]
    assert np.allclose(brute, det["corr_b"], atol=1e-10)


# ------------------------------------------------------------------ outputs (skip when data are absent)
RESID = ROOT / "data/processed/audit/residuals_v1.parquet"
AUDIT = REPO / "results/r10_residual_audit.json"
VOI = REPO / "results/r10_viability_voi.json"


@pytest.mark.skipif(not RESID.exists() or not (REPO / "results/trajectory_cv.json").exists(), reason="data absent")
def test_residuals_reproduce_trajectory_cv():
    df = pd.read_parquet(RESID, columns=["chem", "k", "design", "resid", "fold"])
    ref = json.loads((REPO / "results/trajectory_cv.json").read_text(encoding="utf-8"))["by_k_folds_1to4"]
    assert set(df.fold.unique()) == {1, 2, 3, 4}
    for k in (1, 2, 3, 4):
        d = df[df.k == k].assign(a=lambda x: np.abs(x.resid.astype(float)))
        v = d.groupby(["chem", "k", "design"], observed=True).a.mean().groupby(level="chem", observed=True).mean().mean()
        assert abs(v - ref[str(k)]["neurotrajectory"]) < 1e-3


@pytest.mark.skipif(not AUDIT.exists(), reason="audit not run")
def test_audit_json_consistent():
    a = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert a["integrity_vs_trajectory_cv"]["all_match_1e-3"]
    assert "v1" in a["models"]
    if "primary" in a:
        s4 = a["search"]["4"]
        assert a["primary"]["n_confirmed_programs_confirm_fold4"] == s4["n_confirmed"] == len(s4["confirmed"])
        for name in a["protocol"]["plants"]:
            p = a["power"][name]
            m = p["mde_power0.8_exact"]
            assert a["primary"]["mde_power0.8_vehicle_sd_search_exact"][name] == m
            pw = [p[str(e)]["power_exact"] for e in a["protocol"]["plant_effects_vehicle_sd"]]
            if m["grid"] is not None:
                assert pw[a["protocol"]["plant_effects_vehicle_sd"].index(m["grid"])] >= 0.8
            else:
                assert max(pw) < 0.8


@pytest.mark.skipif(not VOI.exists(), reason="viability VOI not run")
def test_voi_json_labelled():
    v = json.loads(VOI.read_text(encoding="utf-8"))
    assert v["scenario_label"] == "prior viability screening scenario"
    assert "same NFA" in v["leakage_caveat"] and "not an independent experiment" in v["ntp_label"]
    assert v["ntp_subset"]["n_chemicals"] == len(v["ntp_subset"]["chemicals"])
    for k in ("1", "2", "3", "4"):
        e = v["all_chemicals"]["V1_vs_V0"][k]
        assert e["ci95"][0] <= e["mean_diff"] <= e["ci95"][1]
