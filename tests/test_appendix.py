"""Tests for the appendix 'Own prior architectures, evaluated on real data'
(scripts/appendix_own_architectures.py -> results/appendix_own_arch.json).

Synthetic tests run everywhere; tests that need the generated JSON, its part files or the Brewer data
skip when those are absent.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))
sys.path.insert(0, str(ROOT / "repo" / "scripts"))
RESULTS = ROOT / "repo" / "results"
OUT = RESULTS / "appendix_own_arch.json"
PARTS = RESULTS / "appendix_own_arch_parts"
BREWER = ROOT / "data" / "processed" / "brewer" / "wells.parquet"

import appendix_own_architectures as AP  # noqa: E402

ARCH_KEYS = ("A_eikonal_latency", "B_qesn_chip_lattice", "C_doselattice_encoder")
ALLOWED_VERDICTS = set(AP.VERDICT_WORDS) | {"mixed"}


def _load():
    if not OUT.exists():
        pytest.skip("results/appendix_own_arch.json not generated")
    return json.loads(OUT.read_text(encoding="utf-8"))


def _numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


# ------------------------------------------------------------------ synthetic
def test_crank_nicolson_step_is_unitary():
    assert abs(AP.b_unitarity_check(c=2.0) - 1.0) < 1e-10
    assert abs(AP.b_unitarity_check(c=0.5, seed=3) - 1.0) < 1e-10


def test_chip_masks_have_twenty_tunnels():
    walls, chip = AP.b_lattice_mask("walls"), AP.b_lattice_mask("chip")
    extra = chip & ~walls
    assert walls.sum() == 4 * (5 * AP.B_S) ** 2
    assert extra.sum() == 20 * 2 * AP.B_S          # 20 one-cell tunnels spanning 2 electrode pitches
    m, tun = AP.a_mask("chip")
    assert tun.sum() == 20 * 2 * AP.A_S


def test_max_rel_diff():
    assert AP.max_rel_diff({"a": [1.0, 2]}, {"a": [1.0, 2]}) == 0.0
    assert math.isclose(AP.max_rel_diff({"a": 1.0 + 1e-12}, {"a": 1.0}), 1e-12, rel_tol=1e-3)
    assert AP.max_rel_diff({"a": 1.0}, {"b": 1.0}) == float("inf")
    assert AP.max_rel_diff({"a": "x"}, {"a": "y"}) == float("inf")


def _fake_rows(n=9, seed=0, lattice_gain=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"fid": i + 1}
        for h in AP.B_HORIZONS:
            base = float(rng.uniform(0.02, 0.2))
            d = {"persistence": base * 1.4, "var_ridge": base, "esn_nonspatial": base * float(rng.normal(1, 0.02))}
            for m in ("lattice_chip", "lattice_full", "lattice_walls", "lattice_chip_shuffled"):
                d[m] = base * (1 - lattice_gain) * float(rng.normal(1, 0.01))
            row[f"h{h}"] = d
        rows.append(row)
    return rows


def test_b_summary_and_verdicts_on_synthetic():
    tie = AP.b_summary(_fake_rows())
    assert len(tie) == 2 * (5 + 4 + 5)           # per horizon: vs VAR (5 models), vs ESN (4), vs persistence (5)
    for v in tie.values():
        assert v["n"] == 9 and 0 <= v["n_better"] <= 9 and v["ci95"][0] <= v["ci95"][1]
    assert tie["h1|lattice_chip_vs_persistence"]["n_better"] == 9
    assert AP.verdict_b(tie) == "matches but does not beat"
    assert AP.verdict_b(AP.b_summary(_fake_rows(lattice_gain=0.3))) == "better"
    rows = _fake_rows()
    for r in rows:                              # ablations identical to the chip geometry
        for h in AP.B_HORIZONS:
            for x in ("lattice_full", "lattice_walls", "lattice_chip_shuffled"):
                r[f"h{h}"][x] = r[f"h{h}"]["lattice_chip"]
    abl = AP.b_ablation_summary(rows)
    for x in ("lattice_full", "lattice_walls", "lattice_chip_shuffled"):
        s = abl[f"h1|lattice_chip_vs_{x}"]
        assert s["n_within_1pct"] == 9 and s["mean_rel_pct"] == 0.0 and s["n_better"] == 0


def test_verdict_rules_a_and_c():
    mk = lambda lo, hi: {"chip_eik_c_minus_idw_comp_ms": {"mean": (lo + hi) / 2, "ci95": [lo, hi]}}  # noqa: E731
    assert AP.verdict_a(mk(-0.2, 0.6)) == "no gain over interpolation"
    assert AP.verdict_a(mk(0.1, 0.6)) == "worse"
    assert AP.verdict_a(mk(-0.6, -0.1)) == "better"
    ck = lambda cis: {str(k): {"doselattice_vs_neurotrajectory_v1_saved": {"ci95": c}} for k, c in zip(AP.C_KS, cis)}  # noqa: E731
    assert AP.verdict_c(ck([[0.01, 0.1]] * 4)) == "worse"
    assert AP.verdict_c(ck([[-0.01, 0.1]] * 3 + [[0.01, 0.1]])) == "no gain; worse at some k"
    assert AP.verdict_c(ck([[-0.1, -0.01]] * 4)) == "better"
    assert AP.verdict_c(ck([[-0.01, 0.1]] * 4)) == "matches but does not beat"
    assert AP.verdict_c(ck([[-0.1, -0.01]] + [[0.01, 0.1]] * 3)) == "mixed"


def test_doselattice_stencil_conserves_mass_and_matches_parent_shapes():
    torch = pytest.importorskip("torch")
    cls = AP.make_doselattice_class()
    from neurotwin.models.cnp import D_OUT
    torch.manual_seed(0)
    m = cls(use_interp=True, n_freq=8).eval()
    B, N, Q = 2, 5, 7
    cx = torch.linspace(-3, 2, N).repeat(B, 1).unsqueeze(-1)
    cy, cm = torch.randn(B, N, D_OUT), torch.ones(B, N, D_OUT)
    cmask = torch.ones(B, N, dtype=torch.bool)
    qx = torch.linspace(-3.4, 2.4, Q).repeat(B, 1).unsqueeze(-1)
    qi = torch.zeros(B, Q, D_OUT + 1)
    with torch.no_grad():
        mu, sig = m(cx, cy, cm, cmask, qx, qi)
    assert mu.shape == (B, Q, D_OUT) and sig.shape == (B, Q, D_OUT) and bool((sig > 0).all())
    # the 3-point stencil with reflecting ends conserves the total mass
    psi = torch.rand(3, 4, 64)
    tot = psi.sum(-1)
    for _ in range(32):
        pad = torch.cat([psi[..., :1], psi, psi[..., -1:]], -1)
        psi = 0.5 * psi + 0.25 * (pad[..., :-2] + pad[..., 2:])
    assert torch.allclose(psi.sum(-1), tot, rtol=1e-5)


# ------------------------------------------------------------------ generated JSON: schema
def test_schema_and_required_fields():
    d = _load()
    for k in ("title", "status", "reference_models", "summary_table", "architectures", "verdict_vocabulary"):
        assert k in d
    assert "v1" in d["reference_models"]
    assert set(d["architectures"]) == set(ARCH_KEYS)
    for name, a in d["architectures"].items():
        for f in ("architecture", "source", "what_was_tested", "data", "split", "baselines", "metric", "result",
                  "verdict", "verdict_detail", "provenance"):
            assert f in a, (name, f)
        assert a["source"]["repo"].startswith("Agnuxo1/") and a["source"]["licence"] in ("MIT", "Apache-2.0")
        assert a["source"]["author"] == "Francisco Angulo de Lafuente"
        assert isinstance(a["baselines"], list) and len(a["baselines"]) >= 3
        assert a["verdict"] in ALLOWED_VERDICTS
        assert "\n" not in a["verdict"] and len(a["verdict"]) < 40
    assert d["architectures"]["A_eikonal_latency"]["source"]["licence"] == "MIT"
    assert d["architectures"]["B_qesn_chip_lattice"]["source"]["licence"] == "Apache-2.0"


def test_all_numbers_finite():
    d = _load()
    bad = [p for p, v in _numbers(d) if not math.isfinite(v)]
    assert not bad, bad[:10]


def test_language_rules():
    d = _load()
    # only the three evaluated source repositories are cited, and no URLs
    assert {a["source"]["repo"] for a in d["architectures"].values()} == {AP.SOURCES[k]["repo"] for k in "ABC"}
    assert "http" not in json.dumps(d).lower()
    for a in d["architectures"].values():
        prose = " ".join([a["architecture"], a["what_was_tested"], a["verdict"], a["verdict_detail"]]).lower()
        for word in ("quantum", "optical", "neuromorphic", "holographic", "consciousness", "propagation velocity"):
            assert word not in prose, word
    assert "classical" in d["architectures"]["B_qesn_chip_lattice"]["architecture"].lower()


# ------------------------------------------------------------------ generated JSON: coherence
def test_verdicts_follow_the_numbers():
    d = _load()
    A, B, C = (d["architectures"][k] for k in ARCH_KEYS)
    assert AP.verdict_a(A["result"]) == A["verdict"]
    assert AP.verdict_b(B["result"]["all_contrasts"]) == B["verdict"]
    assert AP.verdict_c(C["result"]["by_k"]) == C["verdict"]
    prim = A["result"]["chip_eik_c_minus_idw_comp_ms"]
    assert A["result"]["preregistered_success"]["met"] == bool(
        prim["ci95"][1] < 0 and A["result"]["chip_eik_c_minus_free_eik_ms"]["mean"] < 0)


def test_contrasts_are_internally_consistent():
    d = _load()
    A, B, C = (d["architectures"][k] for k in ARCH_KEYS)
    for k in ("chip_eik_c_minus_idw_comp_ms", "chip_eik_c_minus_idw_ms", "chip_eik_c_minus_free_eik_ms",
              "chip_eik_c_minus_comp_mean_ms"):
        c = A["result"][k]
        assert c["ci95"][0] <= c["mean"] <= c["ci95"][1] and c["n"] == 9 and 0 <= c["recordings_improved"] <= 9
    mae = A["result"]["mean_mae_ms"]
    assert math.isclose(mae["chip_eik_c"] - mae["idw_comp"], A["result"]["chip_eik_c_minus_idw_comp_ms"]["mean"], abs_tol=1e-9)
    for k, c in B["result"]["all_contrasts"].items():
        assert c["n"] == 9 and c["ci95"][0] <= c["ci95"][1], k
    for k in AP.C_KS:
        e = C["result"]["by_k"][str(k)]
        assert e["n_chemicals"] == 49
        for key in ("doselattice_vs_neurotrajectory_v1_saved", "doselattice_vs_neurotrajectory_retrained",
                    "neurotrajectory_retrained_vs_v1_saved", "doselattice_vs_best_baseline"):
            c = e[key]
            assert c["ci95"][0] <= c["mean_diff"] <= c["ci95"][1] and c["n_chemicals"] == 49
            assert c["ci95_rel_pct"][0] <= c["rel_change_pct"] <= c["ci95_rel_pct"][1]
        m = e["mean_curve_mae"]
        rel = 100 * (m["doselattice"] - m["neurotrajectory_v1_saved"]) / m["neurotrajectory_v1_saved"]
        assert abs(rel - e["doselattice_vs_neurotrajectory_v1_saved"]["rel_change_pct"]) < 0.05
        assert e["best_baseline"] in ("zero", "context_mean", "loglinear_interp", "hill_per_endpoint", "analog_knn")


def test_summary_table_matches_details():
    d = _load()
    A, B, C = (d["architectures"][k] for k in ARCH_KEYS)
    rows = {(r["architecture"], r["comparison"]): r for r in d["summary_table"]}
    a = rows[("A_eikonal_latency", "chip_eik_c minus idw_comp")]
    prim = A["result"]["chip_eik_c_minus_idw_comp_ms"]
    assert a["effect_ms"] == round(prim["mean"], 2) and a["ci95_ms"] == [round(x, 2) for x in prim["ci95"]]
    assert a["n_better"] == prim["recordings_improved"] and a["verdict"] == A["verdict"]
    for h in AP.B_HORIZONS:
        for ref in ("var_ridge", "esn_nonspatial"):
            r, s = rows[("B_qesn_chip_lattice", f"h{h}|lattice_chip_vs_{ref}")], B["result"]["all_contrasts"][f"h{h}|lattice_chip_vs_{ref}"]
            assert (r["effect_pct"], r["ci95_pct"], r["n_better"]) == (s["mean_rel_pct"], s["ci95"], s["n_better"])
    for k in AP.C_KS:
        r = rows[("C_doselattice_encoder", f"k={k}|doselattice_vs_neurotrajectory_v1_saved")]
        s = C["result"]["by_k"][str(k)]["doselattice_vs_neurotrajectory_v1_saved"]
        assert (r["effect_pct"], r["ci95_pct"], r["effect_abs"], r["ci95_abs"]) == (s["rel_change_pct"], s["ci95_rel_pct"], s["mean_diff"], s["ci95"])
        assert r["n_better"] == s["n_chem_improved"] and r["verdict"] == C["verdict"]


def test_c_is_labelled_development_fold_v1():
    d = _load()
    C = d["architectures"]["C_doselattice_encoder"]
    assert "DEVELOPMENT FOLD 0 ONLY" in C["data"] and "fold 0" in C["label"]
    cfg = C["provenance"]["config"]
    assert (cfg["likelihood"], cfg["use_interp"], cfg["n_freq"], cfg["steps"]) == ("laplace", True, 8, 4000)
    assert cfg["seeds"] == [0, 1, 2]
    integ = C["result"]["integrity_vs_trajectory_cv_csv"]
    if integ:                                  # saved v1 ensemble and baselines reproduce the v1 CV log
        for v in integ.values():
            assert v["n_matched"] == 196 and v["max_abs_diff_vs_trajectory_cv_csv"] < 1e-6
    fid = C["result"]["port_fidelity"]
    if fid.get("available"):
        assert fid["identical_parameters_after_seeded_init"] and fid["identical_forward_outputs"]


def test_pilot_reproduction_checks():
    d = _load()
    A, B = d["architectures"]["A_eikonal_latency"], d["architectures"]["B_qesn_chip_lattice"]
    assert all(A["provenance"]["pilot_byte_identical"].values())
    pv = B["provenance"]
    assert pv["summary_check"]["byte_identical"]
    if pv["mode"].startswith("reuse"):
        assert pv["reduced_check"] and all(c["byte_identical_excluding_elapsed_s"] for c in pv["reduced_check"])
    assert abs(B["result"]["crank_nicolson_norm_ratio"] - 1.0) < 1e-10


def test_b_summary_recomputes_from_part_file():
    d = _load()
    p = PARTS / "B.json"
    if not p.exists():
        pytest.skip("part file B.json absent")
    part = json.loads(p.read_text(encoding="utf-8"))
    assert len(part["per_recording"]) == 9
    assert AP.b_summary(part["per_recording"]) == d["architectures"]["B_qesn_chip_lattice"]["result"]["all_contrasts"]


# ------------------------------------------------------------------ data-dependent reproduction
def test_a_port_reproduces_json_on_brewer():
    d = _load()
    if not BREWER.exists():
        pytest.skip("Brewer data absent")
    import pandas as pd
    w = pd.read_parquet(BREWER)
    per, summ, _ = AP.a_heldout(w)
    ref = d["architectures"]["A_eikonal_latency"]["result"]
    assert AP.max_rel_diff(summ["mean_mae_ms"], ref["mean_mae_ms"]) < 1e-9
    assert AP.max_rel_diff(summ["chip_eik_c_minus_idw_comp_ms"], ref["chip_eik_c_minus_idw_comp_ms"]) < 1e-9
