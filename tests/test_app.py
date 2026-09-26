"""Tests for the Gradio app's compute core (no server is launched).

Skipped when gradio is not installed. Runs on CPU from the shipped data_bundle.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")

REPO = Path(__file__).resolve().parents[1]
CHEM = "Deltamethrin"


@pytest.fixture(scope="module")
def nt():
    spec = importlib.util.spec_from_file_location("neurochip_app", REPO / "app" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fc(nt):
    return nt.run_forecast(CHEM, 2, "spread")


@pytest.fixture(scope="module")
def rev(nt, fc):
    return nt.reveal(fc)


def _results(name):
    return json.loads((REPO / "results" / name).read_text(encoding="utf-8"))


def test_forecast_shapes_and_keys(nt, fc):
    from neurotwin.models.trajectory import DIVS, FEATURES
    task = nt.get_task(CHEM)
    for key in ("chemical", "fold", "k", "context_logc", "held_logc", "grid_logc", "mu_grid", "scale_grid",
                "mu_held", "scale_held", "band_q", "band_method", "epistemic_score", "abstention",
                "model_never_saw_this_chemical"):
        assert key in fc, key
    assert fc["fold"] == task.fold and fc["model_never_saw_this_chemical"] is True
    assert fc["k"] == 2 and len(fc["context_logc"]) == 2
    H, G = len(fc["held_logc"]), len(fc["grid_logc"])
    assert fc["mu_held"].shape == fc["scale_held"].shape == (H, len(DIVS), len(FEATURES))
    assert fc["mu_grid"].shape == fc["scale_grid"].shape == (G, len(DIVS), len(FEATURES))
    assert H + 2 == len(task.levels)
    assert np.all(fc["scale_held"] > 0) and np.isfinite(fc["mu_grid"]).all()
    assert math.isfinite(fc["epistemic_score"]) and fc["epistemic_score"] > 0 and fc["band_q"] > 0
    ab = fc["abstention"]
    assert set(ab) >= {"available", "score", "threshold", "abstain", "rule", "note"}
    assert (ab["abstain"] is None) == (not ab["available"])


def test_same_numbers_as_demo_engine(nt, fc):
    """The app's ensemble must equal demo.forecast (the engine reused by the command-line demo)."""
    import demo
    task = nt.get_task(CHEM)
    models, _ = nt.get_models(task.fold)
    is_ctx, _ = nt.split_context(task, fc["context_logc"])
    mu, sc = demo.forecast(models, task, is_ctx, fc["held_logc"])
    np.testing.assert_allclose(fc["mu_held"], mu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fc["scale_held"], sc, rtol=1e-5, atol=1e-6)


def test_reveal_metrics(nt, fc, rev):
    assert set(rev["metrics"]) == {"neurotrajectory", "loglinear_interp", "analog_knn"}
    for m in rev["metrics"].values():
        assert math.isfinite(m["curve_mae"]) and m["curve_mae"] >= 0 and len(m["by_div"]) == 4
    assert 0.0 <= rev["band_coverage"] <= 1.0 and rev["n_band_entries"] > 0
    assert rev["obs_held"].shape == fc["mu_held"].shape and rev["ok_held"].dtype == bool
    ref = rev["cv_reference"]
    tcv = _results("trajectory_cv.json")
    assert ref["curve_mae"]["neurotrajectory"] == tcv["by_k"]["2"]["curve_mae"]["mean"]["neurotrajectory"]
    table = nt.metrics_table(rev)
    assert list(table["Method"]) == [nt.METHOD_LABELS[m] for m in ("neurotrajectory", "loglinear_interp", "analog_knn")]


def test_custom_design(nt):
    task = nt.get_task(CHEM)
    fc1 = nt.run_forecast(CHEM, 1, "custom", ["0"])
    assert np.isclose(fc1["context_logc"][0], task.levels[0]) and fc1["k"] == 1
    with pytest.raises(ValueError):
        nt.run_forecast(CHEM, 1, "custom", [])
    with pytest.raises(ValueError):
        nt.run_forecast(CHEM, 1, "custom", [str(i) for i in range(len(task.levels))])
    with pytest.raises(ValueError):
        nt.get_task("not a chemical")


def test_figures(nt, fc, rev):
    from matplotlib.figure import Figure
    f1 = nt.forecast_figure(fc, None, ["firing_rate_mean", "burst_rate"])
    assert isinstance(f1, Figure) and len(f1.axes) == 2 * 4
    f2 = nt.forecast_figure(fc, rev)
    assert len(f2.axes) == len(nt.DEFAULT_FEATURES) * 4
    h = nt.trajectory_figure(fc, rev, 0)
    assert len(h.axes) == 3 + 1          # forecast | observed | error + colour bar
    assert nt.results_figure() is not None


def test_narrow_figures_for_phone_width(nt, fc, rev):
    """Phone layout: same panels, 2 DIV columns per feature, heatmaps stacked, narrower figures."""
    feats = ["firing_rate_mean", "burst_rate", "network_spike_number"]
    wide, narrow = nt.forecast_figure(fc, rev, feats), nt.forecast_figure(fc, rev, feats, narrow=True)
    assert len(narrow.axes) == len(wide.axes) == len(feats) * 4
    assert narrow.get_size_inches()[0] < 0.6 * wide.get_size_inches()[0]
    h = nt.trajectory_figure(fc, rev, None, narrow=True)
    assert len(h.axes) == 3 + 1 and h.get_size_inches()[1] > h.get_size_inches()[0]
    assert len(nt.results_figure(narrow=True).axes) == 2
    assert nt.NARROW_PX > 400 and "window.innerWidth" in nt.VW_JS


def test_results_tables_read_from_json(nt):
    tcv, conf = _results("trajectory_cv.json"), _results("conformal_r7.json")
    r2 = nt.r2_table()
    assert len(r2) == len(tcv["by_k"])
    for _, row in r2.iterrows():
        assert row["NeuroTrajectory"] == tcv["by_k"][str(row["k measured"])]["curve_mae"]["mean"]["neurotrajectory"]
    ct = nt.conformal_table()
    assert len(ct) == sum(1 for e in conf["results"].values() for k in e if k.startswith("k"))
    ab = nt.abstention_table()
    assert ab.iloc[0]["Curve MAE retained"] == conf["abstention"]["curve_mae_retained"]
    epa = _results("epa_baseline_repro.json")
    assert len(nt.epa_table()) == len(epa["models"])
    r8 = _results("r8_chiplayer.json")
    assert r8["source"] in nt.chip_markdown()
    man = json.loads((REPO / "data_bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert man["files"]["nfa_tasks.npz"]["licence"] in nt.about_markdown()
    assert all(Path(p).exists() for p, _ in nt.chip_images())


def test_aggregate_calibration_synthetic(nt):
    rng = np.random.default_rng(0)
    recs = [{"chemical": f"c{f}_{i}", "fold": f, "k": k, "design": 0, "epi": float(rng.uniform()),
             "curve_mae": float(rng.uniform()), "scores": np.abs(rng.normal(size=50)).astype(np.float32)}
            for f in range(3) for i in range(20) for k in (1, 2)]
    out = nt.aggregate_calibration(recs, alpha=0.1, pct=90)
    epi = np.array([r["epi"] for r in recs])
    fo = np.array([r["fold"] for r in recs])
    for g in range(3):
        assert np.isclose(out["epistemic_threshold_by_fold"][str(g)], np.percentile(epi[fo != g], 90))
        for k in ("1", "2"):
            q = out["conformal_qhat_by_fold"][str(g)][k]
            assert 1.0 < q < 3.0                      # ~ 90th percentile of |N(0,1)| = 1.645
    rep = out["reproduction"]
    assert 0 < rep["abstention_rate"] < 0.3 and set(rep["by_k"]) == {"1", "2"}


def test_threshold_loader_missing_and_stale(nt, tmp_path):
    stale = tmp_path / nt.THRESHOLD_FILE
    stale.write_text(json.dumps({"bundle_fingerprint": "0" * 64, "epistemic_threshold_by_fold": {"0": 0.1}}))
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(nt, "threshold_paths", lambda: [tmp_path / "missing.json", stale])
            assert nt.load_thresholds(refresh=True) is None
            assert "different bundle" in nt.threshold_status()
            ab = nt.abstention(0.5, 0, 2)
            assert ab["available"] is False and ab["abstain"] is None
            q, how = nt.band_multiplier(0, 2, 1.6449)
            assert q == 1.6449 and how.startswith("parametric")
    finally:
        nt.load_thresholds(refresh=True)


def test_background_threshold_build(nt):
    """Server start builds missing thresholds once, in a thread, and reports failures without crashing."""
    import threading
    release, calls = threading.Event(), []

    def fake_build(workers=None):
        calls.append(workers)
        release.wait(10)
        if len(calls) > 1:
            raise RuntimeError("boom")
        return {"build_seconds": 0.0}

    def wait_idle():
        for th in threading.enumerate():
            if th.name == "abstention-thresholds":
                th.join(10)
        assert nt._THR["building"] is False

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(nt, "load_thresholds", lambda refresh=False: None)
            mp.setattr(nt, "build_abstention_thresholds", fake_build)
            assert nt.start_threshold_build() is True
            assert nt._THR["building"] is True and "background" in nt._THR["status"]
            assert nt.start_threshold_build() is False          # one build at a time
            release.set()
            wait_idle()
            assert nt.start_threshold_build() is True          # second (failing) build
            wait_idle()
            assert "automatic build failed" in nt._THR["status"] and len(calls) == 2
    finally:
        nt._THR["building"] = False
        nt.load_thresholds(refresh=True)
    with pytest.MonkeyPatch.context() as mp:                   # a valid file is present: nothing to do
        mp.setattr(nt, "load_thresholds", lambda refresh=False: {"epistemic_threshold_by_fold": {}})
        assert nt.start_threshold_build() is False


def test_build_app_without_launch(nt):
    import gradio as gr
    app = nt.build_app()
    assert isinstance(app, gr.Blocks)
