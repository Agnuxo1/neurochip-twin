"""Synthetic-data tests for the trajectory models (run in CI without any downloaded data)."""
import numpy as np
import pytest

from neurotwin.models.trajectory import (ND, NF, AnalogKNN, ChemTask, interp_rows, level_means,
                                         predict_hill, predict_interp, split_context)


def make_task(chem="c", fold=0, ac50=0.5, top=-6.0, h=1.5, reps=3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    levels = np.log10([0.03, 0.1, 0.3, 1, 3, 10, 30]).astype(np.float32)
    logc = np.repeat(levels, reps)
    resp = top / (1 + 10 ** (h * (ac50 - logc)))
    y = np.repeat(resp[:, None, None], ND, 1).repeat(NF, 2).astype(np.float32)
    y += noise * rng.standard_normal(y.shape).astype(np.float32)
    m = np.ones_like(y, bool)
    return ChemTask(chem, fold, logc, y, m, np.array(["p|d"] * len(logc)))


def test_level_means_cache_matches_direct():
    t = make_task(noise=0.3)
    sel = np.isin(t.logc, t.levels[[0, 3, 6]])
    lv, mu, ok = level_means(t, sel)
    for i, l in enumerate(lv):
        assert np.allclose(mu[i], t.y[t.logc == l].mean(0), atol=1e-5)
    assert ok.all()


def test_interp_rows_linear_and_flat_outside():
    lv = np.array([0.0, 1.0])
    mu = np.stack([np.zeros((ND, NF)), np.ones((ND, NF))]).astype(np.float32)
    out = interp_rows(lv, mu, np.ones_like(mu, bool), np.array([-1.0, 0.25, 2.0]))
    assert np.allclose(out[:, 0, 0], [0.0, 0.25, 1.0])


def test_hill_baseline_recovers_known_curve():
    t = make_task(ac50=0.4, top=-5.0, h=1.5)
    ic, it = split_context(t, t.levels[[0, 2, 4, 6]])
    pred = predict_hill(t, ic, t.logc[it])
    assert np.abs(pred - t.y[it]).mean() < 0.35


def test_interp_beats_nothing_on_monotone_curve():
    t = make_task(noise=0.1)
    ic, it = split_context(t, t.levels[[0, 3, 6]])
    e_interp = np.abs(predict_interp(t, ic, t.logc[it]) - t.y[it]).mean()
    assert e_interp < np.abs(t.y[it]).mean()


def test_analog_knn_prefers_similar_chemicals():
    train = [make_task(f"a{i}", 1, ac50=0.5, top=-6) for i in range(5)] + \
            [make_task(f"b{i}", 1, ac50=-2.0, top=4) for i in range(5)]
    test = make_task("q", 0, ac50=0.5, top=-6)
    ic, it = split_context(test, test.levels[[0, 6]])
    pred = AnalogKNN(train, k=5).predict(test, ic, test.logc[it])
    assert np.abs(pred - test.y[it]).mean() < 0.5


@pytest.mark.parametrize("use_interp,use_attention", [(False, True), (True, True), (True, False)])
def test_cnp_shapes_and_zero_context(use_interp, use_attention):
    torch = pytest.importorskip("torch")
    from neurotwin.models.cnp import TrainConfig, predict, train_cnp
    tasks = [make_task(f"c{i}", 1, ac50=0.2 * i, noise=0.2, seed=i) for i in range(6)]
    m = train_cnp(tasks, TrainConfig(steps=5, batch=4, use_interp=use_interp, use_attention=use_attention, n_freq=3),
                  device="cpu")
    t = make_task("q", 0)
    q = np.linspace(-2, 2, 7).astype(np.float32)
    for ctx in (np.zeros(len(t.logc), bool), np.isin(t.logc, t.levels[[1, 4]])):
        mu, sc = predict(m, t, ctx, q, device="cpu")
        assert mu.shape == (7, ND, NF) and sc.shape == (7, ND, NF)
        assert np.isfinite(mu).all() and (sc > 0).all()


def test_bundle_roundtrip(tmp_path):
    from neurotwin.bundle import load_tasks, save_tasks
    tasks = [make_task("x", 2, noise=0.5), make_task("y", 3, noise=0.5, seed=1)]
    p = tmp_path / "b.npz"
    save_tasks(tasks, p)
    back = load_tasks(p)
    assert [b.chem for b in back] == ["x", "y"] and back[1].fold == 3
    assert np.allclose(back[0].y, tasks[0].y, atol=1e-2) and (back[0].m == tasks[0].m).all()
