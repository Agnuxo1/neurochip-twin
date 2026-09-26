"""NeuroChip Twin v2 - CPU demo (no network, < 1 min).

    python demo.py                              # default held-out chemical, 3 measured concentrations
    python demo.py --chemical "Deltamethrin" --k 2
    python demo.py --list                       # chemicals available in the bundle

For the chosen chemical the demo loads the NeuroTrajectory models trained WITHOUT that chemical
(its cross-validation fold), shows the model k measured concentrations, forecasts the full
dose x developmental-time trajectory (17 MEA network features at DIV 5/7/9/12) with a 90 %
predictive band, and then reveals the real held-out wells and scores the forecast against the
same baselines used in the paper. Outputs go to outputs/demo/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from neurotwin.bundle import BUNDLE, load_tasks  # noqa: E402
from neurotwin.models.trajectory import (  # noqa: E402
    DIVS, FEATURES, AnalogKNN, level_means, predict_interp, split_context)

SHOW = ["firing_rate_mean", "network_spike_number", "active_electrodes_number", "correlation_coefficient_mean"]


def load_fold_models(fold: int):
    from neurotwin.models.cnp import Z90, load_fold_models as _load
    models, cfg = _load(fold, "cpu", BUNDLE / "models")
    if not models:
        raise SystemExit(f"no models for fold {fold} in {BUNDLE/'models'}; run scripts/run_trajectory_cv.py "
                         "and scripts/build_bundle.py")
    return models, Z90[cfg["likelihood"]]


def spread_design(levels: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], np.float32)
    idx = np.unique(np.round(np.linspace(0, len(levels) - 1, k)).astype(int))
    return levels[idx]


def forecast(models, task, is_ctx, q):
    from neurotwin.models.cnp import predict
    preds = [predict(m, task, is_ctx, q, device="cpu") for m in models]
    mu = np.mean([p[0] for p in preds], 0)
    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + np.var([p[0] for p in preds], 0))
    return mu, sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chemical", default="Deltamethrin")
    ap.add_argument("--k", type=int, default=3, help="number of measured concentrations given to the model")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(REPO / "outputs/demo"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    tasks = load_tasks()
    if a.list:
        for t in tasks:
            print(f"{t.chem:45s} fold={t.fold} levels={len(t.levels)} label={t.label}")
        return
    by = {t.chem.lower(): t for t in tasks}
    if a.chemical.lower() not in by:
        raise SystemExit(f"unknown chemical '{a.chemical}'; use --list")
    task = by[a.chemical.lower()]
    models, z90 = load_fold_models(task.fold)
    ctx_lv = spread_design(task.levels, min(a.k, len(task.levels) - 1))
    is_ctx, is_tgt = split_context(task, ctx_lv)
    held = np.unique(task.logc[is_tgt])
    mu_h, sc_h = forecast(models, task, is_ctx, held)
    lv, obs, ok = level_means(task, is_tgt)
    curve_mae = float(np.abs(mu_h - obs)[ok].mean())
    interp_mae = float(np.abs(predict_interp(task, is_ctx, held) - obs)[ok].mean())
    train = [t for t in tasks if t.fold != task.fold]
    knn_mae = float(np.abs(AnalogKNN(train).predict(task, is_ctx, held) - obs)[ok].mean())
    idx = np.searchsorted(held, task.logc[is_tgt])
    inside = (np.abs(task.y[is_tgt] - mu_h[idx]) <= z90 * sc_h[idx])[task.m[is_tgt]]
    summary = {
        "chemical": task.chem, "cv_fold": task.fold, "epa_dnt_reference_label": task.label,
        "model_never_saw_this_chemical": True,
        "measured_concentrations_uM": [round(float(10 ** x), 4) for x in ctx_lv],
        "held_out_concentrations_uM": [round(float(10 ** x), 4) for x in held],
        "curve_mae": {"neurotrajectory": round(curve_mae, 3), "loglinear_interp": round(interp_mae, 3),
                      "analog_knn": round(knn_mae, 3)},
        "held_out_wells_inside_90pct_band": round(float(inside.mean()), 3),
        "runtime_s": None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_plots:
        grid = np.linspace(task.levels.min() - 0.3, task.levels.max() + 0.3, 60).astype(np.float32)
        mu_g, sc_g = forecast(models, task, is_ctx, grid)
        _plot(task, is_ctx, is_tgt, grid, mu_g, sc_g, z90, out / "forecast_curves.png")
        _plot_traj(task, is_tgt, held, mu_h, obs, ok, out / "trajectory_heatmap.png")
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _plot(task, is_ctx, is_tgt, grid, mu, sc, z90, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(SHOW), len(DIVS), figsize=(13, 9), sharex=True)
    for i, f in enumerate(SHOW):
        j = FEATURES.index(f)
        for d, div in enumerate(DIVS):
            ax = axes[i, d]
            ax.fill_between(grid, mu[:, d, j] - z90 * sc[:, d, j], mu[:, d, j] + z90 * sc[:, d, j],
                            color="#8fb3e8", alpha=0.45, lw=0, label="90% predictive band")
            ax.plot(grid, mu[:, d, j], color="#1f4e9c", lw=1.8, label="forecast")
            for sel, col, lab in ((is_ctx, "#222222", "measured (given)"), (is_tgt, "#d1495b", "held out (revealed)")):
                mm = task.m[sel][:, d, j]
                ax.scatter(task.logc[sel][mm], task.y[sel][mm, d, j], s=14, color=col, alpha=0.8, label=lab, zorder=3)
            ax.axhline(0, color="#999999", lw=0.8, ls=":")
            if i == 0:
                ax.set_title(f"DIV {div}")
            if d == 0:
                ax.set_ylabel(f.replace("_", " "), fontsize=8)
            if i == len(SHOW) - 1:
                ax.set_xlabel("log10 concentration (µM)")
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"{task.chem}: forecast from measured concentrations vs held-out wells "
                 "(units: vehicle robust SD; model never saw this chemical)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_traj(task, is_tgt, held, mu_h, obs, ok, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = len(held) - 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    vmax = 6
    for ax, M, title in ((axes[0], mu_h[top].T, "forecast"), (axes[1], np.where(ok[top], obs[top], np.nan).T, "observed (held out)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DIVS)), [f"DIV{d}" for d in DIVS])
        ax.set_title(title)
    axes[0].set_yticks(range(len(FEATURES)), [f.replace("_", " ") for f in FEATURES], fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.7, label="effect vs vehicle (robust SD)")
    fig.suptitle(f"{task.chem} at {10 ** held[top]:.3g} µM: developmental trajectory of 17 network features")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
