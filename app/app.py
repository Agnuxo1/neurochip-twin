"""NeuroChip Twin v2 - interactive demo (Gradio, CPU only, offline).

    python app/app.py                      # serve on http://127.0.0.1:7860
    python app/app.py --smoke              # run the compute core for one chemical and print JSON (no server)
    python app/app.py --build-thresholds   # one-off: cross-fitted abstention thresholds and conformal 90 %
                                           # multipliers from the shipped bundle (the server also builds
                                           # them in the background when missing; --no-auto-thresholds)

Every chemical is forecast by the models of ITS cross-validation fold, which never saw it (held-out).
All numbers shown in the Results / Chip layer / About tabs are read at runtime from repo/results/*.json
and data_bundle/manifest.json; nothing is typed by hand. The compute core (run_forecast, reveal, the
figure builders and the results tables) is made of plain functions that tests call without a server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # offline: no telemetry calls

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parent
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import demo as engine  # noqa: E402  (reused engine: load_fold_models, spread_design, forecast, SHOW)
from neurotwin.bundle import BUNDLE, load_tasks, manifest  # noqa: E402
from neurotwin.models.trajectory import (  # noqa: E402
    DIVS, FEATURES, AnalogKNN, level_means, predict_interp, split_context)

DEFAULT_CHEMICAL = "Deltamethrin"
DEFAULT_K = 3
DEFAULT_FEATURES = list(engine.SHOW)
ALPHA = 0.1                      # 90 % predictive band
ABSTAIN_PERCENTILE = 90          # rule stated in results/conformal_r7.json (checked when thresholds are built)
THRESHOLD_FILE = "abstention_thresholds.json"
CACHE_DIR = Path(os.environ.get("NT_APP_CACHE", str(REPO / "outputs" / "app")))
HEAT_VMAX = 6.0                  # colour scale of the trajectory heatmaps (robust SD units), as in demo.py

# colour-blind-safe palette (Okabe-Ito): blue forecast, near-black measured, vermillion revealed
C_LINE, C_BAND, C_MEASURED, C_REVEALED, C_GRID = "#0072B2", "#56B4E9", "#1b1b1b", "#D55E00", "#9a9a9a"
METHOD_COLOURS = {"neurotrajectory": "#0072B2", "loglinear_interp": "#E69F00", "analog_knn": "#009E73",
                  "hill_per_endpoint": "#CC79A7", "context_mean": "#7f7f7f"}
METHOD_LABELS = {"neurotrajectory": "NeuroTrajectory (ours)", "loglinear_interp": "Log-linear interpolation",
                 "analog_knn": "Analog kNN", "hill_per_endpoint": "Hill per endpoint",
                 "context_mean": "Context mean", "zero": "Vehicle (zero effect)"}
UNITS = "effect vs vehicle (robust SD units; 0 = vehicle)"

_LOCK = threading.RLock()


# ====================================================================== data + models (cached)
@lru_cache(maxsize=1)
def _bundle():
    tasks = load_tasks()
    return tasks, {t.chem.lower(): t for t in tasks}


def chemical_names() -> list[str]:
    return sorted((t.chem for t in _bundle()[0]), key=str.lower)


def get_task(name: str):
    t = _bundle()[1].get(str(name).strip().lower())
    if t is None:
        raise ValueError(f"unknown chemical '{name}'")
    return t


@lru_cache(maxsize=None)
def _models_cached(fold: int):
    return engine.load_fold_models(int(fold))       # (models, z90) of the fold that never saw its chemicals


def get_models(fold: int):
    with _LOCK:
        return _models_cached(int(fold))


@lru_cache(maxsize=None)
def _knn_cached(fold: int):
    return AnalogKNN([t for t in _bundle()[0] if t.fold != fold])


def get_knn(fold: int):
    with _LOCK:
        return _knn_cached(int(fold))


@lru_cache(maxsize=None)
def _json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@lru_cache(maxsize=1)
def _per_chemical_cv():
    p = RESULTS / "trajectory_cv_per_chemical.csv"
    if not p.exists():
        return None
    import pandas as pd
    return pd.read_csv(p)


def k_choices() -> list[int]:
    """Numbers of measured concentrations evaluated in the CV (results/trajectory_cv.json 'by_k'), k >= 1."""
    tcv = _json("trajectory_cv.json") or {}
    ks = sorted(int(k) for k in tcv.get("by_k", {}) if int(k) >= 1)
    return ks or [1, 2, 3, 4]


def level_label(logc: float) -> str:
    return f"{10 ** float(logc):.3g} µM"


# ====================================================================== engine
def ensemble_predict(models, task, is_ctx, q):
    """Same ensemble as demo.forecast / scripts/run_conformal.py, also returning the member means.
    mu = mean_m mu_m ; scale = sqrt(mean_m s_m^2 + var_m mu_m)."""
    from neurotwin.models.cnp import predict
    q = np.asarray(q, np.float32)
    preds = [predict(m, task, is_ctx, q, device="cpu") for m in models]
    mus = np.stack([p[0] for p in preds])
    mu = mus.mean(0)
    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + mus.var(0))
    return mu, sc, mus


def epistemic_score(mus: np.ndarray, sc: np.ndarray) -> float:
    """results/conformal_r7.json: mean ensemble std of mu / mean predictive scale (at the held-out levels)."""
    return float(mus.std(0).mean() / sc.mean())


def context_levels(task, k: int = DEFAULT_K, design: str = "spread", custom=None) -> np.ndarray:
    """Measured (context) concentration levels, log10 µM. design='spread' reuses demo.spread_design;
    design='custom' takes level indices into task.levels."""
    kmax = min(max(k_choices()), len(task.levels) - 1)
    if design == "spread":
        return engine.spread_design(task.levels, int(max(1, min(int(k), kmax))))
    if design == "custom":
        idx = sorted({int(i) for i in (custom or [])})
        if not idx:
            raise ValueError("choose at least one measured concentration")
        if len(idx) > kmax:
            raise ValueError(f"choose at most {kmax} measured concentrations (at least one must stay held out)")
        if idx[0] < 0 or idx[-1] >= len(task.levels):
            raise ValueError("concentration index out of range")
        return task.levels[idx].astype(np.float32)
    raise ValueError(f"unknown design '{design}'")


def run_forecast(chemical: str = DEFAULT_CHEMICAL, k: int = DEFAULT_K, design: str = "spread",
                 custom=None, n_grid: int = 60) -> dict:
    """Forecast the whole dose x DIV response of a held-out chemical from k measured concentrations.
    Pure compute (no UI). Arrays: *_grid (G, n_DIV, n_features), *_held (H, n_DIV, n_features)."""
    t0 = time.perf_counter()
    task = get_task(chemical)
    ctx = context_levels(task, k, design, custom)
    is_ctx, is_tgt = split_context(task, ctx)
    held = np.unique(task.logc[is_tgt])
    models, z90 = get_models(task.fold)
    mu_h, sc_h, mus_h = ensemble_predict(models, task, is_ctx, held)
    grid = np.linspace(task.levels.min() - 0.3, task.levels.max() + 0.3, n_grid).astype(np.float32)
    mu_g, sc_g, _ = ensemble_predict(models, task, is_ctx, grid)
    kk = int(len(ctx))
    epi = epistemic_score(mus_h, sc_h)
    q, band_method = band_multiplier(task.fold, kk, z90)
    return {
        "chemical": task.chem, "fold": int(task.fold), "label": task.label, "k": kk, "design": design,
        "model_never_saw_this_chemical": True, "n_ensemble": len(models),
        "levels_logc": task.levels.astype(np.float32), "context_logc": np.asarray(ctx, np.float32),
        "held_logc": held.astype(np.float32), "grid_logc": grid,
        "mu_grid": mu_g, "scale_grid": sc_g, "mu_held": mu_h, "scale_held": sc_h,
        "band_q": float(q), "band_method": band_method, "z90_parametric": float(z90),
        "epistemic_score": epi, "abstention": abstention(epi, task.fold, kk),
        "n_context_wells": int(is_ctx.sum()), "n_heldout_wells": int(is_tgt.sum()),
        "runtime_s": round(time.perf_counter() - t0, 3),
    }


def reveal(fc: dict) -> dict:
    """Score a forecast against the real held-out wells and the paper's few-shot baselines."""
    task = get_task(fc["chemical"])
    is_ctx, is_tgt = split_context(task, fc["context_logc"])
    held = fc["held_logc"]
    lv, obs, ok = level_means(task, is_tgt)
    if not np.allclose(lv, held):
        raise RuntimeError("held-out levels do not match the forecast")
    preds = {"neurotrajectory": fc["mu_held"],
             "loglinear_interp": predict_interp(task, is_ctx, held),
             "analog_knn": get_knn(task.fold).predict(task, is_ctx, held)}
    metrics = {}
    for name, p in preds.items():
        err = np.abs(p - obs)
        by_div = [float(err[:, d][ok[:, d]].mean()) if ok[:, d].any() else float("nan") for d in range(len(DIVS))]
        metrics[name] = {"curve_mae": float(err[ok].mean()), "by_div": by_div}
    idx = np.searchsorted(held, task.logc[is_tgt])
    inside = (np.abs(task.y[is_tgt] - fc["mu_held"][idx]) <= fc["band_q"] * fc["scale_held"][idx])[task.m[is_tgt]]
    best_base = min(("loglinear_interp", "analog_knn"), key=lambda m: metrics[m]["curve_mae"])
    return {
        "chemical": task.chem, "k": fc["k"], "metrics": metrics, "best_baseline": best_base,
        "band_coverage": float(inside.mean()), "n_band_entries": int(inside.size),
        "obs_held": obs, "ok_held": ok,
        "cv_reference": cv_reference(fc["k"]), "cv_this_chemical": cv_record(task.chem, fc["k"]),
    }


def cv_reference(k: int) -> dict | None:
    """Cross-validated reference at k measured concentrations (results/trajectory_cv.json, conformal_r7.json)."""
    tcv = _json("trajectory_cv.json")
    if not tcv or str(k) not in tcv.get("by_k", {}):
        return None
    e = tcv["by_k"][str(k)]
    cm = e["curve_mae"]
    out = {"k": int(k), "n_chemicals": cm["paired_vs_best"]["n_chemicals"],
           "designs_per_k": tcv.get("designs_per_k"),
           "curve_mae": {m: cm["mean"][m] for m in ("neurotrajectory", "loglinear_interp", "analog_knn")},
           "best_baseline": cm["best_baseline"], "paired_vs_best": cm["paired_vs_best"],
           "parametric_band_coverage": e.get("well_interval90", {}).get("coverage_mean")}
    conf = _json("conformal_r7.json")
    ent = (conf or {}).get("results", {}).get(f"alpha_{ALPHA}", {}).get(f"k{k}")
    if ent:
        out["conformal_coverage"] = ent["conformal"]["coverage"]
        out["conformal_ci95"] = ent["conformal"]["ci95"]
    return out


def cv_record(chemical: str, k: int) -> dict | None:
    """This chemical's own CV record (mean over the CV's random designs) from trajectory_cv_per_chemical.csv."""
    d = _per_chemical_cv()
    if d is None:
        return None
    r = d[(d.chemical == chemical) & (d.k == int(k))]
    if r.empty:
        return None
    return {m: float(r[r.method == m].curve_mae.iloc[0])
            for m in ("neurotrajectory", "loglinear_interp", "analog_knn") if (r.method == m).any()}


# ====================================================================== abstention + calibrated band
def bundle_fingerprint() -> str:
    """SHA-256 over the manifest's per-file hashes (tasks + all fold models)."""
    files = manifest()["files"]
    return hashlib.sha256(json.dumps({k: v["sha256"] for k, v in sorted(files.items())}).encode()).hexdigest()


_THR = {"loaded": False, "data": None, "status": "not loaded", "building": False}


def threshold_paths() -> list[Path]:
    return [BUNDLE / THRESHOLD_FILE, CACHE_DIR / THRESHOLD_FILE]


def load_thresholds(refresh: bool = False) -> dict | None:
    """Abstention thresholds / conformal multipliers built from the CV predictions of this bundle.
    Looks in data_bundle/ then in the cache dir (NT_APP_CACHE, default outputs/app). Stale files
    (different bundle fingerprint) are ignored."""
    with _LOCK:
        if _THR["loaded"] and not refresh:
            return _THR["data"]
        _THR.update(loaded=True, data=None,
                    status=f"no {THRESHOLD_FILE} found (run: python app/app.py --build-thresholds)")
        fp = bundle_fingerprint()
        for p in threshold_paths():
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                _THR["status"] = f"unreadable {p.name}: {e}"
                continue
            if d.get("bundle_fingerprint") != fp:
                _THR["status"] = f"{p} was built for a different bundle; ignored"
                continue
            _THR.update(data=d, status=f"loaded from {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
            break
        return _THR["data"]


def threshold_status() -> str:
    load_thresholds()
    return _THR["status"]


def band_multiplier(fold: int, k: int, z90: float) -> tuple[float, str]:
    th = load_thresholds()
    qs = (th or {}).get("conformal_qhat_by_fold", {}).get(str(fold), {})
    if qs:
        kc = str(min(int(k), max(int(x) for x in qs)))
        return float(qs[kc]), (f"cross-conformal 90% band (multiplier calibrated on the held-out chemicals of the "
                               f"other folds, k={kc}{'; k>3 uses k=3' if int(k) > int(kc) else ''})")
    return float(z90), "parametric 90% band (model scale x z90); conformal multipliers not available"


def abstention(epi: float, fold: int, k: int) -> dict:
    conf = _json("conformal_r7.json") or {}
    rule = conf.get("abstention", {}).get("rule", "abstain if epistemic score > 90th percentile of calibration folds")
    th = load_thresholds()
    thr = (th or {}).get("epistemic_threshold_by_fold", {}).get(str(fold))
    if thr is None:
        return {"available": False, "score": float(epi), "threshold": None, "abstain": None, "rule": rule,
                "note": "threshold not available in this bundle: " + threshold_status()}
    ks = (th or {}).get("calibration_k", [])
    note = "Threshold cross-fitted on the CV predictions of the other folds"
    if ks and int(k) > max(ks):
        note += f"; calibrated for k<={max(ks)}, applied here to k={k}"
    return {"available": True, "score": float(epi), "threshold": float(thr), "abstain": bool(epi > thr),
            "rule": rule, "note": note}


def _fold_calibration_records(fold: int, n_designs: int = 3, threads: int | None = None) -> list[dict]:
    """Out-of-fold predictions of one fold's chemicals, mirroring scripts/run_conformal.py
    (k in its KS, first n_designs seeded designs of scripts/run_trajectory_cv.py)."""
    import torch
    if threads:
        torch.set_num_threads(int(threads))
    sys.path.insert(0, str(REPO / "scripts"))
    from run_conformal import KS               # noqa: E402  (same k values as the paper)
    from run_trajectory_cv import designs      # noqa: E402  (same seeded designs as the paper)
    models, _ = get_models(fold)
    out = []
    for t in [x for x in _bundle()[0] if x.fold == fold]:
        for k in KS:
            if k >= len(t.levels):
                continue
            for di, ctx_lv in enumerate(designs(t, k)[:n_designs]):
                ic, it = split_context(t, ctx_lv)
                lv = np.unique(t.logc[it])
                mu, sc, mus = ensemble_predict(models, t, ic, lv)
                idx = np.searchsorted(lv, t.logc[it])
                mm = t.m[it]
                _, obs, ok = level_means(t, it)
                out.append({"chemical": t.chem, "fold": int(fold), "k": int(k), "design": di,
                            "epi": epistemic_score(mus, sc), "curve_mae": float(np.abs(mu - obs)[ok].mean()),
                            "scores": (np.abs(t.y[it] - mu[idx]) / sc[idx])[mm].astype(np.float32)})
    return out


def aggregate_calibration(records: list[dict], alpha: float = ALPHA, pct: float = ABSTAIN_PERCENTILE) -> dict:
    """Cross-fitted per-fold thresholds + split-conformal multipliers (regimes pooled) + reproduction check."""
    folds = sorted({r["fold"] for r in records})
    ks = sorted({r["k"] for r in records})
    epi = np.array([r["epi"] for r in records])
    fo = np.array([r["fold"] for r in records])
    kk = np.array([r["k"] for r in records])
    thr = {g: float(np.percentile(epi[fo != g], pct)) for g in folds}
    qhat = {}
    for g in folds:
        qhat[g] = {}
        for k in ks:
            s = np.concatenate([r["scores"] for r in records if r["fold"] != g and r["k"] == k])
            n = len(s)
            qhat[g][k] = float(np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
    ab = np.array([r["epi"] > thr[r["fold"]] for r in records])
    mae = np.array([r["curve_mae"] for r in records])
    cov = {}
    for k in ks:
        per = {}
        for r in records:
            if r["k"] == k:
                per.setdefault(r["chemical"], []).append(float((r["scores"] <= qhat[r["fold"]][k]).mean()))
        cov[str(k)] = round(float(np.mean([np.mean(v) for v in per.values()])), 4)
    return {
        "epistemic_threshold_by_fold": {str(g): thr[g] for g in folds},
        "conformal_qhat_by_fold": {str(g): {str(k): qhat[g][k] for k in ks} for g in folds},
        "calibration_k": ks,
        "reproduction": {
            "abstention_rate": round(float(ab.mean()), 4),
            "curve_mae_retained": round(float(mae[~ab].mean()), 4) if (~ab).any() else None,
            "curve_mae_abstained": round(float(mae[ab].mean()), 4) if ab.any() else None,
            "curve_mae_all": round(float(mae.mean()), 4),
            "by_k": {str(k): {"rate": round(float(ab[kk == k].mean()), 4)} for k in ks},
            f"conformal_coverage_alpha_{alpha}_regimes_pooled_by_k": cov},
    }


def build_abstention_thresholds(out_path: Path | None = None, workers: int | None = None,
                                n_designs: int = 3) -> dict:
    """One-off build from the shipped bundle (CPU). Writes THRESHOLD_FILE and returns its content."""
    conf = _json("conformal_r7.json") or {}
    rule = conf.get("abstention", {}).get("rule", "")
    if f"{ABSTAIN_PERCENTILE}th percentile" not in rule:
        raise RuntimeError(f"abstention rule in conformal_r7.json changed: '{rule}'")
    t0 = time.time()
    folds = sorted({t.fold for t in _bundle()[0]})
    workers = max(1, min(len(folds), workers or max(1, (os.cpu_count() or 2) // 4)))
    records = []
    if workers == 1:
        for f in folds:
            records += _fold_calibration_records(f, n_designs)
            print(f"fold {f}: {len(records)} records, {time.time() - t0:.0f} s", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        threads = max(1, (os.cpu_count() or 2) // workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f, recs in zip(folds, ex.map(_fold_calibration_records, folds, [n_designs] * len(folds),
                                             [threads] * len(folds))):
                records += recs
                print(f"fold {f}: {len(records)} records, {time.time() - t0:.0f} s", flush=True)
    agg = aggregate_calibration(records)
    out = {
        "description": "Cross-fitted abstention thresholds and conformal 90% multipliers for the NeuroChip Twin app, "
                       "rebuilt from the shipped bundle with the protocol of scripts/run_conformal.py.",
        "protocol": "Out-of-fold predictions (each chemical predicted only by its own fold's ensemble) for k in "
                    f"{agg['calibration_k']} measured concentrations x first {n_designs} seeded designs. Epistemic "
                    "score = mean ensemble std of mu / mean predictive scale. For fold g the threshold is the "
                    f"{ABSTAIN_PERCENTILE}th percentile of the scores of the other folds; the conformal multiplier "
                    f"is the finite-sample (1-alpha) quantile, alpha={ALPHA}, of |y-mu|/scale over the other folds' "
                    "held-out well entries at the same k (cytotoxicity regimes pooled: viability calls are not in "
                    "the bundle).",
        "abstention_rule": rule, "alpha": ALPHA, "n_records": len(records),
        "n_chemicals": len({r["chemical"] for r in records}),
        "bundle_fingerprint": bundle_fingerprint(),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_seconds": round(time.time() - t0, 1),
        **agg,
        "published_reference": {k: conf.get("abstention", {}).get(k) for k in
                                ("abstention_rate", "curve_mae_retained", "curve_mae_abstained", "curve_mae_all")},
    }
    path = Path(out_path) if out_path else CACHE_DIR / THRESHOLD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    load_thresholds(refresh=True)
    return out


def start_threshold_build(workers: int | None = None) -> bool:
    """Server start: if no valid thresholds file exists, build it from the bundle in a daemon thread
    (about 20 s on a desktop CPU, a few minutes on a 2-vCPU Space). Until it finishes the badge says
    'unavailable' and the band is parametric; the next forecast after the build uses the thresholds."""
    with _LOCK:
        if load_thresholds() is not None or _THR["building"]:
            return False
        _THR.update(building=True, status="being built from the cross-validation predictions of this bundle, "
                                           "in the background (the badge appears on the next forecast once done)")

    def _run():
        try:
            out = build_abstention_thresholds(workers=workers)
            print(f"abstention thresholds built in {out['build_seconds']} s; {threshold_status()}", flush=True)
        except Exception as e:                                   # noqa: BLE001  (report, keep serving)
            with _LOCK:
                _THR["status"] = (f"automatic build failed ({type(e).__name__}: {e}); "
                                  "run: python app/app.py --build-thresholds")
            print(f"abstention thresholds: {_THR['status']}", flush=True)
        finally:
            _THR["building"] = False

    threading.Thread(target=_run, daemon=True, name="abstention-thresholds").start()
    return True


# ====================================================================== figures (matplotlib OO API: thread-safe)
def _new_figure(w, h):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=110, layout="constrained")
    fig.patch.set_facecolor("white")
    return fig


def _log_axis(ax):
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:g}"))


def _wrap(name: str, narrow: bool) -> str:
    """Long chemical names wrapped so that figure titles are never clipped."""
    import textwrap
    return "\n".join(textwrap.wrap(name, 44 if narrow else 90)) or name


def forecast_figure(fc: dict, rev: dict | None = None, features=None, narrow: bool = False):
    """Dose-response panels (features x DIV): forecast, 90 % band, measured wells, revealed wells.
    narrow=True (phone width): each feature is a 2 x 2 block of DIV panels in a 6-inch-wide figure."""
    feats = [f for f in (features or DEFAULT_FEATURES) if f in FEATURES] or DEFAULT_FEATURES
    task = get_task(fc["chemical"])
    is_ctx, is_tgt = split_context(task, fc["context_logc"])
    x, q = fc["grid_logc"], fc["band_q"]
    nf, nd = len(feats), len(DIVS)
    if narrow:
        fig = _new_figure(6.2, 3.9 * nf + 1.6)
        axes = fig.subplots(2 * nf, 2, sharex=True, squeeze=False)
        cells = {(i, d): axes[2 * i + d // 2, d % 2] for i in range(nf) for d in range(nd)}
    else:
        fig = _new_figure(11, 1.95 * nf + 1.3)
        axes = fig.subplots(nf, nd, sharex=True, squeeze=False)
        cells = {(i, d): axes[i, d] for i in range(nf) for d in range(nd)}
    for i, f in enumerate(feats):
        j = FEATURES.index(f)
        for d, div in enumerate(DIVS):
            ax = cells[i, d]
            mu, sc = fc["mu_grid"][:, d, j], fc["scale_grid"][:, d, j]
            ax.fill_between(x, mu - q * sc, mu + q * sc, color=C_BAND, alpha=0.35, lw=0, label="90% predictive band")
            ax.plot(x, mu, color=C_LINE, lw=2, label="forecast")
            ax.axhline(0, color=C_GRID, lw=0.8, ls=":")
            if rev is None:
                for lvl in fc["held_logc"]:
                    ax.axvline(lvl, color="#c8c8c8", lw=0.7, ls="--", zorder=0)
            mm = task.m[is_ctx][:, d, j]
            ax.scatter(task.logc[is_ctx][mm], task.y[is_ctx][mm, d, j], s=16, color=C_MEASURED, alpha=0.85,
                       zorder=3, label="measured wells (given to the model)")
            if rev is not None:
                mm = task.m[is_tgt][:, d, j]
                ax.scatter(task.logc[is_tgt][mm], task.y[is_tgt][mm, d, j], s=24, marker="D", color=C_REVEALED,
                           edgecolors="white", linewidths=0.4, alpha=0.9, zorder=4, label="held-out wells (revealed)")
            _log_axis(ax)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color="#eeeeee", lw=0.6)
            if narrow:
                ax.set_title(f"{f.replace('_', ' ')} - DIV {div}", fontsize=8.5)
                ax.tick_params(labelbottom=True)
            else:
                if i == 0:
                    ax.set_title(f"DIV {div}", fontsize=10)
                if d == 0:
                    ax.set_ylabel(f.replace("_", " "), fontsize=8.5)
            if i == nf - 1 and (not narrow or d >= 2):
                ax.set_xlabel("concentration (µM, log)", fontsize=8.5)
    h, lab = cells[0, 0].get_legend_handles_labels()
    if rev is None:
        from matplotlib.lines import Line2D
        h.append(Line2D([0], [0], color="#c8c8c8", ls="--"))
        lab.append("held-out concentration (hidden)")
    fig.legend(h, lab, loc="outside lower center", ncol=2 if narrow else 4, frameon=False, fontsize=8.5)
    sep = "\n" if narrow else " "
    fig.suptitle(f"{_wrap(task.chem, narrow)}\nforecast from {fc['k']} measured concentration(s),{sep}"
                 f"fold {fc['fold']} models (never trained on this chemical)\ny: {UNITS}", fontsize=9.5)
    return fig


def trajectory_figure(fc: dict, rev: dict | None = None, level_index: int | None = None, narrow: bool = False):
    """17 features x DIV heatmaps at one held-out concentration: forecast | observed | observed - forecast.
    narrow=True (phone width) stacks the panels vertically."""
    import matplotlib
    held = fc["held_logc"]
    li = len(held) - 1 if level_index is None else int(np.clip(int(level_index), 0, len(held) - 1))
    panels = [("forecast", fc["mu_held"][li])]
    if rev is not None:
        obs = np.where(rev["ok_held"][li], rev["obs_held"][li], np.nan)
        panels += [("observed (held out)", obs), ("observed - forecast", obs - fc["mu_held"][li])]
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#dddddd")
    if narrow:
        fig = _new_figure(6.0, 4.1 * len(panels) + 0.8)
        axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
    else:
        fig = _new_figure(2.3 * len(panels) + 3.2, 6.3)
        axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    im = None
    for n, (ax, (title, M)) in enumerate(zip(axes, panels)):
        im = ax.imshow(np.ma.masked_invalid(M.T), cmap=cmap, vmin=-HEAT_VMAX, vmax=HEAT_VMAX, aspect="auto")
        ax.set_xticks(range(len(DIVS)), [f"DIV{d}" for d in DIVS], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        if narrow or n == 0:
            ax.set_yticks(range(len(FEATURES)), [f.replace("_", " ") for f in FEATURES], fontsize=7.5)
        if narrow:
            ax.tick_params(labelbottom=True)
    fig.colorbar(im, ax=list(axes), shrink=0.6 if narrow else 0.85, label=UNITS)
    fig.suptitle(f"{_wrap(fc['chemical'], narrow)}\nat {level_label(held[li])}: developmental trajectory of "
                 f"{len(FEATURES)} network features", fontsize=9.5)
    return fig


# ====================================================================== tables (read from results JSON)
def _ci(d, key="mean_diff", ci="ci95", nd=3):
    lo, hi = d[ci]
    return f"{d[key]:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def metrics_table(rev: dict):
    import pandas as pd
    nt = rev["metrics"]["neurotrajectory"]["curve_mae"]
    rows = []
    for m in ("neurotrajectory", "loglinear_interp", "analog_knn"):
        e = rev["metrics"][m]
        rows.append({"Method": METHOD_LABELS[m], "Curve MAE (all DIV)": round(e["curve_mae"], 3),
                     **{f"DIV {d}": round(v, 3) for d, v in zip(DIVS, e["by_div"])},
                     "NeuroTrajectory - method": "-" if m == "neurotrajectory" else f"{nt - e['curve_mae']:+.3f}"})
    return pd.DataFrame(rows)


def r2_table():
    import pandas as pd
    tcv = _json("trajectory_cv.json")
    if not tcv:
        return pd.DataFrame()
    rows = []
    for k in sorted(tcv["by_k"], key=int):
        e = tcv["by_k"][k]
        cm = e["curve_mae"]
        f14 = tcv.get("by_k_folds_1to4", {}).get(k)
        rows.append({
            "k measured": int(k),
            "NeuroTrajectory": cm["mean"]["neurotrajectory"],
            "Log-linear interp.": cm["mean"]["loglinear_interp"],
            "Analog kNN": cm["mean"]["analog_knn"],
            "Hill": cm["mean"]["hill_per_endpoint"],
            "Best baseline": METHOD_LABELS.get(cm["best_baseline"], cm["best_baseline"]),
            "Diff. vs best [95% CI]": _ci(cm["paired_vs_best"]),
            "Rel. change": f"{cm['paired_vs_best']['rel_change_pct']:+.1f}%",
            "Chemicals improved": f"{100 * cm['paired_vs_best']['frac_chem_improved']:.0f}%",
            "Folds 1-4 only: diff. [95% CI]": _ci(f14["paired_vs_best"]) if f14 else "-",
            "Wells in 90% band (parametric)": f"{100 * e['well_interval90']['coverage_mean']:.1f}%",
        })
    return pd.DataFrame(rows)


def conformal_table():
    import pandas as pd
    conf = _json("conformal_r7.json")
    if not conf:
        return pd.DataFrame()
    rows = []
    for a, ent in conf["results"].items():
        for kk, e in ent.items():
            if not kk.startswith("k"):
                continue
            p, c = e["parametric"], e["conformal"]
            rows.append({"Nominal": f"{100 * ent['nominal']:.0f}%", "k measured": int(kk[1:]),
                         "Parametric coverage [95% CI]": f"{p['coverage']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]",
                         "Parametric width": p["width"],
                         "Conformal coverage [95% CI]": f"{c['coverage']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]",
                         "Conformal width": c["width"]})
    return pd.DataFrame(rows)


def abstention_table():
    import pandas as pd
    ab = (_json("conformal_r7.json") or {}).get("abstention")
    if not ab:
        return pd.DataFrame()
    rows = [{"k measured": "all", "Abstention rate": f"{100 * ab['abstention_rate']:.1f}%",
             "Curve MAE retained": ab["curve_mae_retained"], "Curve MAE abstained": ab["curve_mae_abstained"]}]
    for k, e in sorted(ab.get("by_k", {}).items(), key=lambda kv: int(kv[0])):
        rows.append({"k measured": k, "Abstention rate": f"{100 * e['rate']:.1f}%",
                     "Curve MAE retained": e["mae_retained"], "Curve MAE abstained": e["mae_abstained"]})
    return pd.DataFrame(rows)


def epa_table():
    import pandas as pd
    epa = _json("epa_baseline_repro.json")
    if not epa:
        return pd.DataFrame()
    rows = []
    for name, m in epa["models"].items():
        dd = m.get("published_delta_pp", {})
        rows.append({"EPA model": name, "Hits needed": m["threshold_hits"],
                     "Sensitivity %": m["sensitivity_pct"], "Specificity %": m["specificity_pct"],
                     "Balanced accuracy %": m["balanced_accuracy_pct"],
                     "Diff. vs published (pp)": f"{dd.get('sensitivity_pp', float('nan')):+.1f} / "
                                                f"{dd.get('specificity_pp', float('nan')):+.1f} / "
                                                f"{dd.get('balanced_accuracy_pp', float('nan')):+.1f}"})
    return pd.DataFrame(rows)


def results_figure(narrow: bool = False):
    """Curve MAE vs k (left) and paired difference vs best baseline with 95 % CI (right; below if narrow)."""
    tcv = _json("trajectory_cv.json")
    if not tcv:
        return None
    ks = sorted(tcv["by_k"], key=int)
    x = np.array([int(k) for k in ks])
    fig = _new_figure(6.0, 7.8) if narrow else _new_figure(10, 3.8)
    a1, a2 = fig.subplots(2, 1) if narrow else fig.subplots(1, 2)
    markers = {"neurotrajectory": "o", "loglinear_interp": "s", "analog_knn": "^", "hill_per_endpoint": "v",
               "context_mean": "x"}
    for m, col in METHOD_COLOURS.items():
        y = [tcv["by_k"][k]["curve_mae"]["mean"][m] for k in ks]
        a1.plot(x, y, marker=markers[m], color=col, lw=2.6 if m == "neurotrajectory" else 1.4,
                label=METHOD_LABELS[m])
    a1.set_xlabel("measured concentrations k")
    a1.set_ylabel("curve MAE on held-out chemicals")
    a1.set_xticks(x)
    a1.legend(fontsize=8, frameon=False)
    a1.set_title(f"5-fold chemical CV, {tcv['n_chemicals']} chemicals (lower is better)", fontsize=9.5)
    for off, key, lab, col in ((-0.08, "all", "all folds", C_LINE), (0.08, "f14", "folds 1-4 only", "#555555")):
        d, lo, hi = [], [], []
        for k in ks:
            pv = tcv["by_k"][k]["curve_mae"]["paired_vs_best"] if key == "all" else \
                tcv.get("by_k_folds_1to4", {}).get(k, {}).get("paired_vs_best")
            d.append(pv["mean_diff"] if pv else np.nan)
            lo.append(pv["ci95"][0] if pv else np.nan)
            hi.append(pv["ci95"][1] if pv else np.nan)
        d, lo, hi = map(np.array, (d, lo, hi))
        a2.errorbar(x + off, d, yerr=[d - lo, hi - d], fmt="o", color=col, capsize=3, label=lab)
    a2.axhline(0, color=C_GRID, lw=0.8, ls=":")
    a2.set_xticks(x)
    a2.set_xlabel("measured concentrations k")
    a2.set_ylabel("NeuroTrajectory - best baseline")
    a2.set_title("paired bootstrap over chemicals, 95% CI (< 0 favours ours)", fontsize=9.5)
    a2.legend(fontsize=8, frameon=False)
    return fig


def results_markdown() -> str:
    tcv, conf = _json("trajectory_cv.json"), _json("conformal_r7.json")
    if not tcv:
        return "results/trajectory_cv.json not found."
    k_best = max(tcv["by_k"], key=int)
    pv = tcv["by_k"][k_best]["curve_mae"]["paired_vs_best"]
    lines = [
        "### Few-shot dose x developmental-time forecasting on unseen chemicals (R2)",
        f"{tcv['n_chemicals']} chemicals, {len(tcv['outer_folds'])}-fold chemical cross-validation "
        f"(all samples of a chemical in one fold), {len(tcv['seeds'])}-seed ensemble, "
        f"{tcv['designs_per_k']} random designs per k, nested model selection. Metric: curve MAE, "
        "|prediction - mean of replicate wells| at every held-out concentration over "
        f"{len(FEATURES)} features x {len(DIVS)} DIV, in {UNITS}.",
        f"At k={k_best}: {pv['mean_diff']:+.3f} [{pv['ci95'][0]:+.3f}, {pv['ci95'][1]:+.3f}] vs the best baseline "
        f"({pv['rel_change_pct']:+.1f}%), better on {100 * pv['frac_chem_improved']:.0f}% of chemicals. "
        "'Folds 1-4 only' excludes the fold used once during development.",
    ]
    if conf:
        lines += ["### Calibrated uncertainty and abstention (R7)",
                  "All predictions out-of-fold; cross-conformal calibration per cytotoxicity regime x k. "
                  f"Abstention rule: *{conf['abstention']['rule']}*."]
    return "\n\n".join(lines)


def chip_markdown() -> str:
    r8, cp = _json("r8_chiplayer.json"), _json("chiplayer_params.json")
    if not r8:
        return "results/r8_chiplayer.json not found."
    cv = r8["cv"]
    s = cv["summary"]
    cmp_t = cv["comparisons"]["tunnel_mean"]
    cmp_p = cv["comparisons"]["permuted_direction"]
    cond = ", ".join(f"{k} {v}" for k, v in r8["conditions"].items())
    lines = [
        "### Directed propagation on a real 4-compartment MEA (Brewer hippocampal circuit)",
        f"Source: [{r8['source']}]({r8['source']}) ({r8['license']}). {r8['n_recordings']} recordings "
        f"({cond}), {r8['n_axons']} sorted axons crossing microtunnels. Graph: {r8['graph_description']}.",
        f"**Model:** {cv['model']}. **Protocol:** {cv['protocol']}; baselines: {cv['baseline_tunnel']}; "
        f"{cv['baseline_direction_null']}.",
        "| Held-out recording | Model | Tunnel mean | Permuted direction |\n|---|---|---|---|\n"
        f"| Feed-forward fraction MAE | {s['model']['ff_fraction_mae']:.3f} | {s['tunnel_mean']['ff_fraction_mae']:.3f} "
        f"| {s['permuted_direction']['ff_fraction_mae']:.3f} |\n"
        f"| Conduction time MAE (ms) | {s['model']['conduction_mae_ms']:.3f} | {s['tunnel_mean']['conduction_mae_ms']:.3f} "
        f"| {s['permuted_direction']['conduction_mae_ms']:.3f} |",
        f"Model - tunnel mean, FF fraction MAE: {_ci(cmp_t['ff_fraction_mae'], 'difference_model_minus_baseline')} "
        f"(bootstrap p = {cmp_t['ff_fraction_mae']['p_bootstrap_no_improvement']:.3f}); conduction MAE: "
        f"{_ci(cmp_t['conduction_mae_ms'], 'difference_model_minus_baseline')} ms "
        f"(p = {cmp_t['conduction_mae_ms']['p_bootstrap_no_improvement']:.3f}). "
        f"Model - permuted direction, FF fraction MAE: "
        f"{_ci(cmp_p['ff_fraction_mae'], 'difference_model_minus_baseline')} "
        f"(p = {cmp_p['ff_fraction_mae']['p_bootstrap_no_improvement']:.3f}). {cv['direction_null_delay_note']}.",
    ]
    if cp:
        rows = "\n".join(
            f"| {e['pair']} | {e['n_axons_ff']} | {e['n_axons_fb']} | {e['observed_directionality_d']:+.2f} | "
            f"{e['conduction']['median_ms']:.2f} [{e['conduction']['p10_ms']:.2f}, {e['conduction']['p90_ms']:.2f}] |"
            for e in cp["fit_pair_edges"])
        lines += [f"**Fitted pair parameters** ({cp['source_cohort']}; {cp['scope']}). "
                  "Directionality d = (FF - FB) / (FF + FB).",
                  "| Pair | FF axons | FB axons | d | Conduction median [p10, p90] ms |\n|---|---|---|---|---|\n" + rows]
    lines.append(f"*Bursts:* {r8['burst_interpretation']}. These are descriptive parameters of one real circuit; "
                 "any compartment scenario built on them is a simulation, not a calibrated cross-platform model.")
    return "\n\n".join(lines)


def chip_images() -> list[tuple[str, str]]:
    caps = {"chip_direction.png": "Recording-level directed axon balance per compartment pair (NoStim)",
            "chip_cv.png": "Leave-one-recording-out error: pair/tunnel shrinkage vs tunnel mean"}
    return [(str(p), caps.get(p.name, p.stem)) for p in sorted(FIGURES.glob("chip_*.png"))]


EVIDENCE_BOUNDARIES = (
    "- **What the data are:** rat primary cortical neurons in 48-well multi-electrode-array plates "
    "(US EPA Network Formation Assay), recorded at DIV {divs}. This is **not a validated twin of any "
    "commercial organ-on-a-chip**.\n"
    "- **Chip layer:** directed-propagation parameters come from one real 4-compartment hippocampal MEA "
    "(Brewer lab). Compartment scenarios built on them are **simulations**.\n"
    "- **Held-out:** each chemical is forecast only by the models of its own cross-validation fold, which "
    "never saw it. The abstention flag is decided before the reveal, from the model alone.\n"
    "- **Not clinical:** research demonstration of in-vitro developmental neurotoxicity screening; not for "
    "clinical, regulatory or safety decisions.")


def about_markdown() -> str:
    man = manifest()
    files = man["files"]
    n_models = sum(1 for k in files if k.startswith("models/"))
    tasks_meta = files.get("nfa_tasks.npz", {})
    model_lic = next((v["licence"] for k, v in files.items() if k.startswith("models/")), "-")
    r8 = _json("r8_chiplayer.json") or {}
    rows = [f"| `nfa_tasks.npz` ({man['n_chemicals']} chemicals, {man['n_wells']} wells) | "
            f"{tasks_meta.get('source', '-')} | {tasks_meta.get('licence', '-')} |",
            f"| `transform.json` | {files.get('transform.json', {}).get('source', '-')} | "
            f"{files.get('transform.json', {}).get('licence', '-')} |",
            f"| `models/` ({n_models} checkpoints = folds x seeds) | trained by scripts/run_trajectory_cv.py | {model_lic} |"]
    if r8:
        rows.append(f"| Brewer 4-compartment MEA (chip layer, results only) | {r8['source']} | {r8['license']} |")
    return "\n\n".join([
        "### Evidence boundaries", EVIDENCE_BOUNDARIES.format(divs="/".join(map(str, DIVS))),
        "### Data and licences (from `data_bundle/manifest.json`)",
        "| Item | Source | Licence |\n|---|---|---|\n" + "\n".join(rows),
        f"Citation: {tasks_meta.get('citation', '-')}. Every bundle file is pinned by SHA-256 in the manifest "
        "(`python scripts/verify_bundle.py`).",
        "### How it works",
        f"NeuroTrajectory is a conditional neural process meta-trained across chemicals. Given the wells at k "
        f"measured concentrations it predicts, for any concentration, the whole trajectory of {len(FEATURES)} MEA "
        f"network features at DIV {'/'.join(map(str, DIVS))} with a predictive scale; an ensemble of seeds gives "
        "the epistemic score used for abstention. Baselines receive exactly the same measured wells.",
        "### Reproduce (CPU, offline)",
        "```bash\npip install -r requirements-app.txt\npython scripts/verify_bundle.py      # SHA-256 of the bundle\n"
        "python demo.py --chemical Deltamethrin --k 3   # command-line demo\n"
        "python app/app.py --smoke                # compute core, no server\n"
        "python app/app.py --build-thresholds     # abstention thresholds (else built at server start)\n"
        "python app/app.py                        # this app\npython -m pytest tests/test_app.py -q\n```",
        f"Abstention thresholds: {threshold_status()}.",
    ])


# ====================================================================== UI helpers
def _badge_html(ab: dict) -> str:
    if not ab["available"]:
        return (f"<div class='nt-badge nt-na'>Abstention check unavailable</div>"
                f"<div class='nt-note'>Epistemic score {ab['score']:.3f}. {ab['note']}</div>")
    if ab["abstain"]:
        return (f"<div class='nt-badge nt-abstain'>ABSTAIN - low confidence</div>"
                f"<div class='nt-note'>Epistemic score {ab['score']:.3f} &gt; threshold {ab['threshold']:.3f} "
                f"(rule: {ab['rule']}). Measure more concentrations before relying on this forecast. "
                f"{ab['note']}.</div>")
    return (f"<div class='nt-badge nt-ok'>Forecast retained</div>"
            f"<div class='nt-note'>Epistemic score {ab['score']:.3f} &le; threshold {ab['threshold']:.3f} "
            f"(rule: {ab['rule']}). {ab['note']}.</div>")


def _info_md(fc: dict) -> str:
    ctx = ", ".join(level_label(x) for x in fc["context_logc"])
    held = ", ".join(level_label(x) for x in fc["held_logc"])
    lab = f"EPA DNT reference label: **{fc['label']}**" if fc["label"] != "unknown" else "no EPA DNT reference label"
    return (f"**{fc['chemical']}** ({lab}) is scored with the **fold {fc['fold']}** ensemble "
            f"({fc['n_ensemble']} models), trained **without this chemical (held-out)**.  \n"
            f"Measured, given to the model: {fc['k']} concentration(s), {fc['n_context_wells']} wells ({ctx}).  \n"
            f"Hidden, to reveal: {len(fc['held_logc'])} concentration(s), {fc['n_heldout_wells']} wells ({held}).  \n"
            f"<span class='nt-note'>Band: {fc['band_method']}. Compute time {fc['runtime_s']:.2f} s.</span>")


def _reveal_md(fc: dict, rev: dict) -> str:
    m = rev["metrics"]
    win = "beats" if m["neurotrajectory"]["curve_mae"] < m[rev["best_baseline"]]["curve_mae"] else "does not beat"
    lines = [f"**This chemical:** NeuroTrajectory {win} the better baseline "
             f"({METHOD_LABELS[rev['best_baseline']]}) here. Held-out well entries inside the 90% band: "
             f"**{100 * rev['band_coverage']:.1f}%** (n = {rev['n_band_entries']})."]
    ref = rev["cv_reference"]
    if ref:
        pv = ref["paired_vs_best"]
        cm = ref["curve_mae"]
        line = (f"**Cross-validation reference, k={ref['k']}** ({ref['n_chemicals']} held-out chemicals x "
                f"{ref['designs_per_k']} random designs; results/trajectory_cv.json): curve MAE NeuroTrajectory "
                f"{cm['neurotrajectory']:.3f}, log-linear {cm['loglinear_interp']:.3f}, analog kNN "
                f"{cm['analog_knn']:.3f}; paired difference vs best baseline ({METHOD_LABELS[ref['best_baseline']]}) "
                f"{_ci(pv)}, {100 * pv['frac_chem_improved']:.0f}% of chemicals improved.")
        if ref.get("conformal_coverage") is not None:
            line += (f" Conformal 90% coverage at this k: {ref['conformal_coverage']:.3f} "
                     f"[{ref['conformal_ci95'][0]:.3f}, {ref['conformal_ci95'][1]:.3f}].")
        lines.append(line)
    rec = rev["cv_this_chemical"]
    if rec:
        lines.append(f"This chemical in the CV run (k={rev['k']}, mean over random designs): " + ", ".join(
            f"{METHOD_LABELS[k]} {v:.3f}" for k, v in rec.items()) + ". The app uses the design chosen above, "
            "so its numbers differ.")
    return "\n\n".join(lines)


def _custom_choices(task):
    return [(level_label(x), str(i)) for i, x in enumerate(task.levels)]


def _spread_indices(task, k):
    ctx = context_levels(task, k, "spread")
    return [str(int(i)) for i in np.searchsorted(task.levels, ctx)]


CSS = """
.gradio-container {max-width: min(1280px, 100%) !important; min-width: 0 !important; margin: auto !important;}
/* wide tables scroll inside their box instead of widening the page (phone width) */
.gradio-container .table-container {overflow-x: auto !important; max-width: 100%; min-width: 0;}
@media (max-width: 720px) {
  .gradio-container .prose table {display: block; overflow-x: auto; max-width: 100%;}
  .gradio-container .prose th, .gradio-container .prose td {min-width: 5.5em; word-break: normal;
                                                            overflow-wrap: normal;}
}
.nt-badge {display:inline-block; padding:6px 14px; border-radius:6px; font-weight:700; border:2px solid;
           letter-spacing:.02em; margin-bottom:4px;}
.nt-ok {background:#e7f1fa; color:#0b3d66; border-color:#0072B2;}
.nt-abstain {background:#fbeee5; color:#7a2e00; border-color:#D55E00;}
.nt-na {background:#f1f1f1; color:#333; border-color:#8a8a8a;}
.nt-note {font-size:.9em; color:#3a3a3a;}
.nt-bounds {border-left:4px solid #0072B2; padding:4px 12px; background:#f5f9fc;}
"""
# Every figure event passes the browser width as its last input (hidden gr.Number): phone layout
# below NARROW_PX. A single input arrives as a bare value, several as an array.
VW_JS = ("(...args) => { args[args.length - 1] = window.innerWidth; "
         "return args.length === 1 ? args[0] : args; }")
NARROW_PX = 720
# Light theme always (the figures are drawn on white): Gradio adds body.dark from the OS preference, possibly
# after this script runs, so strip it now and whenever it comes back. Gradio 6 runs `js` as a <script>
# (hence the IIFE); Gradio 4/5 call the value of `js` as a function (hence the returned no-op).
FORCE_LIGHT_JS = """(() => {
  const light = () => { for (const el of [document.body, document.documentElement]) {
    if (el && el.classList.contains('dark')) el.classList.remove('dark'); } };
  light();
  for (const el of [document.body, document.documentElement]) {
    if (el) new MutationObserver(light).observe(el, {attributes: true, attributeFilter: ['class']}); }
  return () => {};
})()"""


def build_app():
    import gradio as gr
    names = chemical_names()
    ks = k_choices()
    default = DEFAULT_CHEMICAL if DEFAULT_CHEMICAL.lower() in _bundle()[1] else names[0]
    t0 = get_task(default)

    def narrow(vw) -> bool:
        try:
            return 0 < float(vw) < NARROW_PX
        except (TypeError, ValueError):
            return False

    def ui_forecast(chem, k, design, custom, feats, vw):
        try:
            fc = run_forecast(chem, int(k), "custom" if design == "custom levels" else "spread", custom)
        except ValueError as e:
            raise gr.Error(str(e))
        held = [(level_label(x), str(i)) for i, x in enumerate(fc["held_logc"])]
        nar = narrow(vw)
        return ({"fc": fc, "rev": None}, _info_md(fc), _badge_html(fc["abstention"]),
                forecast_figure(fc, None, feats, nar), gr.update(choices=held, value=held[-1][1]),
                trajectory_figure(fc, None, None, nar), None,
                "Press **Reveal held-out wells** to score this forecast against the real wells and the baselines.")

    def ui_reveal(state, feats, conc, vw):
        if not state or not state.get("fc"):
            raise gr.Error("run a forecast first")
        fc = state["fc"]
        rev = reveal(fc)
        li = int(conc) if conc not in (None, "") else None
        nar = narrow(vw)
        return ({"fc": fc, "rev": rev}, forecast_figure(fc, rev, feats, nar), trajectory_figure(fc, rev, li, nar),
                metrics_table(rev), _reveal_md(fc, rev))

    def ui_redraw(state, feats, vw):
        if not state or not state.get("fc"):
            return gr.skip()
        return forecast_figure(state["fc"], state.get("rev"), feats, narrow(vw))

    def ui_heat(state, conc, vw):
        if not state or not state.get("fc"):
            return gr.skip()
        return trajectory_figure(state["fc"], state.get("rev"), int(conc) if conc not in (None, "") else None,
                                 narrow(vw))

    def ui_chem(chem, k):
        t = get_task(chem)
        return gr.update(choices=_custom_choices(t), value=_spread_indices(t, k))

    with gr.Blocks(title="NeuroChip Twin v2", analytics_enabled=False) as app:
        gr.Markdown(
            "# NeuroChip Twin v2\n"
            "Few-shot forecasting of how a chemical perturbs the **development of neural network activity** "
            f"on a multi-electrode array: measure a few concentrations, forecast the whole dose x "
            f"developmental-time response ({len(FEATURES)} network features x DIV {'/'.join(map(str, DIVS))}) "
            "with a 90% band, then reveal the real held-out wells.")
        vw = gr.Number(value=0, visible=False)        # browser width, filled in by VW_JS on every figure event
        with gr.Tabs():
            with gr.Tab("Forecast & reveal"):
                state = gr.State(None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=290):
                        chem = gr.Dropdown(names, value=default, label=f"Chemical ({len(names)}; type to search)",
                                           filterable=True)
                        k = gr.Slider(min(ks), max(ks), value=min(DEFAULT_K, max(ks)), step=1,
                                      label="Measured concentrations k",
                                      info="wells at k concentrations are shown to the model")
                        design = gr.Radio(["spread", "custom levels"], value="spread", label="Design",
                                          info="spread: k levels evenly spaced over the tested range")
                        custom = gr.CheckboxGroup(_custom_choices(t0), value=_spread_indices(t0, DEFAULT_K),
                                                  label="Measured concentrations (custom)", visible=False,
                                                  info=f"k = number of ticked levels (1-{max(ks)}); "
                                                       "at least one level stays hidden")
                        feats = gr.Dropdown([(f.replace("_", " "), f) for f in FEATURES], value=DEFAULT_FEATURES,
                                            multiselect=True, max_choices=6, label="Features to plot")
                        run_btn = gr.Button("Forecast", variant="primary")
                        rev_btn = gr.Button("Reveal held-out wells", variant="secondary")
                        gr.Markdown("<div class='nt-note nt-bounds'>Held-out: the models scoring a chemical were "
                                    "trained without it. Rat cortical cultures in 48-well MEA plates; not a "
                                    "validated twin of any commercial chip; not clinical.</div>")
                    with gr.Column(scale=3, min_width=320):
                        info = gr.Markdown()
                        badge = gr.HTML()
                        plot = gr.Plot(label="Dose-response forecast per developmental day", show_label=False)
                        conc = gr.Dropdown([], label="Heatmap concentration (held out)", filterable=False)
                        heat = gr.Plot(label="Whole trajectory at one held-out concentration", show_label=False)
                        table = gr.Dataframe(label="Error on the held-out concentrations (curve MAE, lower is better)",
                                             interactive=False, wrap=True,
                                             column_widths=["200px", "110px"] + ["76px"] * len(DIVS) + ["150px"])
                        rev_md = gr.Markdown()
                outs = [state, info, badge, plot, conc, heat, table, rev_md]
                ins = [chem, k, design, custom, feats, vw]
                run_btn.click(ui_forecast, ins, outs, js=VW_JS)
                chem.change(ui_chem, [chem, k], custom).then(ui_forecast, ins, outs, js=VW_JS)
                k.release(ui_chem, [chem, k], custom).then(ui_forecast, ins, outs, js=VW_JS)
                design.change(lambda d: gr.update(visible=d == "custom levels"), design, custom)
                rev_btn.click(ui_reveal, [state, feats, conc, vw], [state, plot, heat, table, rev_md], js=VW_JS)
                feats.change(ui_redraw, [state, feats, vw], plot, js=VW_JS)
                conc.input(ui_heat, [state, conc, vw], heat, js=VW_JS)
                app.load(ui_forecast, ins, outs, js=VW_JS)
            with gr.Tab("Results") as res_tab:
                gr.Markdown(results_markdown())
                res_plot = gr.Plot(results_figure(), label="Few-shot forecasting on held-out chemicals",
                                   show_label=False)
                # redrawn for the browser width when the tab is opened (a load event on a component of a
                # hidden tab can stay 'pending' in Gradio 6)
                res_tab.select(lambda w: results_figure(narrow(w)), vw, res_plot, js=VW_JS,
                               show_progress="hidden")
                gr.Dataframe(r2_table(), label="Curve MAE by number of measured concentrations "
                             "(results/trajectory_cv.json)", interactive=False, wrap=True)
                gr.Dataframe(conformal_table(), label="Predictive-interval coverage on held-out wells "
                             "(results/conformal_r7.json)", interactive=False, wrap=True)
                gr.Dataframe(abstention_table(), label="Abstention: error of retained vs abstained forecasts",
                             interactive=False, wrap=True)
                epa = _json("epa_baseline_repro.json") or {}
                gr.Markdown(f"### EPA reference-chemical baseline, reproduced\n{epa.get('method', '')}; "
                            f"{epa.get('n_reference', '-')} reference chemicals ({epa.get('n_positive', '-')} "
                            f"positive / {epa.get('n_negative', '-')} negative). {epa.get('comparison_explanation', '')} "
                            f"*{epa.get('interpretation', '')}*")
                gr.Dataframe(epa_table(), label="results/epa_baseline_repro.json", interactive=False, wrap=True)
            with gr.Tab("Chip layer"):
                gr.Markdown(chip_markdown())
                with gr.Row():
                    for path, cap in chip_images():
                        gr.Image(path, label=cap, show_label=True, interactive=False, min_width=300)
            with gr.Tab("About & limits") as about_tab:
                about = gr.Markdown(about_markdown())
                about_tab.select(about_markdown, None, about,    # refresh the abstention-threshold status
                                 show_progress="hidden")
    return app


def _launch_theme():
    import gradio as gr
    return gr.themes.Default(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
                             font=["system-ui", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
                             font_mono=["Consolas", "Menlo", "monospace"])


def warm_up(folds=None):
    """Load tasks, thresholds and fold models (+ analog-kNN tables) so that interactions are fast."""
    load_thresholds()
    for f in (sorted({t.fold for t in _bundle()[0]}) if folds is None else folds):
        get_models(f)
        get_knn(f)


def _smoke(chemical: str, k: int) -> dict:
    fc = run_forecast(chemical, k)
    rev = reveal(fc)
    return {"chemical": fc["chemical"], "cv_fold": fc["fold"], "model_never_saw_this_chemical": True,
            "k": fc["k"], "measured_uM": [round(10 ** float(x), 4) for x in fc["context_logc"]],
            "held_out_uM": [round(10 ** float(x), 4) for x in fc["held_logc"]],
            "mu_held_shape": list(fc["mu_held"].shape), "mu_grid_shape": list(fc["mu_grid"].shape),
            "band": fc["band_method"], "band_q": round(fc["band_q"], 4),
            "epistemic_score": round(fc["epistemic_score"], 4), "abstention": fc["abstention"],
            "curve_mae": {m: round(v["curve_mae"], 4) for m, v in rev["metrics"].items()},
            "held_out_entries_inside_band": round(rev["band_coverage"], 4), "forecast_runtime_s": fc["runtime_s"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    ap.add_argument("--smoke", action="store_true", help="run the compute core once and print JSON")
    ap.add_argument("--chemical", default=DEFAULT_CHEMICAL)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--build-thresholds", action="store_true", help=f"build {THRESHOLD_FILE} (one-off)")
    ap.add_argument("--out", default=None, help=f"output path for --build-thresholds (default {CACHE_DIR / THRESHOLD_FILE})")
    ap.add_argument("--workers", type=int, default=None, help="processes for --build-thresholds")
    ap.add_argument("--no-auto-thresholds", action="store_true",
                    help="server: do not build missing abstention thresholds in the background "
                         "(also NT_APP_AUTO_THRESHOLDS=0)")
    a = ap.parse_args(argv)
    if a.build_thresholds:
        out = build_abstention_thresholds(a.out, a.workers)
        print(json.dumps({k: out[k] for k in ("epistemic_threshold_by_fold", "conformal_qhat_by_fold",
                                              "reproduction", "published_reference", "n_records",
                                              "build_seconds")}, indent=2))
        return
    if a.smoke:
        print(json.dumps(_smoke(a.chemical, a.k), indent=2))
        return
    t0 = time.time()
    first = get_task(DEFAULT_CHEMICAL).fold if DEFAULT_CHEMICAL.lower() in _bundle()[1] else 0
    warm_up([first])                                                # the page's first forecast
    app = build_app()
    print(f"ready in {time.time() - t0:.1f} s; abstention thresholds: {threshold_status()}", flush=True)
    app.queue(default_concurrency_limit=4).launch(
        server_name=a.host, server_port=a.port, theme=_launch_theme(), css=CSS, js=FORCE_LIGHT_JS,
        allowed_paths=[str(FIGURES)], footer_links=["gradio"], show_error=True, prevent_thread_lock=True)
    threading.Thread(target=warm_up, daemon=True).start()           # other folds, after the server is up
    if not a.no_auto_thresholds and os.environ.get("NT_APP_AUTO_THRESHOLDS", "1") != "0":
        start_threshold_build()                                     # no-op when a valid file exists
    app.block_thread()


if __name__ == "__main__":
    main()
