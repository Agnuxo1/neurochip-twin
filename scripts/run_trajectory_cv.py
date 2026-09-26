"""R2: few-shot dose x DIV trajectory forecasting on unseen chemicals (EPA NFA), 5-fold chemical CV.

Protocol (fixed before evaluation):
- Outer folds: repo/results/splits_epa_nfa.json (all SPIDs of a chemical in one fold).
- For each held-out chemical and k in {0,1,2,3,4} measured concentration levels, the context is
  every well at k levels (5 seeded designs per k) and the targets are all wells at the other levels.
  Every method receives exactly the same context.
- Primary metric, CURVE MAE: for each held-out level, |prediction - mean of its replicate wells|
  over the observed (DIV, feature) entries; averaged per chemical, then over chemicals.
  Secondary metric, WELL MAE: the same against every individual well (dominated by replicate noise).
- Model selection is nested: inside each outer training set, one inner fold is held out to choose
  the likelihood (gaussian | laplace) and the decoder (plain | interpolation-informed), 4000 steps fixed a priori; the chosen configuration is then
  retrained on the whole outer training set with 3 seeds (ensemble). Outer test chemicals are never
  used for any choice.
- Uncertainty: paired bootstrap over chemicals of (NeuroTrajectory - baseline).

Usage: python repo/scripts/run_trajectory_cv.py [--seeds 0 1 2] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
from neurotwin.models.trajectory import (  # noqa: E402
    AnalogKNN, level_means, load_tasks, predict_ctx_mean, predict_hill, predict_interp,
    predict_zero, split_context)
from neurotwin.models.cnp import Z90, TrainConfig, predict, train_cnp  # noqa: E402

KS = [0, 1, 2, 3, 4]
N_DESIGNS = 5
GRIDS = {
    "v1": [{"likelihood": lk, "steps": 4000, "use_interp": ui, "n_freq": 8}
           for ui in (False, True) for lk in ("gaussian", "laplace")],
    # v2 adds a smoothness-in-dose hyperparameter (number of Fourier frequencies of the dose embedding)
    "v2": [{"likelihood": lk, "steps": 4000, "use_interp": True, "n_freq": nf}
           for nf in (3, 8) for lk in ("gaussian", "laplace")],
}
GRID = GRIDS["v1"]
BASELINES = {"zero": predict_zero, "context_mean": predict_ctx_mean,
             "loglinear_interp": predict_interp, "hill_per_endpoint": predict_hill}


def designs(task, k, n=N_DESIGNS):
    seed = int(hashlib.sha256(f"{task.chem}|{k}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for _ in range(n * 4):
        lv = tuple(sorted(rng.choice(task.levels, size=k, replace=False).tolist())) if k else ()
        if lv not in seen:
            seen.add(lv)
            out.append(np.array(lv, np.float32))
        if len(out) == n or k == 0:
            break
    return out


def curve_and_well_mae(pred_levels, task, it):
    """pred_levels: (L, ND, NF) prediction at the held-out levels (sorted, as level_means)."""
    lv, mu, ok = level_means(task, it)
    curve = float(np.abs(pred_levels - mu)[ok].mean()) if ok.any() else float("nan")
    idx = np.searchsorted(lv, task.logc[it])
    pw = pred_levels[idx]
    mm = task.m[it]
    well = float(np.abs(pw - task.y[it])[mm].mean()) if mm.any() else float("nan")
    return curve, well


def paired_boot(a, b, n=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "rel_change_pct": round(float(100 * d.mean() / b[ok].mean()), 2),
            "n_chemicals": int(ok.sum()), "frac_chem_improved": round(float((d < 0).mean()), 3)}


def cnp_curve_mae(models, task, ic, it, dev):
    lv = np.unique(task.logc[it])
    preds = [predict(m, task, ic, lv, device=dev) for m in models]
    mu = np.mean([p[0] for p in preds], 0)
    return mu, preds


def select_config(train, f, folds, dev, log):
    """Nested choice on an inner fold of the outer training set (never the outer test fold)."""
    inner = folds[(folds.index(f) + 1) % len(folds)]
    tr_in = [t for t in train if t.fold != inner]
    va_in = [t for t in train if t.fold == inner]
    scores = []
    for cfg in GRID:
        m = train_cnp(tr_in, TrainConfig(steps=cfg["steps"], likelihood=cfg["likelihood"], use_interp=cfg["use_interp"], n_freq=cfg.get("n_freq", 8), seed=100), device=dev)
        errs = []
        for t in va_in:
            for k in (1, 2, 3):
                if k >= len(t.levels):
                    continue
                for ctx_lv in designs(t, k, n=2):
                    ic, it = split_context(t, ctx_lv)
                    mu, _ = cnp_curve_mae([m], t, ic, it, dev)
                    errs.append(curve_and_well_mae(mu, t, it)[0])
        scores.append(float(np.nanmean(errs)))
        log(f"  inner fold {inner}: {cfg} -> curve MAE {scores[-1]:.4f}")
    best = GRID[int(np.argmin(scores))]
    return best, {"inner_fold": inner, "grid": GRID, "inner_curve_mae": scores, "chosen": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="1 fold, 1 seed, no nesting (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "repo/results"))
    ap.add_argument("--folds", type=int, nargs="*", help="run only these outer folds (parallel workers)")
    ap.add_argument("--summarize", action="store_true", help="merge per-fold outputs and write the summary")
    ap.add_argument("--grid", default="v1", choices=sorted(GRIDS))
    ap.add_argument("--models-dir", default=str(ROOT / "data/processed/models"))
    ap.add_argument("--tag", default="", help="suffix for outputs (e.g. _v2)")
    a = ap.parse_args()
    global GRID
    GRID = GRIDS[a.grid]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = ROOT / "data/processed/epa_nfa/tasks_cache.pkl"
    tasks, tr = pickle.load(open(cache, "rb")) if cache.exists() else load_tasks(ROOT)
    folds = sorted({t.fold for t in tasks})
    eval_folds = folds[:1] if a.quick else (a.folds if a.folds else folds)
    seeds = a.seeds[:1] if a.quick else a.seeds
    models_dir = Path(a.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    tag = "quick" if a.quick else ("f" + "".join(map(str, eval_folds)))
    parts = out / f"trajectory_cv_parts{a.tag}"
    parts.mkdir(exist_ok=True)
    if a.summarize:
        return summarize(out, parts, seeds, name=f"trajectory_cv{a.tag}.json")
    logf = open(parts / f"log_{tag}.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    rows, calib, chosen = [], [], {}
    t0 = time.time()
    for f in eval_folds:
        train = [t for t in tasks if t.fold != f]
        test = [t for t in tasks if t.fold == f]
        if a.quick:
            cfg, info = {"likelihood": "gaussian", "steps": 800, "use_interp": True}, {"note": "smoke test, no nesting"}
        else:
            cfg, info = select_config(train, f, folds, dev, log)
        chosen[str(f)] = info
        knn = AnalogKNN(train)
        models = []
        for s in seeds:
            m = train_cnp(train, TrainConfig(steps=cfg["steps"], likelihood=cfg["likelihood"], use_interp=cfg["use_interp"], n_freq=cfg.get("n_freq", 8), seed=s), device=dev)
            torch.save({"state": m.state_dict(), "cfg": cfg}, models_dir / f"cnp_fold{f}_seed{s}.pt")
            models.append(m)
        log(f"fold {f}: config {cfg}; {len(models)} models trained ({time.time()-t0:.0f}s)")
        z90 = Z90[cfg["likelihood"]]
        for t in test:
            for k in KS:
                if k >= len(t.levels):
                    continue
                acc = {}
                cov, width = [], []
                for ctx_lv in designs(t, k):
                    ic, it = split_context(t, ctx_lv)
                    lv = np.unique(t.logc[it])
                    for name, fn in BASELINES.items():
                        acc.setdefault(name, []).append(curve_and_well_mae(fn(t, ic, lv), t, it))
                    acc.setdefault("analog_knn", []).append(curve_and_well_mae(knn.predict(t, ic, lv), t, it))
                    mu, preds = cnp_curve_mae(models, t, ic, it, dev)
                    acc.setdefault("neurotrajectory", []).append(curve_and_well_mae(mu, t, it))
                    for s, p in zip(seeds, preds):
                        acc.setdefault(f"neurotrajectory_seed{s}", []).append(curve_and_well_mae(p[0], t, it))
                    # predictive interval check on individual wells (before conformal calibration)
                    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + np.var([p[0] for p in preds], 0))
                    idx = np.searchsorted(lv, t.logc[it])
                    mm = t.m[it]
                    inside = np.abs(t.y[it] - mu[idx]) <= z90 * sc[idx]
                    cov.append(float(inside[mm].mean()))
                    width.append(float((2 * z90 * sc[idx])[mm].mean()))
                for name, v in acc.items():
                    v = np.array(v, float)
                    rows.append({"chemical": t.chem, "fold": f, "k": k, "method": name, "label": t.label,
                                 "curve_mae": float(np.nanmean(v[:, 0])), "well_mae": float(np.nanmean(v[:, 1]))})
                calib.append({"chemical": t.chem, "k": k, "coverage90_well": float(np.mean(cov)),
                              "width90_well": float(np.mean(width))})
        log(f"fold {f}: evaluated {len(test)} chemicals ({time.time()-t0:.0f}s)")

    (parts / f"rows_{tag}.json").write_text(json.dumps({"rows": rows, "calib": calib, "chosen": chosen,
                                                       "elapsed_s": round(time.time() - t0, 1), "device": dev}), encoding="utf-8")
    log(f"fold part {tag} done")
    if a.quick:
        summarize(out, parts, seeds, only=f"rows_{tag}.json", name="trajectory_cv_quick.json")


def summarize(out, parts, seeds, only=None, name="trajectory_cv.json"):
    import pandas as pd
    rows, calib, chosen, el = [], [], {}, 0.0
    files = [parts / only] if only else sorted(p for p in parts.glob("rows_f*.json"))
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        rows += d["rows"]; calib += d["calib"]; chosen.update(d["chosen"]); el += d["elapsed_s"]
    df = pd.DataFrame(rows)
    df.to_csv(out / name.replace(".json", "_per_chemical.csv"), index=False)
    cal = pd.DataFrame(calib)
    summary = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(),
               "n_chemicals": int(df.chemical.nunique()), "outer_folds": sorted(df.fold.unique().tolist()),
               "seeds": seeds, "designs_per_k": N_DESIGNS, "nested_selection": chosen,
               "development_note": ("During development one smoke run (800 steps, 1 seed) was evaluated on outer "
                                    "fold 0 and motivated adding the interpolation-informed decoder as a candidate; "
                                    "the final choice is nested. 'by_k_folds_1to4' excludes fold 0 entirely."),
               "target_units": ("asinh(x/c_f) minus plate/date/DIV vehicle median, divided by pooled "
                                "within-plate vehicle robust SD; clipped to +-10"),
               "by_k": {}}
    base_names = list(BASELINES) + ["analog_knn"]
    summary["by_k_folds_1to4"] = {}
    for k in KS:
        d4 = df[(df.k == k) & (df.fold != 0)].pivot_table(index="chemical", columns="method", values="curve_mae")
        if not d4.empty:
            b4 = min(base_names, key=lambda c: d4[c].mean())
            summary["by_k_folds_1to4"][str(k)] = {"best_baseline": b4, "neurotrajectory": round(float(d4["neurotrajectory"].mean()), 4),
                                                   "baseline": round(float(d4[b4].mean()), 4),
                                                   "paired_vs_best": paired_boot(d4["neurotrajectory"], d4[b4])}
    for k in KS:
        entry = {}
        for metric in ("curve_mae", "well_mae"):
            dk = df[df.k == k].pivot_table(index="chemical", columns="method", values=metric)
            if dk.empty:
                continue
            best = min(base_names, key=lambda c: dk[c].mean())
            entry[metric] = {"mean": {c: round(float(v), 4) for c, v in dk.mean().items()},
                             "best_baseline": best,
                             "paired_vs_best": paired_boot(dk["neurotrajectory"], dk[best]),
                             "paired_vs_all": {b: paired_boot(dk["neurotrajectory"], dk[b]) for b in base_names}}
        ck = cal[cal.k == k]
        entry["well_interval90"] = {"coverage_mean": round(float(ck.coverage90_well.mean()), 4),
                                    "width_mean": round(float(ck.width90_well.mean()), 4)}
        summary["by_k"][str(k)] = entry
        c = entry["curve_mae"]
        print(f"k={k} curve MAE: " + ", ".join(f"{m}={v:.3f}" for m, v in sorted(c["mean"].items(), key=lambda x: x[1])
                                             if not m.startswith("neurotrajectory_seed")))
        print(f"   vs best ({c['best_baseline']}): {c['paired_vs_best']}; interval90 {entry['well_interval90']}")
    summary["elapsed_s_sum_workers"] = round(el, 1)
    (out / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
