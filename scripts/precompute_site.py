"""Precompute the data and images of the static project site (site/) from the shipped bundle and results.

    python scripts/precompute_site.py                 # all 243 chemicals, k = 1, 2, 3 (CPU, a few minutes)
    python scripts/precompute_site.py --limit 5       # quick smoke run on the first 5 chemicals
    python scripts/precompute_site.py --assets-only   # only copy figures and rebuild summary.json (seconds)

Runs offline on CPU from data_bundle/ and results/ only. For every chemical it uses the NeuroTrajectory
ensemble of that chemical's cross-validation fold (trained WITHOUT it), gives the model the wells at k
evenly spread concentrations (the same 'spread' design as demo.py), and stores:

  * the forecast mean and a cross-conformal 90 % half-width on a 32-point log10-dose grid (17 features x 4 DIV).
    Conformal rule (as results/conformal_r7.json, without the cytotoxicity strata, which are not in the bundle):
    nonconformity s = |y - mu| / scale per held-out well entry; for a chemical of fold g and k measured
    concentrations, q_hat = finite-sample 90 % quantile of the scores of the chemicals of the OTHER folds at the
    same k; half-width = q_hat * scale. The parametric band is q_hat-free: z90 * scale (ratio stored in index.json).
  * the forecast at the held-out concentrations, the measured level means and the individual wells,
  * the curve MAE of NeuroTrajectory, log-linear interpolation and analog kNN at the held-out levels, the share of
    held-out well entries inside the conformal and parametric bands, and the epistemic score with the abstention
    flag (rule of results/conformal_r7.json: abstain above the 90th percentile of the scores of the OTHER folds),
  * a DNT hazard probability per k (the classifier of scripts/run_dnt_r4.py: L2 logistic regression on
    developmental-summary features of all measured concentrations of the reference chemicals of the OTHER folds,
    applied to the features of the forecast); k = 3 and 'all doses' are the protocols validated in
    results/dnt_r4.json and are checked here against results/dnt_r4_calls.csv,
  * the two DIV 7 gates of results/r11_early_exit.json (persistence = throughput gate, twin upper 90 % band =
    safety gate, plus the primary twin score), copied per chemical from results/r11_early_exit_rows.csv
    (folds 1-4 only) and re-checked against the per-fold thresholds.

Arrays are stored as integers x100 (two decimals) in [feature][DIV][point] order, null = unobserved.
It also writes site/data/summary.json (headline numbers and key-number cards copied from results/*.json, each
with its source path, and SHA-256 hashes of the sources) and copies the report figures into site/assets/img/.
No number shown by the site is typed by hand: everything is read from these files at runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
SITE = REPO / "site"
KS = [1, 2, 3]
N_GRID = 32
HZ_GRID = 120      # grid of scripts/run_dnt_r4.py
ALPHA = 0.1
Q = 100  # stored value = round(x * Q)

# report figures copied into the site: (source relative to repo/, published name)
FIGURES = [
    ("report/figures/visuals/graphical_abstract.png", "graphical_abstract.png"),
    ("report/figures/visuals/graphical_abstract_16x9.png", "graphical_abstract_16x9.png"),
    ("report/figures/visuals/real_network_raster.png", "real_network_raster.png"),
    ("report/figures/visuals/network_development.png", "network_development.png"),
    ("report/figures/visuals/twin_in_action.png", "twin_in_action.png"),
    ("report/figures/fig1_system.png", "fig1_system.png"),
    ("report/figures/fig2_data.png", "fig2_data.png"),
    ("report/figures/fig3_forecast_main.png", "fig3_forecast_main.png"),
    ("report/figures/fig5_calibration.png", "fig5_calibration.png"),
    ("report/figures/fig6_hazard.png", "fig6_hazard.png"),
    ("report/figures/fig7_dosecompass.png", "fig7_dosecompass.png"),
    ("report/figures/fig8_chip.png", "fig8_chip.png"),
    ("report/figures/fig10_ablations_failures.png", "fig10_ablations_failures.png"),
]

FEATURE_LABELS = {
    "firing_rate_mean": "Mean firing rate",
    "burst_rate": "Burst rate",
    "per_burst_interspike_interval": "Interspike interval within bursts",
    "per_burst_spike_percent": "% spikes in bursts",
    "burst_duration_mean": "Burst duration",
    "interburst_interval_mean": "Interburst interval",
    "active_electrodes_number": "Active electrodes",
    "bursting_electrodes_number": "Bursting electrodes",
    "network_spike_number": "Network spikes",
    "network_spike_peak": "Network spike peak",
    "spike_duration_mean": "Network spike duration",
    "per_network_spike_spike_percent": "% spikes in network spikes",
    "inter_network_spike_interval_mean": "Inter-network-spike interval",
    "network_spike_duration_std": "Network spike duration SD",
    "per_network_spike_spike_number_mean": "Spikes per network spike",
    "correlation_coefficient_mean": "Mean correlation",
    "mutual_information_norm": "Normalised mutual information",
}
FEATURE_GROUPS = {
    "Activity": ["firing_rate_mean", "active_electrodes_number"],
    "Bursting": ["burst_rate", "per_burst_interspike_interval", "per_burst_spike_percent",
                 "burst_duration_mean", "interburst_interval_mean", "bursting_electrodes_number"],
    "Network": ["network_spike_number", "network_spike_peak", "spike_duration_mean",
                "per_network_spike_spike_percent", "inter_network_spike_interval_mean",
                "network_spike_duration_std", "per_network_spike_spike_number_mean"],
    "Connectivity": ["correlation_coefficient_mean", "mutual_information_norm"],
}
DEFAULT_FEATURES = ["firing_rate_mean", "network_spike_number", "active_electrodes_number",
                    "correlation_coefficient_mean"]
METHOD_LABELS = {"neurotrajectory": "NeuroTrajectory", "loglinear_interp": "Log-linear interpolation",
                 "analog_knn": "Analog kNN", "hill_per_endpoint": "Hill per endpoint",
                 "context_mean": "Context mean", "zero": "Zero effect"}


# ------------------------------------------------------------------ helpers
def q_arr(a: np.ndarray, ok: np.ndarray | None = None):
    """(P, ND, NF) float -> nested list [NF][ND][P] of int x100, None where not ok."""
    v = np.rint(np.asarray(a, np.float64) * Q).astype(np.int64)
    v = np.transpose(v, (2, 1, 0))
    if ok is None:
        return v.tolist()
    okt = np.transpose(np.asarray(ok, bool), (2, 1, 0))
    return [[[int(x) if o else None for x, o in zip(row, orow)] for row, orow in zip(fm, fo)]
            for fm, fo in zip(v, okt)]


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


_JCACHE: dict[str, dict] = {}


def load_json(name: str) -> dict:
    if name not in _JCACHE:
        _JCACHE[name] = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    return _JCACHE[name]


def jget(name: str, path: str):
    """Dotted path into results/<name> (dict keys, list indices). Keys that contain dots themselves
    (e.g. 'alpha_0.1', 'AUC.3hit') are matched greedily, longest existing key first."""
    obj = load_json(name)
    parts = path.split(".")
    while parts:
        if isinstance(obj, list):
            obj, parts = obj[int(parts[0])], parts[1:]
            continue
        for j in range(len(parts), 0, -1):
            key = ".".join(parts[:j])
            if key in obj:
                obj, parts = obj[key], parts[j:]
                break
        else:
            raise KeyError(f"{name}:{path}")
    return obj


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


# ------------------------------------------------------------------ per-chemical forecasts
def ensemble(models, task, is_ctx, q):
    """Same combination as demo.forecast, plus the member spread used for the epistemic score."""
    from neurotwin.models.cnp import predict
    preds = [predict(m, task, is_ctx, q, device="cpu") for m in models]
    mus = np.stack([p[0] for p in preds])
    mu = mus.mean(0)
    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + mus.var(0))
    return mu, sc, mus.std(0)


def hazard_features(grid, traj):
    """Features of scripts/run_dnt_r4.py: max |A_f|, signed A_f at the top concentration, BMC per feature."""
    from neurotwin.models.potency import bmc_from_curve, dev_summary
    A = dev_summary(traj)
    bmc, _ = bmc_from_curve(grid, A)
    bmc = np.where(np.isfinite(bmc), bmc, grid.max() + 1.0)
    return np.concatenate([np.abs(A).max(0), A[-1], bmc])


def chemical_records(tasks, limit=None):
    from demo import load_fold_models, spread_design
    from neurotwin.models.trajectory import AnalogKNN, interp_rows, level_means, predict_interp, split_context
    by_fold = {}
    for i, t in enumerate(tasks):
        by_fold.setdefault(t.fold, []).append(i)
    keep = set(range(len(tasks)) if limit is None else range(limit))
    out, fold_info = {}, {}
    for f in sorted(by_fold):
        idx = [i for i in by_fold[f] if i in keep]
        if not idx:
            continue
        models, z90 = load_fold_models(f)
        knn = AnalogKNN([t for t in tasks if t.fold != f])
        fold_info[f] = {"n_models": len(models), "z90": round(float(z90), 4),
                        "n_test_chemicals": len(by_fold[f]), "n_train_chemicals": len(tasks) - len(by_fold[f])}
        t0 = time.time()
        for i in idx:
            t = tasks[i]
            grid = np.linspace(t.levels.min() - 0.3, t.levels.max() + 0.3, N_GRID).astype(np.float32)
            hgrid = np.linspace(t.levels.min(), t.levels.max(), HZ_GRID).astype(np.float32)
            rec = {"t": t, "grid": grid, "z90": float(z90), "k": {},
                   "hz_full": hazard_features(hgrid, interp_rows(t.levels, t._mu, t._ok, hgrid))}
            for k in KS:
                kk = min(k, len(t.levels) - 1)
                ctx_lv = spread_design(t.levels, kk)
                is_ctx, is_tgt = split_context(t, ctx_lv)
                held = np.unique(t.logc[is_tgt])
                mu_h, sc_h, sd_h = ensemble(models, t, is_ctx, held)
                mu_g, sc_g, _ = ensemble(models, t, is_ctx, grid)
                mu_z, _, _ = ensemble(models, t, is_ctx, hgrid)
                lv, obs, ok = level_means(t, is_tgt)
                assert np.allclose(lv, held)
                pi = predict_interp(t, is_ctx, held)
                pk = knn.predict(t, is_ctx, held)
                err = {"model": np.abs(mu_h - obs), "interp": np.abs(pi - obs), "knn": np.abs(pk - obs)}
                curve = {m: float(e[ok].mean()) for m, e in err.items()}
                per_level = {m: [r(float(e[j][ok[j]].mean())) if ok[j].any() else None for j in range(len(held))]
                             for m, e in err.items()}
                wi = np.searchsorted(held, t.logc[is_tgt])
                scores = (np.abs(t.y[is_tgt] - mu_h[wi]) / sc_h[wi])[t.m[is_tgt]]
                rec["k"][k] = {
                    "k_used": int(len(ctx_lv)),
                    "ctx": np.searchsorted(t.levels, ctx_lv).tolist(),
                    "held": np.searchsorted(t.levels, held).tolist(),
                    "mu": mu_g, "sc": sc_g, "mu_held": mu_h, "sc_held": sc_h,
                    "curve_mae": curve, "level_mae": per_level, "scores": scores.astype(np.float64),
                    "coverage_param": float((scores <= z90).mean()) if scores.size else float("nan"),
                    "epi": float(sd_h.mean() / sc_h.mean()),
                    "hz": hazard_features(hgrid, mu_z),
                }
            out[i] = rec
        print(f"fold {f}: {len(idx)} chemicals, {len(models)} models, z90={z90:.3f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    return out, fold_info


def conformal(records, tasks):
    """Cross-conformal q_hat per (fold, k) from the scores of the chemicals of the OTHER folds at the same k."""
    folds = sorted({tasks[i].fold for i in records})
    qhat = {}
    for f in folds:
        for k in KS:
            cal = [rec["k"][k]["scores"] for i, rec in records.items() if tasks[i].fold != f]
            cal = np.concatenate(cal) if cal else np.array([])
            qhat[(f, k)] = conformal_q(cal) if cal.size else float("nan")
    cov = {k: [] for k in KS}
    for i, rec in records.items():
        for k in KS:
            e = rec["k"][k]
            q = qhat[(tasks[i].fold, k)]
            e["qhat"] = q
            e["coverage90"] = float((e["scores"] <= q).mean()) if e["scores"].size else float("nan")
            cov[k].append((e["coverage90"], e["coverage_param"]))
    by_k = {str(k): {"conformal_coverage": r(np.nanmean([c for c, _ in v]), 4),
                     "parametric_coverage": r(np.nanmean([p for _, p in v]), 4),
                     "n_chemicals": len(v)} for k, v in cov.items()}
    return {f"{f}|{k}": r(v, 4) for (f, k), v in qhat.items()}, by_k


def abstention(records, tasks):
    """Cross-fitted threshold per fold: 90th percentile of the epistemic scores of the OTHER folds
    (all k pooled), the rule of results/conformal_r7.json applied to the site's spread design."""
    folds = sorted({tasks[i].fold for i in records})
    thr = {}
    for f in folds:
        cal = [rec["k"][k]["epi"] for i, rec in records.items() if tasks[i].fold != f for k in KS]
        thr[f] = float(np.percentile(cal, 90)) if cal else float("nan")
    rows = []
    for i, rec in records.items():
        for k in KS:
            e = rec["k"][k]
            e["threshold"] = thr[tasks[i].fold]
            e["abstain"] = bool(np.isfinite(e["threshold"]) and e["epi"] > e["threshold"])
            rows.append((k, e["abstain"], e["curve_mae"]["model"]))
    by_k = {}
    for k in KS:
        a = [(ab, m) for kk, ab, m in rows if kk == k]
        ret = [m for ab, m in a if not ab]
        abs_ = [m for ab, m in a if ab]
        by_k[str(k)] = {"rate": r(len(abs_) / len(a), 4), "mae_retained": r(np.mean(ret) if ret else np.nan, 4),
                        "mae_abstained": r(np.mean(abs_) if abs_ else np.nan, 4)}
    return {str(f): r(v, 4) for f, v in thr.items()}, by_k


def rates(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, fn = int((y & yhat).sum()), int((y & ~yhat).sum())
    tn, fp = int((~y & ~yhat).sum()), int((~y & yhat).sum())
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": r(se, 4), "specificity": r(sp, 4),
            "balanced_accuracy": r((se + sp) / 2, 4)}


def hazard(records, tasks):
    """R4 classifier per fold (trained on all-dose features of the reference chemicals of the other folds),
    applied to the all-dose and the k-forecast features of every chemical of the fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    all_lab = [(i, t) for i, t in enumerate(tasks) if t.label in ("positive", "negative")]
    # training features need every reference chemical, even outside --limit
    need = {i for i, _ in all_lab} - set(records)
    if need:
        from neurotwin.models.trajectory import interp_rows
        extra = {}
        for i in need:
            t = tasks[i]
            g = np.linspace(t.levels.min(), t.levels.max(), HZ_GRID).astype(np.float32)
            extra[i] = hazard_features(g, interp_rows(t.levels, t._mu, t._ok, g))
    else:
        extra = {}
    xfull = {i: (records[i]["hz_full"] if i in records else extra[i]) for i, _ in all_lab}
    variants = ["full"] + [str(k) for k in KS]
    probs = {}
    for f in sorted({tasks[i].fold for i in records}):
        tr = [i for i, t in all_lab if t.fold != f]
        X = np.stack([xfull[i] for i in tr])
        y = np.array([tasks[i].label == "positive" for i in tr])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        clf.fit(X, y)
        for i in [i for i in records if tasks[i].fold == f]:
            rec = records[i]
            feats = {"full": rec["hz_full"], **{str(k): rec["k"][k]["hz"] for k in KS}}
            probs[i] = {v: float(clf.predict_proba(feats[v][None])[0, 1]) for v in variants}
    lab_i = [i for i, _ in all_lab if i in probs]
    y = [tasks[i].label == "positive" for i in lab_i]
    summary = {v: rates(y, [probs[i][v] >= 0.5 for i in lab_i]) for v in variants}
    # check against the validated calls of results/dnt_r4.json (k = 3 spread design and all doses)
    check = {}
    calls_csv = RESULTS / "dnt_r4_calls.csv"
    if calls_csv.exists():
        ref = {row["chemical"]: row for row in csv.DictReader(calls_csv.open(encoding="utf-8"))}
        for v, col in (("full", "pred_x_full"), ("3", "pred_x_k3")):
            pairs = [((probs[i][v] >= 0.5), ref[tasks[i].chem][col] == "True") for i in lab_i if tasks[i].chem in ref]
            check[v] = {"n": len(pairs), "agree": int(sum(a == b for a, b in pairs))}
    return probs, summary, check


def gates(tasks):
    """Per-chemical DIV 7 gates from results/r11_early_exit_rows.csv, re-checked against the fold thresholds."""
    rows_p = RESULTS / "r11_early_exit_rows.csv"
    r11 = load_json("r11_early_exit.json")
    out, mismatches = {}, 0
    names = {t.chem for t in tasks}
    for row in csv.DictReader(rows_p.open(encoding="utf-8")):
        if row["chemical"] not in names:
            continue
        f = row["fold"]
        g = {"fold": int(f), "epa_label": row["epa_label"], "reference_score": r(float(row["reference_score"])),
             "reference_active": row["reference_active"] == "True"}
        for m in ("persistence", "twin_upper90", "twin"):
            s = float(row[m])
            thr = float(r11["methods"][m]["thresholds_by_fold"][f])
            ex = s < thr
            mismatches += int(ex != (row[f"exit_{m}"] == "True"))
            g[m] = {"score": r(s), "threshold": r(thr, 4), "exit": bool(row[f"exit_{m}"] == "True")}
        out[row["chemical"]] = g
    return out, mismatches


# ------------------------------------------------------------------ writers
def write_chemicals(records, fold_info, thresholds, abst_by_k, qhat, conf_by_k, hz_probs, hz_summary, hz_check,
                    gate_rows, gate_mismatch, out: Path):
    (out / "chem").mkdir(parents=True, exist_ok=True)
    index = []
    for i in sorted(records):
        rec = records[i]
        t = rec["t"]
        cid = f"c{i:03d}"
        L = len(t.levels)
        counts = np.bincount(np.searchsorted(t.levels, t.logc), minlength=L)
        hp = hz_probs.get(i)
        g = gate_rows.get(t.chem)
        doc = {
            "id": cid, "chem": t.chem, "fold": t.fold, "label": t.label, "n_wells": int(len(t.logc)),
            "levels": [r(x, 4) for x in t.levels], "level_n": counts.tolist(),
            "level_mean": q_arr(t._mu, t._ok),
            "wells": {"lv": np.searchsorted(t.levels, t.logc).tolist(), "y": q_arr(t.y, t.m)},
            "grid": [r(x, 4) for x in rec["grid"]], "k": {},
            "hazard": None if hp is None else {"prob": {v: r(p, 3) for v, p in hp.items()},
                                               "call": {v: bool(p >= 0.5) for v, p in hp.items()}},
            "gate": g,
        }
        summ = {}
        for k, e in rec["k"].items():
            doc["k"][str(k)] = {
                "k_used": e["k_used"], "ctx": e["ctx"], "held": e["held"],
                "mu": q_arr(e["mu"]), "hw": q_arr(e["qhat"] * e["sc"]),
                "mu_held": q_arr(e["mu_held"]), "hw_held": q_arr(e["qhat"] * e["sc_held"]),
                "level_mae": e["level_mae"],
                "metrics": {"curve_mae": {m: r(v) for m, v in e["curve_mae"].items()},
                            "coverage90": r(e["coverage90"]), "coverage90_param": r(e["coverage_param"]),
                            "qhat": r(e["qhat"], 4), "z90": r(rec["z90"], 4),
                            "epi": r(e["epi"], 4), "threshold": r(e["threshold"], 4), "abstain": e["abstain"]},
            }
            summ[str(k)] = {"m": r(e["curve_mae"]["model"]), "i": r(e["curve_mae"]["interp"]),
                            "n": r(e["curve_mae"]["knn"]), "c": r(e["coverage90"]), "a": int(e["abstain"])}
        (out / "chem" / f"{cid}.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        index.append({"id": cid, "chem": t.chem, "fold": t.fold, "label": t.label, "n_levels": L,
                      "n_wells": int(len(t.logc)), "k": summ,
                      "hz": None if hp is None else {v: r(p, 3) for v, p in hp.items()},
                      "gate": None if g is None else {"p": int(g["persistence"]["exit"]),
                                                      "s": int(g["twin_upper90"]["exit"])}})
    f4 = load_json("figures_derived.json").get("fig4", {})
    examples = [{"chemical": f4[q]["chemical"], "quantile": f4[q]["quantile"], "role": role}
                for q, role in (("p90", "strong gain (90th percentile)"), ("median", "typical gain (median)"),
                                ("p10", "model worse than interpolation (10th percentile)"))
                if isinstance(f4.get(q), dict)]
    r11 = load_json("r11_early_exit.json")
    meta = {
        "features": list(FEATURE_LABELS), "feature_labels": FEATURE_LABELS, "feature_groups": FEATURE_GROUPS,
        "default_features": DEFAULT_FEATURES, "divs": [5, 7, 9, 12], "ks": KS, "n_grid": N_GRID, "scale": Q,
        "design": "spread: k concentration levels evenly spaced over the tested range (demo.py spread_design)",
        "units": "effect vs vehicle in robust-SD units of asinh(x/c_f) (0 = vehicle; see transform.json)",
        "band": {"level": 1 - ALPHA,
                 "rule": "cross-conformal: q_hat = finite-sample 90 % quantile of |y - mu| / scale over the held-out "
                         "well entries of the chemicals of the other folds at the same k; half-width = q_hat * scale "
                         "(results/conformal_r7.json additionally stratifies by cytotoxicity regime)",
                 "qhat_by_fold_k": qhat, "site_coverage_by_k": conf_by_k},
        "folds": {str(f): v for f, v in fold_info.items()},
        "abstention": {"rule": "abstain if epistemic score > 90th percentile of the scores of the chemicals "
                               "in the other folds (k = 1..3 pooled), as in results/conformal_r7.json",
                       "thresholds_by_fold": thresholds, "site_design_by_k": abst_by_k},
        "hazard": {"rule": "scripts/run_dnt_r4.py classifier: L2 logistic regression (C = 1, balanced) on developmental-"
                           "summary features of all measured doses of the EPA reference chemicals of the other folds; "
                           "call = probability >= 0.5",
                   "validated": ["full", "3"],
                   "site_rates_on_reference_chemicals": hz_summary, "check_vs_results_dnt_r4_calls": hz_check},
        "gate": {"source": "results/r11_early_exit_rows.csv", "cohort": "outer folds 1-4 (194 chemicals)",
                 "reference": "activity rule on all DIVs and concentrations (BMR = 3 vehicle robust SD)",
                 "threshold_rule": "exit if score < fold threshold (95 % sensitivity on the other three folds)",
                 "gates": {"persistence": "throughput gate: DIV 7 network carried forward",
                           "twin_upper90": "safety gate: upper 90 % band of the twin forecast of DIV 9/12",
                           "twin": "primary twin score (preregistered R11 primary)"},
                 "methods": {m: {kk: r11["methods"][m][kk] for kk in ("sensitivity", "inactive_exit_rate",
                                                                      "overall_exit_rate")}
                             for m in ("persistence", "twin_upper90", "twin")},
                 "recheck_mismatches": gate_mismatch},
        "examples": examples,
        "chemicals": index,
    }
    (out / "index.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    return index


def site_design_means(index):
    res = {}
    for k in KS:
        rows = [c["k"][str(k)] for c in index]
        res[str(k)] = {"n_chemicals": len(rows),
                       "curve_mae": {"neurotrajectory": r(np.mean([x["m"] for x in rows]), 4),
                                     "loglinear_interp": r(np.mean([x["i"] for x in rows]), 4),
                                     "analog_knn": r(np.mean([x["n"] for x in rows]), 4)},
                       "coverage90_mean": r(np.nanmean([x["c"] for x in rows if x["c"] is not None]), 4),
                       "frac_better_than_interp": r(np.mean([x["m"] < x["i"] for x in rows]), 3),
                       "frac_better_than_knn": r(np.mean([x["m"] < x["n"] for x in rows]), 3)}
    return res


def key_numbers():
    """Key-number cards. Every value is read from results/*.json; `sources` lists file:path for each one."""
    S = []

    def src(name, path):
        S.append(f"{name}:{path}")
        return jget(name, path)

    cards = []
    # 1. forecast
    S.clear()
    p = "by_k.3.curve_mae.paired_vs_best"
    rel = -src("trajectory_cv.json", p + ".rel_change_pct")
    lo, hi = src("trajectory_cv.json", p + ".ci95")
    frac = src("trajectory_cv.json", p + ".frac_chem_improved")
    n = src("trajectory_cv.json", p + ".n_chemicals")
    best = METHOD_LABELS.get(src("trajectory_cv.json", "by_k.3.curve_mae.best_baseline"), "best baseline")
    rel4 = -src("trajectory_cv.json", "by_k_folds_1to4.3.paired_vs_best.rel_change_pct")
    cards.append({"id": "forecast", "value": f"{rel:.0f}%", "title": "lower forecast error from 3 measured doses",
                  "detail": f"Than the best of five executed baselines ({best.lower()}); paired 95 % CI of the "
                            f"difference [{lo:.3f}, {hi:.3f}] SD units. Better for {100 * frac:.0f} % of {n} unseen "
                            f"chemicals; {rel4:.1f} % on the four never-inspected folds.",
                  "sources": list(S)})
    # 2. hazard
    S.clear()
    b = "variants.k3.mcnemar_vs_epa"
    ours = src("dnt_r4.json", f"{b}.DIV12.ours.tp")
    nchem = src("dnt_r4.json", f"{b}.DIV12.n_chemicals")
    epa_tp = [src("dnt_r4.json", f"{b}.{rule}.epa_rule_same_chemicals.tp") for rule in ("DIV12", "AUC.3hit", "Top2")]
    ba = src("dnt_r4.json", f"{b}.DIV12.ours.balanced_accuracy")
    epa_ba = [src("dnt_r4.json", f"{b}.{rule}.epa_rule_same_chemicals.balanced_accuracy")
              for rule in ("DIV12", "AUC.3hit", "Top2")]
    auc = src("dnt_r4.json", "variants.k3.auroc")
    cards.append({"id": "hazard", "value": f"{ours} vs {min(epa_tp)}–{max(epa_tp)}",
                  "title": "known neurotoxicants flagged from 3 doses, vs EPA rules on all doses",
                  "detail": f"Same {nchem} reference chemicals. Balanced accuracy {100 * ba:.1f} % vs "
                            f"{100 * min(epa_ba):.1f}–{100 * max(epa_ba):.1f} % for the exactly reproduced EPA "
                            f"hit-count rules; AUROC {auc:.3f}.",
                  "sources": list(S)})
    # 3. coverage
    S.clear()
    cov = src("conformal_r7.json", "results.alpha_0.1.k3.conformal.coverage")
    clo, chi = src("conformal_r7.json", "results.alpha_0.1.k3.conformal.ci95")
    par = src("conformal_r7.json", "results.alpha_0.1.k3.parametric.coverage")
    cards.append({"id": "coverage", "value": f"{100 * cov:.1f}%", "title": "observed coverage of the nominal 90 % band",
                  "detail": f"Cross-conformal, Mondrian by cytotoxicity × k; 95 % CI {100 * clo:.1f}–"
                            f"{100 * chi:.1f} % (parametric band before calibration: {100 * par:.1f} %).",
                  "sources": list(S)})
    # 4. abstention
    S.clear()
    rate = src("conformal_r7.json", "abstention.abstention_rate")
    mr = src("conformal_r7.json", "abstention.curve_mae_retained")
    ma = src("conformal_r7.json", "abstention.curve_mae_abstained")
    cards.append({"id": "abstain", "value": f"{100 * rate:.1f}%", "title": "forecasts flagged “I don’t know”",
                  "detail": f"Abstained forecasts are the least reliable: curve error {ma:.2f} vs {mr:.2f} for the "
                            f"retained ones.", "sources": list(S)})
    # 5. wells
    S.clear()
    ws = src("potency_r3.json", "by_k.3.wells_saved_median_pct")
    cards.append({"id": "wells", "value": f"{ws:.0f}%", "title": "fewer treated wells per chemical (median, 3 doses)",
                  "detail": "Measuring 3 of the tested concentrations instead of the full series; potency is still "
                            "taken by interpolation of the measured doses (R3, a negative result for model-derived "
                            "potency).", "sources": list(S)})
    # 6. DIV 7 gate
    S.clear()
    pers = src("r11_early_exit.json", "methods.persistence.overall_exit_rate")
    safe = src("r11_early_exit.json", "methods.twin_upper90.overall_exit_rate")
    a = src("report_secondary.json", "r11_epa_labels.positive.persistence_vs_twin_upper90.events_a")
    bb = src("report_secondary.json", "r11_epa_labels.positive.persistence_vs_twin_upper90.events_b")
    npos = src("report_secondary.json", "r11_epa_labels.positive.n")
    cards.append({"id": "gate", "value": f"{100 * pers:.0f}%", "title": "of chemicals can stop at DIV 7 (throughput gate)",
                  "detail": f"Persistence gate, 2 of 4 recordings. The twin’s safety gate stops "
                            f"{100 * safe:.0f} % and lets {bb} of {npos} known neurotoxicants exit instead of {a} "
                            f"(secondary, post-hoc test).", "sources": list(S)})
    # scale strip
    S.clear()
    scale = {"chemicals": src("trajectory_cv.json", "n_chemicals"),
             "axons": src("data_audit_brewer.json", "n_axons"),
             "recordings": src("data_audit_brewer.json", "n_recordings")}
    return cards, {**scale, "sources": list(S)}


def write_summary(out: Path, index):
    cv = load_json("trajectory_cv.json")
    cf = load_json("conformal_r7.json")
    r8 = load_json("r8_chiplayer.json")
    epa = load_json("epa_baseline_repro.json")
    methods = ["neurotrajectory", "loglinear_interp", "analog_knn", "hill_per_endpoint", "context_mean", "zero"]
    by_k = {}
    for k, v in cv["by_k"].items():
        c = v["curve_mae"]
        by_k[k] = {"curve_mae": {m: c["mean"][m] for m in methods if m in c["mean"]},
                   "best_baseline": c["best_baseline"], "paired_vs_best": c["paired_vs_best"]}
    conformal = {}
    for a, ent in cf["results"].items():
        conformal[a] = {"nominal": ent["nominal"],
                        **{kk: {"parametric": ent[kk]["parametric"], "conformal": ent[kk]["conformal"]}
                           for kk in ent if kk.startswith("k")}}
    cards, scale = key_numbers()
    manifest = json.loads((REPO / "data_bundle" / "manifest.json").read_text(encoding="utf-8"))
    src_names = ("trajectory_cv.json", "conformal_r7.json", "r8_chiplayer.json", "epa_baseline_repro.json",
                 "dnt_r4.json", "dnt_r4_calls.csv", "r11_early_exit.json", "r11_early_exit_rows.csv",
                 "report_secondary.json", "potency_r3.json", "data_audit_brewer.json", "figures_derived.json")
    summary = {
        "generated_by": "scripts/precompute_site.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {n: sha256(RESULTS / n) for n in src_names}
                   | {"data_bundle/nfa_tasks.npz": sha256(REPO / "data_bundle" / "nfa_tasks.npz")},
        "key_numbers": cards,
        "scale": {**scale, "wells": manifest.get("n_wells"), "features": 17, "divs": [5, 7, 9, 12]},
        "method_labels": METHOD_LABELS,
        "trajectory": {"description": cv["description"], "n_chemicals": cv["n_chemicals"],
                       "outer_folds": cv["outer_folds"], "seeds": cv["seeds"], "designs_per_k": cv["designs_per_k"],
                       "by_k": by_k},
        "conformal": {"n_chemicals": cf["n_chemicals"], "results": conformal, "abstention": cf["abstention"]},
        "chip": {"n_recordings": r8.get("n_recordings"), "n_axons": r8.get("n_axons"),
                 "license": r8.get("license"), "source": r8.get("source")},
        "epa_baseline": {"n_reference": epa["n_reference"], "n_positive": epa["n_positive"],
                         "n_negative": epa["n_negative"]},
        "site_design_check": {"note": "The explorer uses one evenly spread design per k; the report averages seeded "
                                      "random designs per k, so these means differ slightly from trajectory_cv.json.",
                              "by_k": site_design_means(index) if index else None},
        "manifest": {kk: vv for kk, vv in manifest.items() if kk in ("n_chemicals", "n_wells")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def copy_figures(site: Path):
    dst = site / "assets" / "img"
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for src_rel, name in FIGURES:
        src = REPO / src_rel
        shutil.copyfile(src, dst / name)
        hashes[name] = {"source": src_rel, "sha256": sha256(src)}
    (site / "data" / "figures.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SITE / "data"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N chemicals (smoke test)")
    ap.add_argument("--assets-only", action="store_true", help="copy figures and rebuild summary.json only")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    copy_figures(out.parent)
    if a.assets_only:
        idx_p = out / "index.json"
        index = json.loads(idx_p.read_text(encoding="utf-8"))["chemicals"] if idx_p.exists() else []
        write_summary(out, index)
        print(f"assets and summary rebuilt in {time.time() - t0:.1f}s")
        return
    warnings.filterwarnings("ignore", category=UserWarning)
    from neurotwin.bundle import load_tasks
    tasks = load_tasks()
    records, fold_info = chemical_records(tasks, a.limit)
    qhat, conf_by_k = conformal(records, tasks)
    thresholds, abst_by_k = abstention(records, tasks)
    hz_probs, hz_summary, hz_check = hazard(records, tasks)
    gate_rows, gate_mismatch = gates(tasks)
    if a.limit is None:
        stale = {p.name for p in (out / "chem").glob("c*.json")} - {f"c{i:03d}.json" for i in records}
        for n in stale:
            (out / "chem" / n).unlink()
    index = write_chemicals(records, fold_info, thresholds, abst_by_k, qhat, conf_by_k, hz_probs, hz_summary,
                            hz_check, gate_rows, gate_mismatch, out)
    write_summary(out, index)
    size = sum(p.stat().st_size for p in out.rglob("*.json"))
    print(json.dumps({"chemicals": len(index), "site_conformal_by_k": conf_by_k, "site_abstention_by_k": abst_by_k,
                      "hazard_site_rates": hz_summary, "hazard_check_vs_dnt_r4": hz_check,
                      "gate_rows": len(gate_rows), "gate_recheck_mismatches": gate_mismatch,
                      "total_json_MB": round(size / 1e6, 2), "runtime_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
