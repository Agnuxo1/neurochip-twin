"""Precompute the data of the static explorer site (site/data/*.json) from the shipped bundle.

    python scripts/precompute_site.py                 # all 243 chemicals, k = 1, 2, 3 (CPU, a few minutes)
    python scripts/precompute_site.py --limit 5       # quick smoke run on the first 5 chemicals

Runs offline on CPU from data_bundle/ only. For every chemical it uses the NeuroTrajectory ensemble
of that chemical's cross-validation fold (trained WITHOUT it), gives the model the wells at k
evenly spread concentrations (the same 'spread' design as demo.py), and stores:

  * the forecast mean and 90 % half-width on a 40-point log10-dose grid (17 features x 4 DIV),
  * the forecast at the held-out concentrations,
  * the measured / held-out level means and the individual wells,
  * the curve MAE of NeuroTrajectory, log-linear interpolation and analog kNN at the held-out levels,
    the share of held-out well entries inside the 90 % band, and the epistemic score with the
    abstention flag (rule of results/conformal_r7.json: score = mean ensemble std of mu / mean
    predictive scale; abstain above the 90th percentile of the chemicals of the OTHER folds).

Arrays are stored as integers x100 (two decimals) in [feature][DIV][point] order, null = unobserved.
It also writes site/data/summary.json with the headline numbers copied from repo/results/*.json.
No number shown by the site is typed by hand: everything is read from these files at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from demo import load_fold_models, spread_design  # noqa: E402  (same engine as the CPU demo)
from neurotwin.bundle import BUNDLE, load_tasks  # noqa: E402
from neurotwin.models.cnp import predict  # noqa: E402
from neurotwin.models.trajectory import (  # noqa: E402
    CLIP, DIVS, FEATURES, AnalogKNN, level_means, predict_interp, split_context)

RESULTS = REPO / "results"
KS = [1, 2, 3]
N_GRID = 40
Q = 100  # stored value = round(x * Q)

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


def ensemble(models, task, is_ctx, q):
    """Same combination as demo.forecast, plus the member spread used for the epistemic score."""
    preds = [predict(m, task, is_ctx, q, device="cpu") for m in models]
    mus = np.stack([p[0] for p in preds])
    mu = mus.mean(0)
    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + mus.var(0))
    return mu, sc, mus.std(0)


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ per-chemical forecasts
def chemical_records(tasks, limit=None):
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
            rec = {"t": t, "grid": grid, "k": {}}
            for k in KS:
                kk = min(k, len(t.levels) - 1)
                ctx_lv = spread_design(t.levels, kk)
                is_ctx, is_tgt = split_context(t, ctx_lv)
                held = np.unique(t.logc[is_tgt])
                mu_h, sc_h, sd_h = ensemble(models, t, is_ctx, held)
                mu_g, sc_g, _ = ensemble(models, t, is_ctx, grid)
                lv, obs, ok = level_means(t, is_tgt)
                assert np.allclose(lv, held)
                pi = predict_interp(t, is_ctx, held)
                pk = knn.predict(t, is_ctx, held)
                err = {"model": np.abs(mu_h - obs), "interp": np.abs(pi - obs), "knn": np.abs(pk - obs)}
                curve = {m: float(e[ok].mean()) for m, e in err.items()}
                per_level = {m: [r(float(e[j][ok[j]].mean())) if ok[j].any() else None for j in range(len(held))]
                             for m, e in err.items()}
                wi = np.searchsorted(held, t.logc[is_tgt])
                inside = (np.abs(t.y[is_tgt] - mu_h[wi]) <= z90 * sc_h[wi])[t.m[is_tgt]]
                epi = float(sd_h.mean() / sc_h.mean())
                rec["k"][k] = {
                    "k_used": int(len(ctx_lv)),
                    "ctx": np.searchsorted(t.levels, ctx_lv).tolist(),
                    "held": np.searchsorted(t.levels, held).tolist(),
                    "mu": mu_g, "hw": z90 * sc_g, "mu_held": mu_h, "hw_held": z90 * sc_h,
                    "curve_mae": curve, "level_mae": per_level,
                    "coverage90": float(inside.mean()) if inside.size else float("nan"),
                    "epi": epi,
                }
            out[i] = rec
        print(f"fold {f}: {len(idx)} chemicals, {len(models)} models, z90={z90:.3f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    return out, fold_info


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


# ------------------------------------------------------------------ writers
def write_chemicals(records, fold_info, thresholds, abst_by_k, out: Path):
    (out / "chem").mkdir(parents=True, exist_ok=True)
    index = []
    for i in sorted(records):
        rec = records[i]
        t = rec["t"]
        cid = f"c{i:03d}"
        L = len(t.levels)
        counts = np.bincount(np.searchsorted(t.levels, t.logc), minlength=L)
        doc = {
            "id": cid, "chem": t.chem, "fold": t.fold, "label": t.label, "n_wells": int(len(t.logc)),
            "levels": [r(x, 4) for x in t.levels], "level_n": counts.tolist(),
            "level_mean": q_arr(t._mu, t._ok),
            "wells": {"lv": np.searchsorted(t.levels, t.logc).tolist(), "y": q_arr(t.y, t.m)},
            "grid": [r(x, 4) for x in rec["grid"]], "k": {},
        }
        summ = {}
        for k, e in rec["k"].items():
            doc["k"][str(k)] = {
                "k_used": e["k_used"], "ctx": e["ctx"], "held": e["held"],
                "mu": q_arr(e["mu"]), "hw": q_arr(e["hw"]),
                "mu_held": q_arr(e["mu_held"]), "hw_held": q_arr(e["hw_held"]),
                "level_mae": e["level_mae"],
                "metrics": {"curve_mae": {m: r(v) for m, v in e["curve_mae"].items()},
                            "coverage90": r(e["coverage90"]), "epi": r(e["epi"], 4),
                            "threshold": r(e["threshold"], 4), "abstain": e["abstain"]},
            }
            summ[str(k)] = {"m": r(e["curve_mae"]["model"]), "i": r(e["curve_mae"]["interp"]),
                            "n": r(e["curve_mae"]["knn"]), "c": r(e["coverage90"]), "a": int(e["abstain"])}
        (out / "chem" / f"{cid}.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        index.append({"id": cid, "chem": t.chem, "fold": t.fold, "label": t.label, "n_levels": L,
                      "n_wells": int(len(t.logc)), "k": summ})
    meta = {
        "features": FEATURES, "feature_labels": FEATURE_LABELS, "feature_groups": FEATURE_GROUPS,
        "default_features": DEFAULT_FEATURES, "divs": DIVS, "ks": KS, "n_grid": N_GRID, "scale": Q, "clip": CLIP,
        "design": "spread: k concentration levels evenly spaced over the tested range (demo.py spread_design)",
        "units": "effect vs vehicle in robust-SD units of asinh(x/c_f) (0 = vehicle; see transform.json)",
        "band": "90 % predictive band of the 3-seed ensemble of the chemical's CV fold (parametric, as demo.py)",
        "folds": {str(f): v for f, v in fold_info.items()},
        "abstention": {"rule": "abstain if epistemic score > 90th percentile of the scores of the chemicals "
                               "in the other folds (k = 1..3 pooled), as in results/conformal_r7.json",
                       "thresholds_by_fold": thresholds, "site_design_by_k": abst_by_k},
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


def write_summary(out: Path, index):
    cv = load_json("trajectory_cv.json")
    cf = load_json("conformal_r7.json")
    r8 = load_json("r8_chiplayer.json")
    cp = load_json("chiplayer_params.json")
    epa = load_json("epa_baseline_repro.json")
    methods = ["neurotrajectory", "loglinear_interp", "analog_knn", "hill_per_endpoint", "context_mean", "zero"]
    by_k = {}
    for k, v in cv["by_k"].items():
        c = v["curve_mae"]
        by_k[k] = {"curve_mae": {m: c["mean"][m] for m in methods if m in c["mean"]},
                   "best_baseline": c["best_baseline"], "paired_vs_best": c["paired_vs_best"],
                   "paired_vs": {m: c["paired_vs_all"][m] for m in ("loglinear_interp", "analog_knn", "hill_per_endpoint")
                                 if m in c["paired_vs_all"]},
                   "well_interval90": v.get("well_interval90")}
    conformal = {}
    for a, ent in cf["results"].items():
        conformal[a] = {"nominal": ent["nominal"],
                        **{kk: {"parametric": ent[kk]["parametric"], "conformal": ent[kk]["conformal"]}
                           for kk in ent if kk.startswith("k")}}
    rcv = r8["cv"]
    chip = {
        "title": r8.get("title"), "source": r8.get("source"), "license": r8.get("license"),
        "n_recordings": r8.get("n_recordings"), "n_axons": r8.get("n_axons"), "conditions": r8.get("conditions"),
        "graph_description": r8.get("graph_description"), "burst_interpretation": r8.get("burst_interpretation"),
        "cv": {"protocol": rcv.get("protocol"), "model": rcv.get("model"), "baseline_tunnel": rcv.get("baseline_tunnel"),
               "baseline_direction_null": rcv.get("baseline_direction_null"), "summary": rcv.get("summary"),
               "comparisons": rcv.get("comparisons"),
               "folds": [{"condition": f.get("condition"), "fid": f.get("fid"), "n_axons": f.get("n_axons"),
                          "n_observed_tunnels": f.get("n_observed_tunnels"), "scores": f.get("scores")}
                         for f in rcv.get("folds", [])]},
        "params": {"source_cohort": cp.get("source_cohort"), "scope": cp.get("scope"), "units": cp.get("units"),
                   "pair_edges": [{kk: vv for kk, vv in e.items() if kk != "conduction"} |
                                  {"conduction": {kk: vv for kk, vv in e["conduction"].items() if kk != "values_ms"}}
                                  for e in cp["fit_pair_edges"]],
                   "n_tunnels": len(cp.get("fit_tunnels", []))},
        "observed": [{"condition": o["condition"], "fid": o["fid"],
                      "edges": [{kk: e.get(kk) for kk in ("pair", "n_axons_ff", "n_axons_fb", "ff_fraction",
                                                          "directionality_d")} for e in o["edges"]]}
                     for o in cp.get("observed_per_recording", [])],
    }
    summary = {
        "generated_by": "scripts/precompute_site.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {n: sha256(RESULTS / n) for n in ("trajectory_cv.json", "conformal_r7.json", "r8_chiplayer.json",
                                                    "chiplayer_params.json", "epa_baseline_repro.json")}
                   | {"data_bundle/nfa_tasks.npz": sha256(BUNDLE / "nfa_tasks.npz")},
        "trajectory": {"description": cv["description"], "n_chemicals": cv["n_chemicals"],
                       "outer_folds": cv["outer_folds"], "seeds": cv["seeds"], "designs_per_k": cv["designs_per_k"],
                       "target_units": cv.get("target_units"), "development_note": cv.get("development_note"),
                       "by_k": by_k, "by_k_folds_1to4": cv.get("by_k_folds_1to4")},
        "conformal": {"description": cf["description"], "n_chemicals": cf["n_chemicals"],
                      "chemicals_per_regime": cf.get("chemicals_per_regime"), "results": conformal,
                      "abstention": cf["abstention"]},
        "chip": chip,
        "epa_baseline": {"method": epa["method"], "n_reference": epa["n_reference"], "n_positive": epa["n_positive"],
                         "n_negative": epa["n_negative"], "models": epa["models"],
                         "comparison_explanation": epa.get("comparison_explanation"),
                         "interpretation": epa.get("interpretation")},
        "site_design_check": {"note": "Explorer uses one evenly spread design per k; the paper averages 5 seeded "
                                      "random designs per k, so these means differ slightly from trajectory_cv.json.",
                              "by_k": site_design_means(index)},
        "manifest": {kk: vv for kk, vv in json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8")).items()
                     if kk in ("n_chemicals", "n_wells")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "site/data"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N chemicals (smoke test)")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    records, fold_info = chemical_records(tasks, a.limit)
    thresholds, abst_by_k = abstention(records, tasks)
    if a.limit is None:
        stale = {p.name for p in (out / "chem").glob("c*.json")} - {f"c{i:03d}.json" for i in records}
        for n in stale:
            (out / "chem" / n).unlink()
    index = write_chemicals(records, fold_info, thresholds, abst_by_k, out)
    write_summary(out, index)
    size = sum(p.stat().st_size for p in out.rglob("*.json"))
    print(json.dumps({"chemicals": len(index), "thresholds_by_fold": thresholds, "site_abstention_by_k": abst_by_k,
                      "total_json_MB": round(size / 1e6, 2), "runtime_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
