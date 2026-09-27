"""R10 option D: honest-ceiling audit of NeuroTrajectory's out-of-fold residuals and value of information of viability.

Models: the PRIMARY v1 cross-validated models (data/processed/models, the nested selection recorded in
results/trajectory_cv.json), read-only. The exploratory pilots in colab/research/fran/_pilot used models_v2; the
delivered numbers below use v1.

1. Residuals. For every chemical of outer folds 1-4 (never seen by its fold's models), k in {1,2,3,4} measured
   concentration levels and the same 5 seeded designs as trajectory_cv (`designs`), residual
   r = ensemble prediction - observed level mean at every held-out (level, DIV, feature) entry. The script checks
   that the mean curve MAE per k reproduces trajectory_cv.json (by_k_folds_1to4).
2. Darwin-Cage search (port of the team lead's search, repo/NOTICE) over programs built only from information
   available at prediction time: feature, DIV, k, extrapolation side, plate|date, annotated chemical class (R10),
   and binned log c, distance to the nearest context level, predicted mean, predicted scale, plate vehicle level,
   number of replicate wells. Search on chemical folds 1-3, confirmation on fold-4 chemicals that the search never
   sees (then rotated as a sensitivity analysis). A program is confirmed only if its confirmation correlation beats
   the original z/sqrt(n) rule AND a chemical sign-flip randomisation test at the same one-sided level (z = 3).
3. Power analysis with plant: plate|date shifts and feature x DIV x class interactions of 0.05/0.1/0.2/0.4/0.8
   vehicle robust SD (the target unit) added to the real residual, 5 random sign draws each. A plant is recovered
   when the search confirms a program containing all planted atoms (the real residual may carry confirmed
   structure of its own, so 'any confirmation' is not a valid recovery criterion). Minimum detectable effect (MDE)
   at power 0.8; the 'oracle' power (planted program handed to the confirmation test) bounds it from below.
4. Secondary: curve-MAE change when the confirmed correction is subtracted, strictly nested (for each test fold the
   search and its confirmation use only the other three folds), paired bootstrap by chemical; baseline = a generic
   residual learner (HistGradientBoosting) with nested selection.
5. Viability value of information (results/r10_viability_voi.json): nested chemical-grouped corrector with and
   without viability information. Arms: PubChem AB/LDH chemical-level calls ('prior viability screening scenario',
   with leakage caveat); NTP DNT-DIVER per-well viability as chemical-level summaries (same scenario) and restricted
   to the wells at the measured context concentrations only (no hidden-concentration information).

Usage: python repo/scripts/run_residual_audit.py [--regen] [--fresh] [--evals N] [--plant-evals N] [--reps R]
       [--stages residuals search power nested voi figure]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("NT_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))

from neurotwin.audit.darwin import (Alphabet, AtomCache, DarwinSearch, integer_view,  # noqa: E402
                                    plant_residual, program_key, uses_atoms)
from neurotwin.models.trajectory import DIVS, FEATURES, ND, NF, level_means, split_context  # noqa: E402

AUDIT = ROOT / "data/processed/audit"
RESID = AUDIT / "residuals_v1.parquet"
DESIGN_TAB = AUDIT / "designs_v1.parquet"
PARTS = AUDIT / "parts"
MODELS = ROOT / "data/processed/models"
OUT = ROOT / "repo/results"
FIG = OUT / "figures"
KS = (1, 2, 3, 4)
FOLDS = (1, 2, 3, 4)
PRIMARY_CONFIRM = 4
EFFECTS = (0.05, 0.1, 0.2, 0.4, 0.8)
CATS = ["feat", "div", "k", "extrap", "plate", "cls"]
NUM_SCALE = {"logc": 10, "dist": 10, "pred": 10, "pred_scale": 100, "veh_z": 10, "nrep": 1}
ALPHABET = Alphabet(cats=tuple(CATS), nums=tuple(NUM_SCALE), ops=("raw", "div", "qbin"),
                    divs=(2, 3, 5, 10, 20, 50), qbins=(2, 3, 4, 5, 8, 10, 16, 20))
PLANTS = {"plate_shift": [("cat", "plate")],
          "feature_x_div_x_class": [("cat", "feat"), ("cat", "div"), ("cat", "cls")]}
NTP_EP = {"AB": "viability AB", "cell": "cell viability"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rnd(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ------------------------------------------------------------------------------------------------ residuals
def vehicle_z(tr):
    """Plate vehicle level per (plate|date, DIV, feature): asinh(median / c_f), z-scored across plates per DIV."""
    veh = pd.read_parquet(ROOT / "data/processed/epa_nfa/vehicle_stats.parquet")
    veh["pd"] = veh.plate.astype(str) + "|" + veh.date.astype(str)
    out = {}
    for fi, f in enumerate(FEATURES):
        col = f"vehicle_median_{f}"
        g = np.arcsinh(veh[col].to_numpy(float) / tr.c[f])
        for d, dv in enumerate(DIVS):
            m = veh["div"].to_numpy() == dv
            mu, sd = np.nanmean(g[m]), np.nanstd(g[m])
            for p, v in zip(veh.pd[m], g[m]):
                out[(p, d, fi)] = (v - mu) / sd if sd > 0 and np.isfinite(v) else np.nan
    return out


def regenerate(tasks, tr):
    import torch
    from neurotwin.models.cnp import load_fold_models, predict
    from run_trajectory_cv import designs
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "4")))
    vz = vehicle_z(tr)
    cols = {c: [] for c in ("chem", "fold", "k", "design", "logc", "li", "div", "feat", "dist", "extrap", "above_min",
                            "above_max", "pred", "pred_scale", "obs", "nrep", "plate", "veh_z")}
    drows = []
    t0 = time.time()
    for f in FOLDS:
        models, cfg = load_fold_models(f, device="cpu", directory=MODELS)
        assert len(models) == 3, f"expected 3 v1 models for fold {f}"
        for t in [t for t in tasks if t.fold == f]:
            for k in KS:
                if k >= len(t.levels):
                    continue
                for di, ctx in enumerate(designs(t, k)):
                    ic, it = split_context(t, ctx)
                    lv, mu, ok = level_means(t, it)
                    preds = [predict(m, t, ic, lv, device="cpu") for m in models]
                    pm = np.mean([p[0] for p in preds], 0)
                    ps = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + np.var([p[0] for p in preds], 0))
                    plates, nreps = [], []
                    for l in lv:
                        sel = it & (t.logc == l)
                        plates.append(pd.Series(t.plate[sel]).value_counts().index[0])
                        nreps.append(int(sel.sum()))
                    li, d, fi = np.nonzero(ok)
                    n = len(li)
                    cmin, cmax = float(ctx.min()), float(ctx.max())
                    lvl = lv[li].astype(np.float64)
                    cols["chem"].append(np.full(n, t.chem, object))
                    cols["fold"].append(np.full(n, f, np.int8))
                    cols["k"].append(np.full(n, k, np.int8))
                    cols["design"].append(np.full(n, di, np.int8))
                    cols["logc"].append(lvl.astype(np.float32))
                    cols["li"].append(li.astype(np.int8))
                    cols["div"].append(np.asarray(DIVS, np.int8)[d])
                    cols["feat"].append(fi.astype(np.int8))
                    cols["dist"].append(np.abs(lvl[:, None] - ctx[None, :]).min(1).astype(np.float32))
                    cols["extrap"].append(np.where(lvl > cmax, 1, np.where(lvl < cmin, -1, 0)).astype(np.int8))
                    cols["above_min"].append((lvl - cmin).astype(np.float32))
                    cols["above_max"].append((lvl - cmax).astype(np.float32))
                    cols["pred"].append(pm[li, d, fi].astype(np.float32))
                    cols["pred_scale"].append(ps[li, d, fi].astype(np.float32))
                    cols["obs"].append(mu[li, d, fi].astype(np.float32))
                    cols["nrep"].append(np.asarray(nreps, np.int16)[li])
                    pl = np.asarray(plates, object)[li]
                    cols["plate"].append(pl)
                    cols["veh_z"].append(np.array([vz.get((p, dd, ff), np.nan) for p, dd, ff in zip(pl, d, fi)],
                                                  np.float32))
                    drows.append({"chem": t.chem, "fold": f, "k": k, "design": di,
                                  "ctx": ";".join(f"{x:.6f}" for x in ctx)})
        log(f"residuals fold {f}: {sum(len(a) for a in cols['k'])} rows ({time.time() - t0:.0f}s)")
    df = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()})
    df["chem"] = df.chem.astype("category")
    df["plate"] = df.plate.astype("category")
    df["resid"] = (df.pred - df.obs).astype(np.float32)
    AUDIT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESID)
    pd.DataFrame(drows).to_parquet(DESIGN_TAB)
    return df


def curve_table(df, abs_col):
    """Curve MAE per (chemical, k): mean |.| over entries of a design, then mean over designs (as trajectory_cv)."""
    g = df.groupby(["chem", "k", "design"], observed=True)[abs_col].mean()
    return g.groupby(level=["chem", "k"], observed=True).mean()


def integrity(df):
    ref = json.loads((OUT / "trajectory_cv.json").read_text(encoding="utf-8"))["by_k_folds_1to4"]
    ct = curve_table(df.assign(a=np.abs(df.resid.to_numpy(np.float64))), "a")
    out = {}
    for k in KS:
        v = float(ct.xs(k, level="k").mean())
        out[str(k)] = {"curve_mae_regenerated": round(v, 4), "trajectory_cv_json": ref[str(k)]["neurotrajectory"],
                       "abs_diff": round(abs(v - ref[str(k)]["neurotrajectory"]), 6),
                       "n_chemicals": int(ct.xs(k, level="k").shape[0])}
    out["all_match_1e-3"] = bool(all(out[str(k)]["abs_diff"] < 1e-3 for k in KS))
    return out


# ------------------------------------------------------------------------------------------------ covariates
def chemical_covariates(tasks):
    w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna()
    dtx = w.drop_duplicates().groupby("chemical").dtxsid.apply(set).to_dict()
    ann = pd.read_csv(ROOT / "data/neurotox_probe/nfa_refine/annotate_dnt_ref_chems.csv")
    cls = {}
    for _, r in ann.iterrows():        # same annotation rule as run_selectivity_failures.py (R10), case-folded
        c = r.get("neuro.class") if isinstance(r.get("neuro.class"), str) else r.get("Class")
        if isinstance(c, str) and isinstance(r.get("dsstox_substance_id"), str):
            c = {"pyrethroid2": "pyrethroid", "pyrethoid": "pyrethroid"}.get(c, c)
            cls[r["dsstox_substance_id"]] = c.strip().lower()
    via = pd.read_parquet(ROOT / "data/processed/epa_nfa/viability.parquet")
    pub = via[via.source.str.startswith("PubChem")]
    calls = {}
    for ep in ("viability_AB", "viability_LDH"):
        s = pub[pub.endpoint == ep].set_index("dtxsid").hit_call
        calls[ep] = {d: (np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
                     for d, v in s.items()}
    ntp = via[via.source.str.startswith("NTP")].copy()
    ntp["lc"] = np.log10(ntp.concentration_uM.astype(float))
    med = ntp.groupby(["dtxsid", "ntp_endpoint", "lc"]).response_normalized.median()
    rows = []
    for t in tasks:
        ds = dtx.get(t.chem, set())
        c = next((cls[d] for d in sorted(ds) if d in cls), "unannotated")
        ab = [calls["viability_AB"].get(d, np.nan) for d in ds]
        ldh = [calls["viability_LDH"].get(d, np.nan) for d in ds]
        nd = sorted(d for d in ds if d in set(ntp.dtxsid))
        row = {"chem": t.chem, "fold": t.fold, "cls": c,
               "cyto_AB": np.nanmax(ab) if np.isfinite(ab).any() else np.nan,
               "cyto_LDH": np.nanmax(ldh) if np.isfinite(ldh).any() else np.nan,
               "ntp_dtxsid": nd[0] if nd else None}
        for tag, ep in NTP_EP.items():
            row[f"ntp_min_{tag}"] = float(med.loc[(nd[0], ep)].min()) if nd and (nd[0], ep) in med.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows), med


def ntp_match_audit(tasks, chem_tab):
    """Do the NTP 'USEPA 91 neuron firing' viability wells look like the NFA wells of the same chemical?"""
    via = pd.read_parquet(ROOT / "data/processed/epa_nfa/viability.parquet")
    ntp = via[via.source.str.startswith("NTP")]
    byc = {t.chem: t for t in tasks}
    same_conc, same_reps, n = 0, 0, 0
    per = []
    for _, r in chem_tab[chem_tab.ntp_dtxsid.notna()].iterrows():
        t = byc[r.chem]
        nfa = pd.Series(np.round(10 ** t.logc.astype(np.float64), 4)).value_counts().sort_index()
        eq_c, eq_r = True, True
        for ep in NTP_EP.values():
            s = ntp[(ntp.dtxsid == r.ntp_dtxsid) & (ntp.ntp_endpoint == ep)].concentration_uM.round(4).value_counts()
            s = s.sort_index()
            eq_c &= set(s.index) == set(nfa.index)
            eq_r &= eq_c and all(int(s.get(c, -1)) == int(nfa[c]) for c in nfa.index)
        n += 1
        same_conc += int(eq_c)
        same_reps += int(eq_r)
        per.append({"chem": r.chem, "same_concentrations": bool(eq_c), "same_replicate_counts": bool(eq_r)})
    return {"n_ntp_chemicals_in_cohort": n, "n_identical_concentration_sets": same_conc,
            "n_identical_replicates_per_concentration": same_reps,
            "ntp_plate_ids": sorted(ntp.ntp_plate.astype(str).unique().tolist()),
            "nfa_source_files_note": "NFA wells of these chemicals come from the NTP91 / Frank2017 / ToxCast2016 NFA "
                                     "source files (wells.parquet: source_file)",
            "per_chemical": per}


def ntp_context_features(df, dtab, chem_tab, med):
    """Viability of the wells at the measured CONTEXT concentrations only (never at hidden levels)."""
    ntpd = chem_tab.set_index("chem").ntp_dtxsid.to_dict()
    out = {f"ntp_ctx_{s}_{tag}": np.full(len(df), np.nan, np.float32) for s in ("min", "near") for tag in NTP_EP}
    key = df.chem.astype(str) + "|" + df.k.astype(str) + "|" + df.design.astype(str)
    dtab = dtab.assign(key=dtab.chem.astype(str) + "|" + dtab.k.astype(str) + "|" + dtab.design.astype(str))
    ctx_of = dict(zip(dtab.key, dtab.ctx))
    idx_by_key = pd.Series(np.arange(len(df))).groupby(key.to_numpy()).apply(np.asarray)
    for kk, rows in idx_by_key.items():
        chem = kk.split("|")[0]
        d = ntpd.get(chem)
        if d is None:
            continue
        ctx = np.array([float(x) for x in ctx_of[kk].split(";")])
        q = df.logc.to_numpy()[rows].astype(np.float64)
        near = np.abs(q[:, None] - ctx[None, :]).argmin(1)
        for tag, ep in NTP_EP.items():
            if (d, ep) not in med.index:
                continue
            s = med.loc[(d, ep)]
            lc = s.index.to_numpy(float)
            v = np.array([s.iloc[int(np.abs(lc - c).argmin())] if np.abs(lc - c).min() < 0.02 else np.nan for c in ctx])
            if np.isfinite(v).any():
                out[f"ntp_ctx_min_{tag}"][rows] = np.nanmin(v)
            out[f"ntp_ctx_near_{tag}"][rows] = v[near]
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------------------------------------------ search
def search_view(df):
    v = df[CATS + list(NUM_SCALE)].copy()
    v["veh_z"] = v.veh_z.fillna(0.0)
    return integer_view(v, CATS, NUM_SCALE)


def part(name, fresh, fn):
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"{name}.json"
    if p.exists() and not fresh:
        return json.loads(p.read_text(encoding="utf-8"))
    res = fn()
    p.write_text(json.dumps(res, default=float), encoding="utf-8")
    return res


def slim(res, keep_hall=10):
    keys = ("evaluated", "generations", "seconds", "threshold", "z", "p_one_sided", "n_confirm_rows",
            "n_confirm_groups", "n_hall", "familywise_fp_expected", "n_pass_original_rule")
    out = {k: res[k] for k in keys}
    out["n_confirmed"] = len(res["confirmed"])
    out["confirmed"] = res["confirmed"]
    out["top_hall"] = res["hall"][:keep_hall]
    out["best_fit_trace"] = [(h["evaluated"], round(h["best_fit"], 5)) for h in res["history"]]
    return out


def run_search(D, r, fold, groups, confirm, evals, seed, cache=None, pop=80, stop_when=None, null=True):
    """One Darwin-Cage search (confirmation on `confirm`), plus the fitness of random keys as a reference."""
    s = DarwinSearch(D=D, r=r, fold=fold, alphabet=ALPHABET, confirm_fold=confirm, groups=groups, cache=cache)
    res = s.run(max_evals=evals, pop=pop, seed=seed, log_every=5, stop_when=stop_when)
    res["null_fitness_random_keys"] = list(s.null(reps=10, n_groups=500, seed=seed)) if null else None
    return s, res


def primary_search(df, D, groups, a):
    r = df.resid.to_numpy(np.float64)
    fold = df.fold.to_numpy()
    cache = AtomCache(D, max_items=48)
    out = {}
    for c in (PRIMARY_CONFIRM,) + tuple(x for x in FOLDS if x != PRIMARY_CONFIRM):
        def go(c=c):
            log(f"search confirm fold {c}")
            _, res = run_search(D, r, fold, groups, c, a.evals, seed=2026 + c, cache=cache)
            sl = slim(res)
            sl["null_fitness_random_keys"] = res["null_fitness_random_keys"]
            return sl
        out[str(c)] = part(f"search_confirm{c}", a.fresh, go)
        log(f"  confirm fold {c}: evaluated {out[str(c)]['evaluated']}, hall {out[str(c)]['n_hall']}, "
            f"pass original rule {out[str(c)]['n_pass_original_rule']}, confirmed {out[str(c)]['n_confirmed']}")
    return out


def power_analysis(df, D, groups, a):
    r = df.resid.to_numpy(np.float64)
    fold = df.fold.to_numpy()
    cache = AtomCache(D, max_items=48)
    res = {}
    s0 = DarwinSearch(D=D, r=r, fold=fold, alphabet=ALPHABET, confirm_fold=PRIMARY_CONFIRM, groups=groups, cache=cache)
    for name, prog in PLANTS.items():
        det0 = s0.confirm_detail(program_key(prog, D, cache))
        res[name] = {"unplanted_program_on_real_residual": {
            "confirm_corr": det0["confirm"], "p_signflip": det0["p_signflip"], "null_sd": det0.get("null_sd"),
            "confirmed": bool(det0["confirm"] > s0.threshold and det0["p_signflip"] < s0.p_one)}}
        for e in EFFECTS:
            reps = []
            for rep in range(a.reps):
                def go(e=e, rep=rep, prog=prog, name=name):
                    rp, _ = plant_residual(r, D, prog, e, seed=1000 * rep + int(1000 * e), cache=cache)
                    unplanted = {}

                    def recovered(h):
                        """Confirmed, contains every planted atom, and NOT confirmed on the unplanted residual
                        (so a program riding on the real structure does not count as recovering the plant)."""
                        if not uses_atoms(h["program"], prog):
                            return False
                        pid = json.dumps(h["program"])
                        if pid not in unplanted:
                            d0 = s0.confirm_detail(program_key([tuple(x) for x in h["program"]], D, cache))
                            unplanted[pid] = bool(d0["confirm"] > s0.threshold and d0["p_signflip"] < s0.p_one)
                        return not unplanted[pid]
                    s, out = run_search(D, rp, fold, groups, PRIMARY_CONFIRM, a.plant_evals, seed=7 + rep, cache=cache,
                                        pop=50, stop_when=lambda conf: any(recovered(h) for h in conf), null=False)
                    det = s.confirm_detail(program_key(prog, D, cache))
                    conf = out["confirmed"]
                    rec = [h for h in conf if recovered(h)]
                    return {"effect": e, "rep": rep, "evaluated": out["evaluated"], "n_confirmed": len(conf),
                            "recovered_any": bool(conf),
                            "recovered_exact": bool(rec),
                            "recovered_partial": bool(any(any(uses_atoms(h["program"], [x]) for x in prog) for h in conf)),
                            "n_confirmed_with_plant_atoms_but_confirmed_unplanted":
                                int(sum(uses_atoms(h["program"], prog) and not recovered(h) for h in conf)),
                            "oracle_confirm_corr": det["confirm"], "oracle_p_signflip": det["p_signflip"],
                            "oracle_confirmed": bool(det["confirm"] > s.threshold and det["p_signflip"] < s.p_one),
                            "recovering_program": rec[0]["program"] if rec else None}
                reps.append(part(f"plant_{name}_{e}_{rep}", a.fresh, go))
            res[name][str(e)] = {"reps": reps,
                                 "power_any": float(np.mean([x["recovered_any"] for x in reps])),
                                 "power_exact": float(np.mean([x["recovered_exact"] for x in reps])),
                                 "power_oracle": float(np.mean([x["oracle_confirmed"] for x in reps]))}
            log(f"plant {name} effect {e}: power any {res[name][str(e)]['power_any']}, exact "
                f"{res[name][str(e)]['power_exact']}, oracle {res[name][str(e)]['power_oracle']}")
        for kind in ("any", "exact", "oracle"):
            pw = [res[name][str(e)][f"power_{kind}"] for e in EFFECTS]
            res[name][f"mde_power0.8_{kind}"] = mde(EFFECTS, pw)
    res["recovery_criterion"] = ("exact (primary): the search confirms a program that contains every planted atom and "
                                 "that is not confirmed on the unplanted residual; any: at least one confirmed program "
                                 "(contaminated when the real residual carries confirmed structure); oracle: the planted "
                                 "program itself passes the confirmation rule. Runs stop early once 'exact' holds.")
    return res


def column_sets(confirmed):
    """Distinct sets of columns used by confirmed programs (programs differing only in bin widths collapse)."""
    sets = {}
    for h in confirmed:
        cols = tuple(sorted({a[1] for a in h["program"]}))
        sets[" x ".join(cols)] = sets.get(" x ".join(cols), 0) + 1
    return dict(sorted(sets.items(), key=lambda kv: -kv[1]))


def describe_structure(df):
    """Descriptive (all folds, in-sample): residual mean and median by decile of the predicted value and of the
    plate vehicle level, i.e. where the conditional mean of the residual moves."""
    out = {}
    for col in ("pred", "veh_z"):
        q = pd.qcut(df[col], 10, duplicates="drop")
        g = df.groupby(q, observed=True).resid.agg(["mean", "median", "count"])
        out[f"by_{col}_decile"] = [{"range": str(i), "mean_resid": rnd(r["mean"]), "median_resid": rnd(r["median"]),
                                    "n": int(r["count"])} for i, r in g.iterrows()]
    return out


def mde(effects, power, target=0.8):
    """Smallest grid effect with power >= target, and a log-linear interpolation between grid points."""
    grid = next((e for e, p in zip(effects, power) if p >= target), None)
    interp = None
    for (e0, p0), (e1, p1) in zip(zip(effects, power), zip(effects[1:], power[1:])):
        if p0 < target <= p1:
            interp = float(10 ** (np.log10(e0) + (target - p0) / (p1 - p0) * (np.log10(e1) - np.log10(e0))))
            break
    if power and power[0] >= target:
        interp = float(effects[0])
    return {"grid": grid, "interpolated": rnd(interp, 4), "below_grid": bool(power and power[0] >= target),
            "above_grid": grid is None}


def nested_darwin_correction(df, a):
    """For each test fold c: search on two of the other folds, confirm on the third, apply to c."""
    corr = np.zeros(len(df))
    info = {}
    for i, c in enumerate(FOLDS):
        inner = FOLDS[(i + 1) % len(FOLDS)]
        m = (df.fold != c).to_numpy()
        sub = df[m]
        Ds = search_view(sub)

        def go(c=c, inner=inner, sub=sub, Ds=Ds):
            log(f"nested search: test fold {c}, inner confirm fold {inner}")
            _, res = run_search(Ds, sub.resid.to_numpy(np.float64), sub.fold.to_numpy(), sub.chem.cat.codes.to_numpy(),
                                inner, a.nested_evals, seed=3030 + c)
            return slim(res, keep_hall=3)
        res = part(f"nested_test{c}", a.fresh, go)
        info[str(c)] = {"inner_confirm_fold": inner, "evaluated": res["evaluated"], "n_confirmed": res["n_confirmed"],
                        "applied_program": res["confirmed"][0]["program"] if res["confirmed"] else None}
        if res["confirmed"]:
            D = search_view(df)
            key = program_key([tuple(x) for x in res["confirmed"][0]["program"]], D)
            nk = int(key.max()) + 1
            s = np.bincount(key[m], weights=df.resid.to_numpy(np.float64)[m], minlength=nk)
            n = np.bincount(key[m], minlength=nk)
            est = s / (n + 5.0)
            corr[~m] = est[key[~m]]
    return corr, info


# ------------------------------------------------------------------------------------------------ correctors
def hgb(kind):
    from sklearn.ensemble import HistGradientBoostingRegressor
    if kind == "hgb_l2":      # the pilot's learner
        return dict(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200)
    if kind == "hgb_l1":
        return dict(loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200)
    raise ValueError(kind)


CANDIDATES = ("identity", "hgb_l2", "hgb_l1")


def fit_predict(kind, Xtr, ytr, Xte, cat_mask, seed=0):
    if kind == "identity":
        return np.zeros(len(Xte))
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(**hgb(kind), categorical_features=cat_mask, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def curve_mae_after(df, idx, corr):
    sub = df.iloc[idx]
    a = np.abs(sub.resid.to_numpy(np.float64) - corr)
    t = curve_table(sub.assign(a=a), "a")
    return float(t.mean())


def nested_corrector(df, feats, cats, max_train, tag, fresh):
    """Out-of-fold correction: for each outer fold c, choose among CANDIDATES by leave-one-fold-out over the three
    training folds (chemical-grouped), refit the choice on all three, predict fold c."""
    def go():
        X = df[feats].to_numpy(np.float64)
        y = df.resid.to_numpy(np.float64)
        fold = df.fold.to_numpy()
        cat_mask = np.array([f in cats for f in feats])
        rng = np.random.default_rng(0)
        corr = np.zeros(len(df))
        info = {}
        for c in FOLDS:
            tr_folds = [x for x in FOLDS if x != c]
            scores = {}
            for kind in CANDIDATES:
                sc = []
                for v in tr_folds:
                    tr = np.where(np.isin(fold, [x for x in tr_folds if x != v]))[0]
                    te = np.where(fold == v)[0]
                    if len(tr) > max_train:
                        tr = np.sort(rng.choice(tr, max_train, replace=False))
                    p = fit_predict(kind, X[tr], y[tr], X[te], cat_mask)
                    sc.append(curve_mae_after(df, te, p))
                scores[kind] = float(np.mean(sc))
            best = min(scores, key=scores.get)
            tr = np.where(np.isin(fold, tr_folds))[0]
            if len(tr) > max_train:
                tr = np.sort(rng.choice(tr, max_train, replace=False))
            te = np.where(fold == c)[0]
            corr[te] = fit_predict(best, X[tr], y[tr], X[te], cat_mask)
            info[str(c)] = {"inner_curve_mae": {k: round(v, 5) for k, v in scores.items()}, "chosen": best}
            log(f"  [{tag}] fold {c}: chosen {best} {info[str(c)]['inner_curve_mae']}")
        return {"corr": corr.tolist(), "info": info}
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"corrector_{tag}.npz"
    if p.exists() and not fresh:
        z = np.load(p, allow_pickle=True)
        return z["corr"], json.loads(str(z["info"]))
    res = go()
    np.savez(p, corr=np.asarray(res["corr"]), info=json.dumps(res["info"]))
    return np.asarray(res["corr"]), res["info"]


def compare(df, corr_a, corr_b=None, chems=None):
    """Per k: paired bootstrap by chemical of curve MAE after correction a vs after b (b = none if omitted)."""
    from run_trajectory_cv import paired_boot
    sel = np.ones(len(df), bool) if chems is None else df.chem.isin(chems).to_numpy()
    sub = df[sel]
    r = sub.resid.to_numpy(np.float64)
    ta = curve_table(sub.assign(a=np.abs(r - corr_a[sel])), "a")
    tb = curve_table(sub.assign(a=np.abs(r - (0 if corr_b is None else corr_b[sel]))), "a")
    out = {}
    for k in KS:
        a_, b_ = ta.xs(k, level="k"), tb.xs(k, level="k")
        pb = paired_boot(a_.to_numpy(), b_.reindex(a_.index).to_numpy())
        out[str(k)] = {"curve_mae_after": round(float(a_.mean()), 4), "curve_mae_reference": round(float(b_.mean()), 4),
                       **pb}
    return out


# ------------------------------------------------------------------------------------------------ figures
COL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e",
       "grid": "#e4e3df", "surface": "#fcfcfb"}
MARK = ["o", "s", "^", "D"]


def _style(ax, title):
    ax.set_title(title, fontsize=10, color=COL["ink"], loc="left")
    ax.grid(True, color=COL["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(COL["ink2"])
    ax.tick_params(colors=COL["ink2"], labelsize=8)


def _rel_series(ax, comps, labels, title):
    ks = np.array([int(k) for k in KS])
    w = 0.8 / max(len(comps), 1)
    for i, (c, lab) in enumerate(zip(comps, labels)):
        rel = [c[str(k)]["rel_change_pct"] for k in KS]
        lo = [100 * c[str(k)]["ci95"][0] / c[str(k)]["curve_mae_reference"] for k in KS]
        hi = [100 * c[str(k)]["ci95"][1] / c[str(k)]["curve_mae_reference"] for k in KS]
        x = ks - 0.4 + w * (i + 0.5)
        ax.errorbar(x, rel, yerr=[np.subtract(rel, lo), np.subtract(hi, rel)], fmt=MARK[i % 4], ms=6, lw=2,
                    color=COL[f"s{i + 1}"], capsize=0, label=lab)
    ax.axhline(0, color=COL["ink2"], lw=1)
    ax.set_xticks(ks)
    ax.set_xlabel("measured context concentrations k", fontsize=9, color=COL["ink2"])
    ax.set_ylabel("curve MAE change vs reference (%)", fontsize=9, color=COL["ink2"])
    _style(ax, title)
    ax.legend(fontsize=7.5, frameon=False, loc="best")


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    au = json.loads((OUT / "r10_residual_audit.json").read_text(encoding="utf-8"))
    if "search" in au and "power" in au and "secondary_nested_correction" in au:
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor=COL["surface"])
        rot = sorted(au["search"], key=int)
        x = np.arange(len(rot))
        a1 = [au["search"][c]["n_pass_original_rule"] for c in rot]
        a2 = [au["search"][c]["n_confirmed"] for c in rot]
        axs[0].bar(x - 0.2, a1, 0.38, color=COL["s1"], label="original z/sqrt(n) rule only")
        axs[0].bar(x + 0.2, a2, 0.38, color=COL["s2"], label="+ chemical sign-flip test (used)")
        for xi, v in zip(x - 0.2, a1):
            axs[0].text(xi, v, str(v), ha="center", va="bottom", fontsize=7, color=COL["ink2"])
        for xi, v in zip(x + 0.2, a2):
            axs[0].text(xi, v, str(v), ha="center", va="bottom", fontsize=7, color=COL["ink2"])
        axs[0].set_xticks(x, [f"fold {c}" + (" (primary)" if int(c) == PRIMARY_CONFIRM else "") for c in rot])
        axs[0].set_ylim(0, 1.3 * max(a1 + a2 + [1]))
        axs[0].set_ylabel("confirmed programs", fontsize=9, color=COL["ink2"])
        _style(axs[0], "a  Programs confirmed on unseen chemicals")
        axs[0].legend(fontsize=7.5, frameon=False)
        eff = au["protocol"]["plant_effects_vehicle_sd"]
        for i, (name, lab) in enumerate((("plate_shift", "plate|date shift"),
                                         ("feature_x_div_x_class", "feature x DIV x class"))):
            pw = au["power"][name]
            xe = np.asarray(eff) * (0.96 if i == 0 else 1.04)      # small offset so coincident curves stay visible
            axs[1].plot(xe, [pw[str(e)]["power_exact"] for e in eff], marker=MARK[i], ms=7, lw=2,
                        color=COL[f"s{i + 1}"], label=f"{lab}: search")
            axs[1].plot(xe, [pw[str(e)]["power_oracle"] for e in eff], marker=MARK[i], ms=6, lw=1.5, ls="--",
                        color=COL[f"s{i + 1}"], mfc="white", label=f"{lab}: oracle program")
        axs[1].axhline(0.8, color=COL["ink2"], lw=1, ls=":")
        axs[1].set_xscale("log")
        axs[1].set_xticks(eff, [str(e) for e in eff])
        axs[1].set_ylim(-0.05, 1.05)
        axs[1].set_xlabel("planted effect (vehicle robust SD)", fontsize=9, color=COL["ink2"])
        axs[1].set_ylabel("power (fraction of 5 plants recovered)", fontsize=9, color=COL["ink2"])
        _style(axs[1], "b  Power of the audit (plant)")
        axs[1].legend(fontsize=7, frameon=False, loc="upper left")
        sec = au["secondary_nested_correction"]
        _rel_series(axs[2], [sec["darwin_confirmed_correction"]["vs_uncorrected"],
                             sec["baseline_generic_learner_prediction_time_covariates"]["vs_uncorrected"]],
                    ["confirmed Darwin program (nested)", "generic learner, HGB (nested)"],
                    "c  Subtracting the learned residual structure")
        fig.tight_layout()
        fig.savefig(FIG / "r10_residual_audit.png", dpi=160, facecolor=COL["surface"])
        plt.close(fig)
        log("wrote figure r10_residual_audit.png")
    vp = OUT / "r10_viability_voi.json"
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), facecolor=COL["surface"])
        _rel_series(axs[0], [v["all_chemicals"]["V0_vs_uncorrected"], v["all_chemicals"]["V1_vs_uncorrected"]],
                    ["no viability (V0)", "+ AB/LDH calls, scenario (V1)"],
                    f"a  All {v['all_chemicals']['n_chemicals']} chemicals, vs uncorrected")
        ns = v["ntp_subset"]
        _rel_series(axs[1], [ns["V0_prediction_time_vs_uncorrected"], ns["V1_pubchem_AB_LDH_calls_vs_uncorrected"],
                             ns["V2_ntp_chemical_summary_vs_uncorrected"], ns["V3_ntp_context_only_vs_uncorrected"]],
                    ["no viability (V0)", "+ AB/LDH calls, scenario (V1)", "+ NTP summary, scenario (V2)",
                     "+ NTP context wells only (V3)"],
                    f"b  {ns['n_chemicals']} chemicals with NTP per-well viability")
        fig.tight_layout()
        fig.savefig(FIG / "r10_viability_voi.png", dpi=160, facecolor=COL["surface"])
        plt.close(fig)
        log("wrote figure r10_viability_voi.png")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate residuals even if cached")
    ap.add_argument("--fresh", action="store_true", help="ignore cached search / corrector parts")
    ap.add_argument("--evals", type=int, default=2000, help="programs evaluated per real search")
    ap.add_argument("--nested-evals", type=int, default=1000)
    ap.add_argument("--plant-evals", type=int, default=500)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=200_000, help="row subsample for corrector fits")
    ap.add_argument("--stages", nargs="*", default=["residuals", "search", "power", "nested", "voi", "figure"])
    a = ap.parse_args()
    t0 = time.time()
    tasks, tr = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    if a.regen or not RESID.exists():
        df = regenerate(tasks, tr)
    else:
        df = pd.read_parquet(RESID)
    dtab = pd.read_parquet(DESIGN_TAB)
    integ = integrity(df)
    log(f"integrity vs trajectory_cv.json: {integ}")
    chem_tab, med = chemical_covariates([t for t in tasks if t.fold in FOLDS])
    ct = chem_tab.set_index("chem")
    chem_str = df.chem.astype(str)
    for col in ("cls", "cyto_AB", "cyto_LDH", "ntp_min_AB", "ntp_min_cell"):
        df[col] = chem_str.map(ct[col]).to_numpy()
    groups = df.chem.cat.codes.to_numpy()
    D = search_view(df)
    common = {"models": "v1 (primary; data/processed/models; nested selection of results/trajectory_cv.json)",
              "pilot_models_note": "exploratory pilots in colab/research/fran/_pilot used models_v2; not used here",
              "residual_definition": "ensemble mean prediction minus observed level mean (units: vehicle robust SD)",
              "n_residual_rows": int(len(df)), "n_chemicals": int(df.chem.nunique()),
              "rows_by_k": {str(k): int((df.k == k).sum()) for k in KS},
              "chemicals_by_fold": {str(f): int(df[df.fold == f].chem.nunique()) for f in FOLDS},
              "integrity_vs_trajectory_cv": integ}
    audit_p = OUT / "r10_residual_audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else {}
    if "search" in a.stages:
        audit["search"] = primary_search(df, D, groups, a)
    if "power" in a.stages:
        audit["power"] = power_analysis(df, D, groups, a)
    need_base = "nested" in a.stages or "voi" in a.stages
    base_feats = ["k", "logc", "div", "feat", "dist", "extrap", "above_min", "above_max", "pred", "pred_scale",
                  "nrep", "veh_z", "cls_code"]
    df["cls_code"] = pd.factorize(df.cls.astype(str))[0]
    base_cats = {"div", "feat", "cls_code"}
    if need_base:
        c0, i0 = nested_corrector(df, base_feats, base_cats, a.max_train, "V0_prediction_time", a.fresh)
    if "nested" in a.stages:
        corr, info = nested_darwin_correction(df, a)
        audit["secondary_nested_correction"] = {
            "darwin_confirmed_correction": {"per_test_fold": info, "vs_uncorrected": compare(df, corr)},
            "baseline_generic_learner_prediction_time_covariates": {
                "features": base_feats, "nested_choice": i0, "vs_uncorrected": compare(df, c0)}}
    if "voi" in a.stages:
        ctxf = ntp_context_features(df, dtab, chem_tab, med)
        df = pd.concat([df, ctxf], axis=1)
        ntp_chems = sorted(chem_tab[chem_tab.ntp_dtxsid.notna()].chem.astype(str))
        df["ntp_available"] = df.chem.astype(str).isin(ntp_chems).astype(float)
        arms = {"V1_pubchem_AB_LDH_calls": base_feats + ["cyto_AB", "cyto_LDH"],
                "V2_ntp_chemical_summary": base_feats + ["ntp_min_AB", "ntp_min_cell"],
                "V3_ntp_context_only": base_feats + list(ctxf.columns),
                "C_ntp_indicator_only": base_feats + ["ntp_available"]}
        corrs = {"V0_prediction_time": (c0, i0)}
        for name, feats in arms.items():
            corrs[name] = nested_corrector(df, feats, base_cats, a.max_train, name, a.fresh)
        voi = {"description": "Viability value of information: nested chemical-grouped residual corrector with vs "
                              "without viability information (R10, option D, part 3).", **common,
               "scenario_label": "prior viability screening scenario",
               "leakage_caveat": ("PubChem AB/LDH calls (AIDs 2284083, 2284068) are measured at DIV12 on the same NFA "
                                  "plates and aggregated over all concentrations, including the concentrations the "
                                  "forecast hides. Arms V1 and V2 therefore emulate a prior viability screen and are "
                                  "NOT evidence that viability improves the forecast. Arm V3 uses only the wells at "
                                  "the measured context concentrations."),
               "ntp_independence_audit": ntp_match_audit(tasks, chemical_covariates(tasks)[0]),
               "ntp_label": ("NTP DNT-DIVER 2018 'USEPA 91 neuron firing' per-well viability: an independent source "
                             "file, but most likely the viability readout of the same NFA plates (see "
                             "ntp_independence_audit); treated as a same-plate replication, not an independent "
                             "experiment."),
               "corrector": {"candidates": list(CANDIDATES), "selection": "leave-one-fold-out over the three "
                             "training folds (chemical-grouped), criterion = curve MAE after correction",
                             "max_train_rows": a.max_train, "base_features": base_feats},
               "arms": {"V0_prediction_time": "prediction-time covariates only (no viability)",
                        "V1_pubchem_AB_LDH_calls": "V0 + PubChem AB/LDH chemical-level hit calls (scenario, leaky)",
                        "V2_ntp_chemical_summary": "V0 + NTP per-well viability, min over all concentrations of the "
                                                   "median normalised response (scenario, leaky)",
                        "V3_ntp_context_only": "V0 + NTP per-well viability at the measured context concentrations "
                                               "only (min over context levels, value at the context level nearest "
                                               "to the query); no hidden-concentration information",
                        "C_ntp_indicator_only": "control: V0 + an indicator of NTP data availability (checks that "
                                                "V2/V3 gains are not a subset effect of the missing-value pattern)"},
               "nested_choice": {k: v[1] for k, v in corrs.items()},
               "all_chemicals": {
                   "n_chemicals": int(df.chem.nunique()),
                   "V0_vs_uncorrected": compare(df, c0),
                   "V1_vs_uncorrected": compare(df, corrs["V1_pubchem_AB_LDH_calls"][0]),
                   "V1_vs_V0": compare(df, corrs["V1_pubchem_AB_LDH_calls"][0], c0)},
               "ntp_subset": {"n_chemicals": len(ntp_chems), "chemicals": ntp_chems}}
        for name in corrs:
            voi["ntp_subset"][f"{name}_vs_uncorrected"] = compare(df, corrs[name][0], chems=ntp_chems)
            if name != "V0_prediction_time":
                voi["ntp_subset"][f"{name}_vs_V0"] = compare(df, corrs[name][0], c0, chems=ntp_chems)
        voi["elapsed_s"] = round(time.time() - t0, 1)
        (OUT / "r10_viability_voi.json").write_text(json.dumps(voi, indent=1, default=float), encoding="utf-8")
        log("wrote r10_viability_voi.json")
    audit.update({"description": "R10 honest-ceiling audit: Darwin-Cage search over NeuroTrajectory out-of-fold "
                                 "residuals with a power analysis (option D)", **common,
                  "engine": "neurotwin.audit.darwin (port of the team lead's Darwin-Cage search; see repo/NOTICE)",
                  "alphabet": {"categorical": CATS, "numeric_integer_scale": NUM_SCALE, "ops": list(ALPHABET.ops),
                               "divs": list(ALPHABET.divs), "qbins": list(ALPHABET.qbins),
                               "note": "information available at prediction time only; veh_z missing -> 0 in the "
                                       "search view", "n_veh_z_missing": int(df.veh_z.isna().sum())},
                  "protocol": {"search_folds": [f for f in FOLDS if f != PRIMARY_CONFIRM],
                               "confirm_fold": PRIMARY_CONFIRM, "rotation": "each fold in turn (sensitivity)",
                               "confirmation_rule": "confirm corr > z/sqrt(n_confirm_rows) AND chemical sign-flip "
                                                    "p < P(Z>z), z = 3, 4000 sign flips",
                               "evals_real": a.evals, "evals_nested": a.nested_evals, "evals_plant": a.plant_evals,
                               "plant_reps": a.reps, "plant_effects_vehicle_sd": list(EFFECTS),
                               "plants": {k: [list(x) for x in v] for k, v in PLANTS.items()}}})
    if "search" in audit and "power" in audit:
        s4 = audit["search"][str(PRIMARY_CONFIRM)]
        audit["primary"] = {
            "n_confirmed_programs_confirm_fold4": s4["n_confirmed"],
            "n_confirmed_column_sets_confirm_fold4": len(column_sets(s4["confirmed"])),
            "confirmed_column_sets_confirm_fold4": column_sets(s4["confirmed"]),
            "n_confirmed_single_atom_programs_confirm_fold4": int(sum(len(h["program"]) == 1 for h in s4["confirmed"])),
            "confirmed_single_atom_programs_confirm_fold4": [h["program"][0] for h in s4["confirmed"]
                                                             if len(h["program"]) == 1],
            "n_confirmed_programs_by_rotation": {c: v["n_confirmed"] for c, v in audit["search"].items()},
            "confirmed_column_sets_by_rotation": {c: column_sets(v["confirmed"]) for c, v in audit["search"].items()},
            "best_confirmed_confirm_fold4": s4["confirmed"][0] if s4["confirmed"] else None,
            "mde_power0.8_vehicle_sd_search_exact": {n: audit["power"][n]["mde_power0.8_exact"] for n in PLANTS},
            "mde_power0.8_vehicle_sd_oracle": {n: audit["power"][n]["mde_power0.8_oracle"] for n in PLANTS}}
        audit["structure_description"] = describe_structure(df)
    audit["elapsed_s_last_run"] = round(time.time() - t0, 1)
    audit_p.write_text(json.dumps(audit, indent=1, default=float), encoding="utf-8")
    log(f"wrote {audit_p}")
    if "figure" in a.stages:
        make_figures()


if __name__ == "__main__":
    main()
