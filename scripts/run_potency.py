"""R3: potency and activity recovered from only k measured concentrations, against INDEPENDENT references.

Why two references (v2 of this protocol): a reference computed by interpolating the same observed level
means that a method is given would favour interpolation by construction (it matches the reference exactly
at the measured levels). So:
(A) External reference: EPA's published tcplfit2 fits of the NFA AUC endpoints (independent pipeline).
    Chemical active if any AUC endpoint is active; chemical potency = log10 of the smallest BMD among its
    active endpoints. Compared by activity agreement (accuracy, Cohen kappa) and Spearman rank
    correlation of potencies among chemicals active in both (definitions differ, so ranks not values).
(B) Replicate-split internal reference: each chemical's wells are split into two halves by plate (by well
    when a chemical has a single plate). Methods see only half A at k levels; the reference potency is the
    same rule applied to the interpolated level means of half B at ALL levels. No shared noise.
Few-shot estimates: every method gets the same k levels of half A (seeded designs as in R2) and its
prediction on a 120-point grid over the tested range is passed through neurotwin.models.potency.
Cross-validated models only (a chemical is scored by models that never saw it). Paired bootstrap over
chemicals of the per-chemical composite error (|dBMC| log10 if active in both, 1 on activity
disagreement, 0 if both inactive). Wells saved = 1 - k / levels.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.models.cnp import load_fold_models, predict  # noqa: E402
from neurotwin.models.potency import BMR, chemical_potency, dev_summary  # noqa: E402
from neurotwin.models.trajectory import (AnalogKNN, ChemTask, interp_rows, predict_hill,  # noqa: E402
                                         predict_interp, split_context)
from run_trajectory_cv import designs, paired_boot  # noqa: E402

KS = [2, 3, 4]
METHODS = ["neurotrajectory", "loglinear_interp", "hill_per_endpoint", "analog_knn"]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def half_split(t: ChemTask):
    plates = np.unique(t.plate)
    if len(plates) >= 2:
        a_pl = set(plates[::2])
        sel = np.array([p in a_pl for p in t.plate])
    else:
        sel = np.zeros(len(t.logc), bool)
        for l in t.levels:
            idx = np.where(t.logc == l)[0]
            sel[idx[::2]] = True
    mk = lambda s: ChemTask(t.chem, t.fold, t.logc[s], t.y[s], t.m[s], t.plate[s], t.label)
    A, B = mk(sel), mk(~sel)
    if len(A.levels) != len(t.levels) or len(B.levels) != len(t.levels):
        return None, None
    return A, B


def epa_reference(root):
    r = pd.read_parquet(root / "data/processed/epa_nfa/reference_potency.parquet")
    a = r[r.analysis.astype(str).str.upper().eq("AUC")]
    w = pd.read_parquet(root / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna().drop_duplicates()
    ref = {}
    for chem, g in w.groupby("chemical"):
        x = a[a.dtxsid.isin(g.dtxsid)]
        if x.empty:
            continue
        act = x[x.active.astype(bool)]
        pot = float(np.log10(act.bmd_uM.clip(lower=1e-6).min())) if len(act) and act.bmd_uM.notna().any() else np.nan
        ref[chem] = {"active": bool(len(act) > 0), "bmc_log10": pot}
    return ref


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    epa = epa_reference(ROOT)
    rows = []
    for f in sorted({t.fold for t in tasks}):
        train = [t for t in tasks if t.fold != f]
        knn = AnalogKNN(train)
        models, _ = load_fold_models(f, dev)
        for t in [x for x in tasks if x.fold == f]:
            A, B = half_split(t)
            grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
            refB = chemical_potency(grid, dev_summary(interp_rows(B.levels, B._mu, B._ok, grid))) if A is not None else None
            fullA = chemical_potency(grid, dev_summary(interp_rows(t.levels, t._mu, t._ok, grid)))
            e = epa.get(t.chem)
            rows.append({"chemical": t.chem, "fold": f, "k": "all", "design": 0, "method": "observed_all_levels",
                         "est_active": fullA["active"], "est_bmc": fullA["bmc_log10"],
                         "refB_active": None, "refB_bmc": None, "epa_active": e["active"] if e else None,
                         "epa_bmc": e["bmc_log10"] if e else None, "n_levels": len(t.levels)})
            if A is None:
                continue
            for k in KS:
                if k >= len(t.levels):
                    continue
                for di, ctx_lv in enumerate(designs(t, k)):
                    ic, _ = split_context(A, ctx_lv)
                    preds = {"neurotrajectory": np.mean([predict(m, A, ic, grid, device=dev)[0] for m in models], 0),
                             "loglinear_interp": predict_interp(A, ic, grid),
                             "hill_per_endpoint": predict_hill(A, ic, grid),
                             "analog_knn": knn.predict(A, ic, grid)}
                    for name, P in preds.items():
                        est = chemical_potency(grid, dev_summary(P))
                        rows.append({"chemical": t.chem, "fold": f, "k": k, "design": di, "method": name,
                                     "est_active": est["active"], "est_bmc": est["bmc_log10"],
                                     "refB_active": refB["active"], "refB_bmc": refB["bmc_log10"],
                                     "epa_active": e["active"] if e else None, "epa_bmc": e["bmc_log10"] if e else None,
                                     "n_levels": len(t.levels)})
        print(f"fold {f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "repo/results/potency_r3_rows.csv", index=False)
    out = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(), "bmr_vehicle_sd": BMR,
           "n_chemicals": int(df.chemical.nunique()),
           "n_chemicals_with_epa_reference": int(df[df.method == "observed_all_levels"].epa_active.notna().sum()),
           "n_chemicals_replicate_split": int(df[df.method == "neurotrajectory"].chemical.nunique()),
           "observed_all_levels_vs_epa": None, "by_k": {}}

    def vs_epa(g):
        g = g[g.epa_active.notna()]
        ea, oa = g.epa_active.astype(bool), g.est_active.astype(bool)
        both = g[ea & oa & g.epa_bmc.notna() & g.est_bmc.notna()]
        rho = spearmanr(both.est_bmc, both.epa_bmc).statistic if len(both) > 5 else np.nan
        return {"activity_accuracy": round(float((ea == oa).mean()), 4), "activity_kappa": round(kappa(ea, oa), 4),
                "spearman_potency": round(float(rho), 4), "n_active_both": int(len(both)), "n": int(len(g))}

    out["observed_all_levels_vs_epa"] = vs_epa(df[df.method == "observed_all_levels"])
    for k in KS:
        dk = df[df.k == k]
        ent = {"wells_saved_median_pct": round(float(100 * np.median(1 - k / dk.drop_duplicates("chemical").n_levels)), 1),
               "vs_epa": {}, "vs_replicate_split": {}}
        comp = {}
        for name, g in dk.groupby("method"):
            gm = g.groupby("chemical").agg(est_active=("est_active", "mean"), est_bmc=("est_bmc", "median"),
                                           epa_active=("epa_active", "first"), epa_bmc=("epa_bmc", "first"))
            gm["est_active"] = gm.est_active >= 0.5
            ent["vs_epa"][name] = vs_epa(gm.reset_index())
            ra, ea = g.refB_active.astype(bool), g.est_active.astype(bool)
            err = np.where(ra & ea, (g.est_bmc - g.refB_bmc).abs(), np.where(ra == ea, 0.0, 1.0))
            both = g[ra & ea]
            ent["vs_replicate_split"][name] = {
                "activity_accuracy": round(float((ra == ea).mean()), 4), "activity_kappa": round(kappa(ra, ea), 4),
                "bmc_abs_err_log10_median": round(float((both.est_bmc - both.refB_bmc).abs().median()), 4),
                "frac_within_0p5_log": round(float(((both.est_bmc - both.refB_bmc).abs() <= 0.5).mean()), 4),
                "composite_mean": round(float(err.mean()), 4)}
            comp[name] = pd.Series(err, index=g.index).groupby(g.chemical).mean()
        base = [m for m in comp if m != "neurotrajectory"]
        best = min(base, key=lambda m: comp[m].mean())
        idx = comp["neurotrajectory"].index
        ent["replicate_split_best_baseline"] = best
        ent["replicate_split_paired_vs_best"] = paired_boot(comp["neurotrajectory"].loc[idx], comp[best].loc[idx])
        ent["replicate_split_paired_vs_all"] = {b: paired_boot(comp["neurotrajectory"].loc[idx], comp[b].loc[idx]) for b in base}
        out["by_k"][str(k)] = ent
        print(k, {m: (v["composite_mean"], ent["vs_epa"][m]["spearman_potency"], ent["vs_epa"][m]["activity_kappa"])
                  for m, v in ent["vs_replicate_split"].items()}, ent["replicate_split_paired_vs_best"], flush=True)
    print("observed all levels vs EPA:", out["observed_all_levels_vs_epa"])
    (ROOT / "repo/results/potency_r3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote potency_r3.json")


if __name__ == "__main__":
    main()
