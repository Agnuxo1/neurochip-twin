"""R3b (secondary, registered after R3 v2): potency from the model-completed concentration series.

Estimator 'neurotrajectory_fill': at the measured levels use the observed level means; at the tested
levels that were NOT measured use the NeuroTrajectory forecast (ensemble of the chemical's own fold);
linear interpolation in log dose between levels; then the same potency rule as R3 v2. Same references
(EPA tcplfit2 AUC fits; replicate-split internal reference), same designs (k = 2, 3, 4), same baselines.
Rationale recorded before running (colab/decisions.md): the dense-grid estimator of R3 v2 can create
spurious benchmark crossings between tested concentrations. R3 v2 is reported regardless.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.models.cnp import load_fold_models, predict  # noqa: E402
from neurotwin.models.potency import chemical_potency, dev_summary  # noqa: E402
from neurotwin.models.trajectory import (AnalogKNN, interp_rows, predict_hill,  # noqa: E402
                                         predict_interp, split_context)
from run_potency import epa_reference, half_split, kappa  # noqa: E402
from run_trajectory_cv import designs, paired_boot  # noqa: E402

KS = [2, 3, 4]


def fill_curve(models, A, ic, ctx_lv, grid, dev):
    L = A.levels
    measured = np.isin(L, ctx_lv)
    mu = A._mu.copy()
    ok = A._ok.copy()
    if (~measured).any():
        pred = np.mean([predict(m, A, ic, L[~measured], device=dev)[0] for m in models], 0)
        mu[~measured] = pred
        ok[~measured] = True
    return interp_rows(L, mu, ok, grid)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    epa = epa_reference(ROOT)
    rows = []
    for f in sorted({t.fold for t in tasks}):
        knn = AnalogKNN([t for t in tasks if t.fold != f])
        models, _ = load_fold_models(f, dev)
        for t in [x for x in tasks if x.fold == f]:
            A, B = half_split(t)
            if A is None:
                continue
            grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
            refB = chemical_potency(grid, dev_summary(interp_rows(B.levels, B._mu, B._ok, grid)))
            e = epa.get(t.chem)
            for k in KS:
                if k >= len(t.levels):
                    continue
                for di, ctx_lv in enumerate(designs(t, k)):
                    ic, _ = split_context(A, ctx_lv)
                    preds = {"neurotrajectory_fill": fill_curve(models, A, ic, ctx_lv, grid, dev),
                             "neurotrajectory_grid": np.mean([predict(m, A, ic, grid, device=dev)[0] for m in models], 0),
                             "loglinear_interp": predict_interp(A, ic, grid),
                             "hill_per_endpoint": predict_hill(A, ic, grid),
                             "analog_knn": knn.predict(A, ic, grid)}
                    for name, P in preds.items():
                        est = chemical_potency(grid, dev_summary(P))
                        rows.append({"chemical": t.chem, "k": k, "design": di, "method": name,
                                     "est_active": est["active"], "est_bmc": est["bmc_log10"],
                                     "refB_active": refB["active"], "refB_bmc": refB["bmc_log10"],
                                     "epa_active": e["active"] if e else None, "epa_bmc": e["bmc_log10"] if e else None,
                                     "n_levels": len(t.levels)})
        print(f"fold {f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "repo/results/potency_r3b_rows.csv", index=False)
    from scipy.stats import spearmanr
    out = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(), "by_k": {}}
    for k in KS:
        dk = df[df.k == k]
        ent, comp = {"vs_epa": {}, "vs_replicate_split": {}}, {}
        for name, g in dk.groupby("method"):
            gm = g.groupby("chemical").agg(est_active=("est_active", "mean"), est_bmc=("est_bmc", "median"),
                                           epa_active=("epa_active", "first"), epa_bmc=("epa_bmc", "first")).reset_index()
            gm = gm[gm.epa_active.notna()]
            ea, oa = gm.epa_active.astype(bool), gm.est_active >= 0.5
            both = gm[ea & oa & gm.epa_bmc.notna() & gm.est_bmc.notna()]
            ent["vs_epa"][name] = {"activity_accuracy": round(float((ea == oa).mean()), 4), "activity_kappa": round(kappa(ea, oa), 4),
                                   "spearman_potency": round(float(spearmanr(both.est_bmc, both.epa_bmc).statistic), 4),
                                   "n_active_both": int(len(both))}
            ra, ea2 = g.refB_active.astype(bool), g.est_active.astype(bool)
            err = np.where(ra & ea2, (g.est_bmc - g.refB_bmc).abs(), np.where(ra == ea2, 0.0, 1.0))
            ent["vs_replicate_split"][name] = {"composite_mean": round(float(err.mean()), 4),
                                               "activity_kappa": round(kappa(ra, ea2), 4)}
            comp[name] = pd.Series(err, index=g.index).groupby(g.chemical).mean()
        idx = comp["neurotrajectory_fill"].index
        ent["fill_paired_vs"] = {b: paired_boot(comp["neurotrajectory_fill"].loc[idx], comp[b].loc[idx])
                                 for b in comp if b != "neurotrajectory_fill"}
        out["by_k"][str(k)] = ent
        print(k, {m: (v["composite_mean"], ent["vs_epa"][m]["spearman_potency"], ent["vs_epa"][m]["activity_kappa"])
                  for m, v in ent["vs_replicate_split"].items()}, flush=True)
        print("   fill vs:", {b: (v["mean_diff"], v["ci95"]) for b, v in ent["fill_paired_vs"].items()}, flush=True)
    (ROOT / "repo/results/potency_r3b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote potency_r3b.json")


if __name__ == "__main__":
    main()
