"""R5 selectivity and R10 failure analysis.

R5: does network disruption occur below cytotoxic concentrations for developmental neurotoxicants?
  network potency  = neurotwin potency rule on the observed level means (all tested concentrations);
  cytotox potency  = smallest AC50 (uM) of the active AlamarBlue / LDH calls of the NFA assay
                     (PubChem CCTE_Shafer_MEA_dev, chemical level - never the same well);
                     chemicals with no active call get the top tested concentration x 10 (censored);
  selectivity index SI = log10(cytotox AC50) - log10(network BMC); network-inactive chemicals get SI = 0.
  Reported: AUROC (chemical bootstrap CI) of SI, of network potency alone and of cytotoxic potency alone
  for the EPA DNT reference label; fraction 'specific network disruption' (SI >= 0.5) by label; N per stratum.
R10: per annotated chemical class (EPA annotation 'neuro.class' / 'Class'), the k=3 curve error of
  NeuroTrajectory and of the best baseline, the fraction of chemicals improved, the abstention rate
  (from results/conformal_r7_per_chemical.csv when present) and the count of 'no estimable' curves
  (chemicals with no feature reaching the benchmark response at any tested concentration).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
from neurotwin.models.potency import chemical_potency, dev_summary  # noqa: E402
from neurotwin.models.trajectory import interp_rows  # noqa: E402


def boot_auc(y, s, n=2000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main():
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna().drop_duplicates()
    dtx = w.groupby("chemical").dtxsid.apply(set)
    v = pd.read_parquet(ROOT / "data/processed/epa_nfa/viability.parquet")
    v = v[v.source.str.startswith("PubChem")]
    rows = []
    for t in tasks:
        grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
        pot = chemical_potency(grid, dev_summary(interp_rows(t.levels, t._mu, t._ok, grid)))
        vv = v[v.dtxsid.isin(dtx.get(t.chem, set()))]
        act = vv[vv.hit_call.astype(object).eq(True) & ~vv.hit_call_conflict.astype(bool)]
        cyto = float(np.log10(act.ac50_uM.min())) if len(act) and act.ac50_uM.notna().any() else float(t.levels.max() + 1.0)
        net = pot["bmc_log10"] if pot["active"] else np.nan
        rows.append({"chemical": t.chem, "label": t.label, "network_active": pot["active"], "network_bmc": net,
                     "cyto_log10_ac50": cyto, "cyto_active": bool(len(act)),
                     "si": (cyto - net) if pot["active"] else 0.0, "top_tested": float(t.levels.max())})
    D = pd.DataFrame(rows)
    L = D[D.label.isin(["positive", "negative"])].copy()
    y = (L.label == "positive").astype(int).values
    net_score = np.where(L.network_active, -(L.network_bmc.fillna(10)), -10.0)
    cyto_score = -L.cyto_log10_ac50.values
    r5 = {"description": "R5 selectivity index of network disruption vs chemical-level cytotoxicity",
          "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
          "auroc_selectivity_index": round(float(roc_auc_score(y, L.si)), 4), "auroc_si_ci95": boot_auc(y, L.si.values),
          "auroc_network_potency_only": round(float(roc_auc_score(y, net_score)), 4), "auroc_network_ci95": boot_auc(y, net_score),
          "auroc_cytotoxic_potency_only": round(float(roc_auc_score(y, cyto_score)), 4), "auroc_cyto_ci95": boot_auc(y, cyto_score),
          "specific_disruption_frac": {lab: round(float((g.si >= 0.5).mean()), 4) for lab, g in L.groupby("label")},
          "network_active_frac": {lab: round(float(g.network_active.mean()), 4) for lab, g in L.groupby("label")},
          "cyto_active_frac": {lab: round(float(g.cyto_active.mean()), 4) for lab, g in L.groupby("label")},
          "n_by_label_and_cyto": {f"{lab}|{'cytotoxic' if c else 'not_cytotoxic'}": int(len(g)) for (lab, c), g in L.groupby(["label", "cyto_active"])},
          "note": "Cytotoxicity is a chemical-level anchor (PubChem AB/LDH of the NFA assay), not a same-well viability readout."}
    (ROOT / "repo/results/r5_selectivity.json").write_text(json.dumps(r5, indent=2), encoding="utf-8")
    print("R5", {k: r5[k] for k in ("auroc_selectivity_index", "auroc_si_ci95", "auroc_network_potency_only", "auroc_cytotoxic_potency_only", "specific_disruption_frac")})

    # R10
    ann = pd.read_csv(ROOT / "data/neurotox_probe/nfa_refine/annotate_dnt_ref_chems.csv")
    cls = {}
    for _, r in ann.iterrows():
        c = r.get("neuro.class") if isinstance(r.get("neuro.class"), str) else r.get("Class")
        if isinstance(c, str):
            c = {"pyrethroid2": "pyrethroid", "pyrethoid": "pyrethroid"}.get(c, c)
            for key in (r.get("dsstox_substance_id"),):
                if isinstance(key, str):
                    cls[key] = c
    chem_cls = {chem: next((cls[d] for d in ds if d in cls), "unannotated") for chem, ds in dtx.items()}
    cv = pd.read_csv(ROOT / "repo/results/trajectory_cv_per_chemical.csv")
    k3 = cv[cv.k == 3].pivot_table(index="chemical", columns="method", values="curve_mae")
    base = [c for c in ("loglinear_interp", "analog_knn", "hill_per_endpoint", "context_mean", "zero") if c in k3]
    best = min(base, key=lambda c: k3[c].mean())
    k3["class"] = [chem_cls.get(c, "unannotated") for c in k3.index]
    ab_p = ROOT / "repo/results/conformal_r7_per_chemical.csv"
    ab = pd.read_csv(ab_p).groupby("chemical").abstain.mean() if ab_p.exists() else None
    no_est = D.set_index("chemical").network_active.eq(False)
    r10 = {"description": "R10 failure analysis by annotated chemical class (k = 3 measured concentrations)",
           "best_baseline": best, "classes": {}}
    for c, g in k3.groupby("class"):
        ent = {"n": int(len(g)), "neurotrajectory_curve_mae": round(float(g.neurotrajectory.mean()), 4),
               "baseline_curve_mae": round(float(g[best].mean()), 4),
               "frac_improved": round(float((g.neurotrajectory < g[best]).mean()), 4),
               "n_no_estimable_potency": int(no_est.reindex(g.index).fillna(False).sum())}
        if ab is not None:
            ent["abstention_rate"] = round(float(ab.reindex(g.index).mean()), 4)
        r10["classes"][c] = ent
    r10["n_no_estimable_total"] = int(no_est.sum())
    (ROOT / "repo/results/r10_failures.json").write_text(json.dumps(r10, indent=2), encoding="utf-8")
    print("R10", json.dumps(r10["classes"], indent=0)[:1500])


if __name__ == "__main__":
    main()
