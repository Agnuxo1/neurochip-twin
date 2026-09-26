"""R7: conformal calibration of NeuroTrajectory predictive intervals, with Mondrian strata and abstention.

All predictions are out-of-fold (a chemical is predicted only by models that never saw it).
Nonconformity score per held-out well entry: s = |y - mu| / scale (scale = ensemble predictive scale).
Cross-conformal: for each outer fold g, the quantile q_hat(1-alpha) is computed on the scores of the
chemicals of the OTHER folds (finite-sample corrected), per Mondrian stratum
(cytotoxicity regime x number of measured concentrations k), and applied to fold g.
Cytotoxicity regime (chemical level, never same well): 'cytotoxic' if the PubChem/invitrodb AlamarBlue
or LDH call for the NFA assay is active, 'non-cytotoxic' if both are inactive, else 'indeterminate'.
Reported: observed coverage (mean over held-out chemicals, bootstrap 95 % CI by chemical) and mean
interval width at alpha = 0.2 / 0.1 / 0.05, before (parametric) and after conformal calibration, with N per stratum.
Abstention: epistemic score = mean ensemble std of mu / mean predictive scale; threshold = 90th percentile
on the calibration folds; we report curve error and coverage for retained vs abstained predictions.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.models.cnp import Z90, NeuroTrajectoryCNP, predict  # noqa: E402
from neurotwin.models.trajectory import level_means, split_context  # noqa: E402
from run_trajectory_cv import designs  # noqa: E402

ALPHAS = [0.2, 0.1, 0.05]
KS = [1, 2, 3]


def zq(lik, alpha):
    return (math.sqrt(2) * _erfinv(1 - alpha)) if lik == "gaussian" else -math.log(alpha)


def _erfinv(y):
    from scipy.special import erfinv
    return float(erfinv(y))


def cyto_regime(root):
    v = pd.read_parquet(root / "data/processed/epa_nfa/viability.parquet")
    w = pd.read_parquet(root / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna().drop_duplicates()
    pc = v[v.source.str.contains("PubChem", case=False, na=False)] if "source" in v else v
    reg = {}
    for chem, g in w.groupby("chemical"):
        r = pc[pc.dtxsid.isin(g.dtxsid)]
        calls = r.hit_call.dropna().astype(str).str.lower().tolist() if len(r) else []
        conflict = bool(r.hit_call_conflict.fillna(False).astype(bool).any()) if "hit_call_conflict" in r else False
        if any(c in ("active", "1", "true", "hit") for c in calls) and not conflict:
            reg[chem] = "cytotoxic"
        elif calls and all(c in ("inactive", "0", "false", "no hit") for c in calls):
            reg[chem] = "non-cytotoxic"
        else:
            reg[chem] = "indeterminate"
    return reg


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    reg = cyto_regime(ROOT)
    recs, curve = [], []
    lik_by_fold = {}
    for f in sorted({t.fold for t in tasks}):
        from neurotwin.models.cnp import load_fold_models
        models, cfg0 = load_fold_models(f, dev)
        lik_by_fold[f] = cfg0["likelihood"]
        for t in [x for x in tasks if x.fold == f]:
            for k in KS:
                if k >= len(t.levels):
                    continue
                for di, ctx_lv in enumerate(designs(t, k)[:3]):
                    ic, it = split_context(t, ctx_lv)
                    lv = np.unique(t.logc[it])
                    preds = [predict(m, t, ic, lv, device=dev) for m in models]
                    mus = np.stack([p[0] for p in preds])
                    mu = mus.mean(0)
                    sc = np.sqrt(np.mean([p[1] ** 2 for p in preds], 0) + mus.var(0))
                    epi = float(mus.std(0).mean() / sc.mean())
                    idx = np.searchsorted(lv, t.logc[it])
                    mm = t.m[it]
                    s = (np.abs(t.y[it] - mu[idx]) / sc[idx])[mm]
                    recs.append({"chemical": t.chem, "fold": f, "k": k, "design": di, "regime": reg.get(t.chem, "indeterminate"),
                                 "lik": lik_by_fold[f], "scores": s.astype(np.float32), "scale": sc[idx][mm].astype(np.float32),
                                 "epi": epi})
                    lvm, obs, ok = level_means(t, it)
                    curve.append({"chemical": t.chem, "fold": f, "k": k, "design": di, "epi": epi,
                                  "curve_mae": float(np.abs(mu - obs)[ok].mean())})
        print(f"fold {f} predicted", flush=True)
    R = pd.DataFrame(recs)
    C = pd.DataFrame(curve)
    out = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(),
           "n_chemicals": int(R.chemical.nunique()),
           "chemicals_per_regime": R.drop_duplicates("chemical").regime.value_counts().to_dict(), "results": {}}
    rng = np.random.default_rng(0)

    def boot_mean(x):
        x = np.asarray(x)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    for alpha in ALPHAS:
        rows = []
        for g in sorted(R.fold.unique()):
            cal, te = R[R.fold != g], R[R.fold == g]
            for (k, regime), tg in te.groupby(["k", "regime"]):
                cg = cal[(cal.k == k) & (cal.regime == regime)]
                if len(cg) < 5:                                  # too few calibration chemicals -> pool regimes
                    cg = cal[cal.k == k]
                sc_all = np.concatenate(cg.scores.values)
                n = len(sc_all)
                q = float(np.quantile(sc_all, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
                for _, r in tg.iterrows():
                    z = zq(r.lik, alpha)
                    rows.append({"chemical": r.chemical, "k": k, "regime": regime,
                                 "cov_param": float((r.scores <= z).mean()), "width_param": float((2 * z * r.scale).mean()),
                                 "cov_conf": float((r.scores <= q).mean()), "width_conf": float((2 * q * r.scale).mean())})
        D = pd.DataFrame(rows).groupby(["chemical", "k", "regime"], as_index=False).mean()
        ent = {"nominal": 1 - alpha}
        for k in KS:
            dk = D[D.k == k]
            ent[f"k{k}"] = {"parametric": {"coverage": round(float(dk.cov_param.mean()), 4), "ci95": boot_mean(dk.cov_param),
                                           "width": round(float(dk.width_param.mean()), 3)},
                            "conformal": {"coverage": round(float(dk.cov_conf.mean()), 4), "ci95": boot_mean(dk.cov_conf),
                                          "width": round(float(dk.width_conf.mean()), 3)},
                            "by_regime": {rg: {"n_chemicals": int(len(x)), "conformal_coverage": round(float(x.cov_conf.mean()), 4),
                                               "width": round(float(x.width_conf.mean()), 3)} for rg, x in dk.groupby("regime")}}
        out["results"][f"alpha_{alpha}"] = ent
        print(alpha, {k: (ent[f'k{k}']['parametric']['coverage'], ent[f'k{k}']['conformal']['coverage']) for k in KS}, flush=True)
    # abstention (cross-fitted threshold)
    ab = []
    for g in sorted(C.fold.unique()):
        thr = float(np.percentile(C[C.fold != g].epi, 90))
        x = C[C.fold == g].copy()
        x["abstain"] = x.epi > thr
        ab.append(x)
    A = pd.concat(ab)
    out["abstention"] = {"rule": "abstain if epistemic score > 90th percentile of calibration folds",
                         "abstention_rate": round(float(A.abstain.mean()), 4),
                         "curve_mae_retained": round(float(A[~A.abstain].curve_mae.mean()), 4),
                         "curve_mae_abstained": round(float(A[A.abstain].curve_mae.mean()), 4),
                         "curve_mae_all": round(float(A.curve_mae.mean()), 4),
                         "by_k": {str(k): {"rate": round(float(a.abstain.mean()), 4),
                                           "mae_retained": round(float(a[~a.abstain].curve_mae.mean()), 4),
                                           "mae_abstained": round(float(a[a.abstain].curve_mae.mean()), 4)}
                                  for k, a in A.groupby("k")}}
    print("abstention", out["abstention"], flush=True)
    (ROOT / "repo/results/conformal_r7.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote conformal_r7.json")


if __name__ == "__main__":
    main()
