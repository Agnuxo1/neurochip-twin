"""R4: chemical-level DNT hazard call from developmental trajectories vs the reproduced EPA rules.

Cohort: chemicals with an EPA reference label (dnt_ref_tbl2 intersection, as in
results/epa_baseline_repro.json). Features per chemical (from the developmental summary
A_f(c) = mean over DIV of the normalised effect, see neurotwin.models.potency):
  for each of the 17 features: max |A_f| over tested concentrations, signed A_f at the top
  concentration, and the BMC (log10 uM; top tested concentration + 1 if inactive).
Classifier: L2 logistic regression with balanced class weights, standardised features, C fixed
a priori (1.0); out-of-fold probabilities with the frozen chemical folds.
Variants: (full) features from all measured concentrations; (few-shot, k=3) features from the
NeuroTrajectory forecast given 3 spread concentrations (cross-validated models).
Comparators on the same chemicals: EPA DIV12 / AUC.3hit / Top2 hit-count rules (from the
reproduction; note Top-k features were selected in-sample by EPA).
Statistics: sensitivity/specificity with Clopper-Pearson 95 % CIs, balanced accuracy, AUROC with
chemical bootstrap CI, exact McNemar test vs each EPA rule. N is small (19 negatives): no
superiority claim is made unless the McNemar p < 0.05.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.models.potency import bmc_from_curve, dev_summary  # noqa: E402
from neurotwin.models.trajectory import interp_rows, split_context  # noqa: E402


def features_from_curve(grid, traj):
    A = dev_summary(traj)                       # (G, NF)
    bmc, _ = bmc_from_curve(grid, A)
    bmc = np.where(np.isfinite(bmc), bmc, grid.max() + 1.0)
    return np.concatenate([np.abs(A).max(0), A[-1], bmc])


def cp_ci(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def rates(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, fn = int((y & yhat).sum()), int((y & ~yhat).sum())
    tn, fp = int((~y & ~yhat).sum()), int((~y & yhat).sum())
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": round(se, 4), "sensitivity_ci95": cp_ci(tp, tp + fn),
            "specificity": round(sp, 4), "specificity_ci95": cp_ci(tn, tn + fp), "balanced_accuracy": round((se + sp) / 2, 4)}


def mcnemar(y, a, b):
    y, a, b = map(lambda v: np.asarray(v, bool), (y, a, b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"only_ours_correct": n01, "only_epa_correct": n10, "exact_p": round(float(p), 4)}


def main():
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    lab = [t for t in tasks if t.label in ("positive", "negative")]
    repro = json.loads((ROOT / "repo/results/epa_baseline_repro.json").read_text(encoding="utf-8"))
    # EPA per-chemical hit sums (reproduced) -> rule calls
    hits = pd.read_csv(ROOT / "data/processed/epa_nfa/upstream/Bioactivity_bin_tbl_comp_methods.csv") \
        if (ROOT / "data/processed/epa_nfa/upstream/Bioactivity_bin_tbl_comp_methods.csv").exists() else None
    rows = []
    for t in lab:
        grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
        full = features_from_curve(grid, interp_rows(t.levels, t._mu, t._ok, grid))
        rows.append({"chemical": t.chem, "fold": t.fold, "y": t.label == "positive", "x_full": full})
    # few-shot features from cross-validated forecasts (k = 3 spread levels)
    try:
        import torch
        from neurotwin.models.cnp import NeuroTrajectoryCNP, predict
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cache = {}
        for r, t in zip(rows, lab):
            if t.fold not in cache:
                from neurotwin.models.cnp import load_fold_models
                cache[t.fold] = load_fold_models(t.fold, dev)[0]
            idx = np.unique(np.round(np.linspace(0, len(t.levels) - 1, 3)).astype(int))
            ic, _ = split_context(t, t.levels[idx])
            grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
            P = np.mean([predict(m, t, ic, grid, device=dev)[0] for m in cache[t.fold]], 0)
            r["x_k3"] = features_from_curve(grid, P)
    except Exception as e:  # noqa: BLE001
        print("few-shot variant skipped:", e)
    df = pd.DataFrame(rows)
    # EPA per-chemical calls (reproduced rules) joined by DTXSID
    epa_calls = None
    if hits is not None:
        w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna().drop_duplicates()
        dtx = w.groupby("chemical").dtxsid.first()
        h = hits.groupby("dsstox_substance_id")[["div12.hitsum", "auc.hitsum", "top2.feature.hitsum"]].max()
        thr = {"DIV12": ("div12.hitsum", repro["models"]["DIV12"]["threshold_hits"]),
               "AUC.3hit": ("auc.hitsum", repro["models"]["AUC.3hit"]["threshold_hits"]),
               "Top2": ("top2.feature.hitsum", repro["models"]["Top2"]["threshold_hits"])}
        epa_calls = {}
        for rule, (col, th) in thr.items():
            name = "epa_" + rule
            df[name] = [(h.loc[dtx[c], col] >= th) if (c in dtx.index and dtx[c] in h.index) else np.nan for c in df.chemical]
            epa_calls[rule] = name
    out = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(),
           "n_positive": int(df.y.sum()), "n_negative": int((~df.y).sum()), "variants": {}}
    for var in [c for c in ("x_full", "x_k3") if c in df]:
        prob = np.zeros(len(df))
        for f in sorted(df.fold.unique()):
            tr, te = df.fold != f, df.fold == f
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            clf.fit(np.stack(df.x_full[tr] if var == "x_full" else df.x_full[tr]), df.y[tr])   # train on full-info features
            prob[te.values] = clf.predict_proba(np.stack(df[var][te]))[:, 1]
        yhat = prob >= 0.5
        rng = np.random.default_rng(0)
        aucs = []
        for _ in range(2000):
            i = rng.integers(0, len(df), len(df))
            if df.y.values[i].all() or (~df.y.values[i]).all():
                continue
            aucs.append(roc_auc_score(df.y.values[i], prob[i]))
        ent = {"rates": rates(df.y, yhat), "auroc": round(float(roc_auc_score(df.y, prob)), 4),
               "auroc_ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]}
        out["variants"][var.replace("x_", "")] = ent
        df["pred_" + var] = yhat
        if epa_calls is not None:
            ent["mcnemar_vs_epa"] = {}
            for rule, col in epa_calls.items():
                ok = df[col].notna()
                ent["mcnemar_vs_epa"][rule] = {"n_chemicals": int(ok.sum()),
                                               "ours": rates(df.y[ok], yhat[ok.values]),
                                               "epa_rule_same_chemicals": rates(df.y[ok], df[col][ok].astype(bool)),
                                               **mcnemar(df.y[ok], yhat[ok.values], df[col][ok].astype(bool))}
        print(var, ent, flush=True)
    out["epa_reference_rules_published_on_their_cohort"] = {k: repro["models"][k] for k in ("DIV12", "AUC.3hit", "Top2")}
    out["note"] = ("EPA rule results are reproduced on their reference cohort (86 pos / 19 neg); our cohort is the "
                   "subset of those chemicals present as tasks here. With 19 negatives no balanced-accuracy "
                   "difference of a few points can be significant; see McNemar where per-chemical EPA calls are available.")
    (ROOT / "repo/results/dnt_r4.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.drop(columns=[c for c in df.columns if c.startswith("x_")]).to_csv(ROOT / "repo/results/dnt_r4_calls.csv", index=False)
    print("wrote dnt_r4.json")


if __name__ == "__main__":
    main()
