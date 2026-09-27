"""R11: early-exit triage from DIV 5/7 recordings (preregistered: docs/prereg/r11_early_exit.md).

For every chemical of the outer folds 1-4, only DIV 5 and DIV 7 wells are visible. Four scores
(twin, twin + uncertainty, early-raw rule, persistence) estimate whether the complete DIV 5-12 assay
would call the chemical active (fixed rule of neurotwin.models.potency, BMR = 3 vehicle robust SD).
Chemicals scoring below a nested threshold (95 % sensitivity on the other three folds) exit at DIV 7.
Conceptual antecedent: the early-rejection filter of "Speaking to Silicon" (arXiv:2601.12032).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
from neurotwin.models.potency import BMR, dev_summary  # noqa: E402
from neurotwin.models.trajectory import ChemTask, interp_rows  # noqa: E402

PROTOCOL = ROOT / "repo/docs/prereg/r11_early_exit.md"
PROTOCOL_SHA = "c92b9476be82b0a8b2d29556e096047edc8ff314505b4658b7f02c5ecabbf159"
EARLY = [0, 1]                    # DIV 5, DIV 7 (indices into DIVS = [5, 7, 9, 12])
SENS_TARGET = 0.95
N_BOOT = 10_000
Z90 = 1.645
METHODS = ["twin", "twin_upper90", "early_raw", "persistence"]
BASELINES = ["early_raw", "persistence"]


def cp_ci(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def grid_of(t):
    return np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)


def score_from_summary(A):
    """Max over concentrations and features of |A_f| in BMR units (>= 1 <=> active)."""
    return float(np.abs(A).max() / BMR)


def early_task(t: ChemTask) -> ChemTask:
    keep = np.zeros(t.m.shape[1], bool)
    keep[EARLY] = True
    m = t.m & keep[None, :, None]
    return ChemTask(t.chem, t.fold, t.logc.copy(), (t.y * m).astype(np.float32), m, t.plate.copy(), t.label)


def scores_for(t: ChemTask, models, dev):
    from neurotwin.models.cnp import predict
    g = grid_of(t)
    ref = score_from_summary(dev_summary(interp_rows(t.levels, t._mu, t._ok, g)))
    # early-raw: the rule on the two visible DIVs
    raw = interp_rows(t.levels, t._mu[:, EARLY], t._ok[:, EARLY], g)
    s_raw = score_from_summary(raw.mean(1))
    # persistence: DIV 9 and 12 = DIV 7
    pers = np.concatenate([raw, raw[:, 1:2], raw[:, 1:2]], 1)
    s_pers = score_from_summary(dev_summary(pers))
    # twin: every concentration as context, DIV 9/12 masked
    te = early_task(t)
    preds = [predict(m, te, np.ones(len(te.logc), bool), g, device=dev) for m in models]
    mu = np.mean([p[0] for p in preds], 0)
    sig = np.mean([p[1] for p in preds], 0)
    A = dev_summary(mu)
    sA = np.sqrt((sig ** 2).sum(1)) / mu.shape[1]      # sd of the DIV mean, independence assumed
    s_twin = score_from_summary(A)
    s_up = float(((np.abs(A) + Z90 * sA).max()) / BMR)
    return {"reference_score": ref, "reference_active": bool(ref >= 1.0), "twin": s_twin,
            "twin_upper90": s_up, "early_raw": s_raw, "persistence": s_pers}


def nested_threshold(scores, active, folds, f):
    """Largest tau with sensitivity >= SENS_TARGET on the reference-active chemicals of the other folds."""
    a = np.sort(scores[(folds != f) & active])
    j = int(np.floor((1 - SENS_TARGET) * len(a)))
    return float(a[j])


def evaluate(scores, active, folds):
    tau = {f: nested_threshold(scores, active, folds, f) for f in np.unique(folds)}
    exit_ = np.array([s < tau[f] for s, f in zip(scores, folds)])
    tp = int((active & ~exit_).sum()); fn = int((active & exit_).sum())
    tn = int((~active & exit_).sum()); fp = int((~active & ~exit_).sum())
    return exit_, {"thresholds_by_fold": {int(k): round(v, 4) for k, v in tau.items()},
                   "sensitivity": round(tp / max(tp + fn, 1), 4), "sensitivity_ci95": cp_ci(tp, tp + fn),
                   "inactive_exit_rate": round(tn / max(tn + fp, 1), 4), "inactive_exit_ci95": cp_ci(tn, tn + fp),
                   "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                   "overall_exit_rate": round(float(exit_.mean()), 4),
                   "auroc": round(float(roc_auc_score(active, scores)), 4)}


def boot_auroc(y, s, rng, n=2000):
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or (~y[i]).all():
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def paired_exit_diff(active, exit_a, exit_b, rng, n=N_BOOT):
    inact = ~active
    d0 = exit_a[inact].mean() - exit_b[inact].mean()
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(active), len(active))
        m = inact[i]
        if not m.any():
            continue
        bs.append(exit_a[i][m].mean() - exit_b[i][m].mean())
    return {"diff_inactive_exit_rate": round(float(d0), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "n_boot": len(bs)}


def main():
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert sha == PROTOCOL_SHA, f"protocol changed after registration: {sha}"
    import torch
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "4")))
    from neurotwin.models.cnp import load_fold_models
    dev = "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    tasks = [t for t in tasks if t.fold in (1, 2, 3, 4)]
    rows, cache = [], {}
    for t in tasks:
        if t.fold not in cache:
            cache[t.fold] = load_fold_models(t.fold, dev)[0]
            assert cache[t.fold], f"no v1 models for fold {t.fold}"
        r = {"chemical": t.chem, "fold": int(t.fold), "epa_label": t.label}
        r.update(scores_for(t, cache[t.fold], dev))
        rows.append(r)
    folds = np.array([r["fold"] for r in rows])
    active = np.array([r["reference_active"] for r in rows])
    rng = np.random.default_rng(0)
    out = {"description": __doc__.strip().splitlines()[0], "protocol_file": "docs/prereg/r11_early_exit.md",
           "protocol_sha256": sha, "protocol_registered_utc": "2026-09-26T13:03:27Z",
           "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "antecedent": "Angulo de Lafuente, Veselov & Goodman, Speaking to Silicon, arXiv:2601.12032 (2026) - early-rejection filter (conceptual)",
           "models": "v1 cross-validated NeuroTrajectory (data/processed/models), ensemble mean, own held-out fold only",
           "n_chemicals": len(rows), "n_reference_active": int(active.sum()), "n_reference_inactive": int((~active).sum()),
           "underpowered": bool((~active).sum() < 15), "methods": {}}
    exits = {}
    for m in METHODS:
        s = np.array([r[m] for r in rows])
        exits[m], ent = evaluate(s, active, folds)
        ent["auroc_ci95"] = boot_auroc(active, s, rng)
        out["methods"][m] = ent
    best = max(BASELINES, key=lambda b: out["methods"][b]["inactive_exit_rate"])
    prim = paired_exit_diff(active, exits["twin"], exits[best], rng)
    safe = out["methods"]["twin"]["sensitivity_ci95"][0] >= 0.85
    out["primary"] = {"comparison": f"twin vs best baseline ({best})", **prim,
                      "safety_sensitivity_ci_lower_ge_0.85": bool(safe),
                      "twin_better": bool(prim["ci95"][0] > 0 and safe and not out["underpowered"])}
    out["secondary_twin_upper90_vs_best"] = paired_exit_diff(active, exits["twin_upper90"], exits[best], rng)
    # secondary: EPA DNT reference labels (thresholds unchanged)
    lab = np.array([r["epa_label"] in ("positive", "negative") for r in rows])
    ypos = np.array([r["epa_label"] == "positive" for r in rows])
    epa = {"n_chemicals": int(lab.sum()), "n_positive": int(ypos[lab].sum()), "n_negative": int((~ypos[lab]).sum()), "methods": {}}
    for m in METHODS:
        e = exits[m][lab]; y = ypos[lab]
        tp, fn = int((y & ~e).sum()), int((y & e).sum()); tn, fp = int((~y & e).sum()), int((~y & ~e).sum())
        epa["methods"][m] = {"positives_continuing": tp, "positives_exited": fn, "sensitivity_ci95": cp_ci(tp, tp + fn),
                             "negatives_exited": tn, "negatives_continuing": fp}
    out["secondary_epa_labels"] = epa
    ex = exits["twin"]
    out["operational_saving_assumptions"] = {
        "chemicals_exiting_at_DIV7_twin": int(ex.sum()), "fraction": round(float(ex.mean()), 4),
        "recording_days_saved_per_exit": 2, "recording_days_per_full_assay": 4,
        "culture_days_saved_per_exit": 5, "culture_days_per_full_assay": 12,
        "recording_sessions_saved_pct": round(float(100 * ex.mean() * 2 / 4), 2),
        "note": "Assumptions, not measured costs: an exiting chemical skips the DIV 9 and DIV 12 recordings and 5 of 12 culture days."}
    (ROOT / "repo/results/r11_early_exit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    import csv
    with open(ROOT / "repo/results/r11_early_exit_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + [f"exit_{m}" for m in METHODS])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, **{f"exit_{m}": bool(exits[m][i]) for m in METHODS}})
    print(json.dumps({k: out[k] for k in ("n_chemicals", "n_reference_active", "n_reference_inactive", "primary")}, indent=1))
    for m in METHODS:
        e = out["methods"][m]
        print(m, "sens", e["sensitivity"], e["sensitivity_ci95"], "inactive exit", e["inactive_exit_rate"], "AUROC", e["auroc"])


if __name__ == "__main__":
    main()
