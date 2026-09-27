"""R11b evaluation: early-exit triage with the twin retrained on DIV 9/12-masked context
(preregistered: docs/prereg/r11b_early_exit_retrained.md). Reuses the R11 pipeline (run_early_exit.py) unchanged and
adds: twin R11b vs twin v1, a late-only (DIV 9/12) reference as a circularity control, and a
non-inferiority check of dose interpolation with full DIV context (k = 3, trajectory_cv designs).
CPU only (inference).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
import run_early_exit as R11  # noqa: E402
from neurotwin.models.potency import dev_summary  # noqa: E402
from neurotwin.models.trajectory import interp_rows, split_context  # noqa: E402

PROTOCOL = ROOT / "repo/docs/prereg/r11b_early_exit_retrained.md"
PROTOCOL_SHA = "3ed9b54de673ee23f0a7f5afdc1d88ed87069e76ca052875a634eb48d6eb1a51"
R11B_DIR = ROOT / "data/processed/models_r11b"
V1_DIR = ROOT / "data/processed/models"
LATE = [2, 3]
NONINF_MAX_PCT = 2.0


def late_reference(t):
    g = R11.grid_of(t)
    return R11.score_from_summary(interp_rows(t.levels, t._mu[:, LATE], t._ok[:, LATE], g).mean(1))


def noninferiority(tasks, v1, r11b, rng):
    import run_trajectory_cv as TCV
    from neurotwin.models.cnp import predict
    a, b = [], []                                   # per-chemical mean curve MAE: r11b, v1
    for t in tasks:
        if len(t.levels) <= 3:
            continue
        ea, eb = [], []
        for ctx_lv in TCV.designs(t, 3):
            ic, it = split_context(t, ctx_lv)
            lv = np.unique(t.logc[it])
            for models, acc in ((r11b[t.fold], ea), (v1[t.fold], eb)):
                mu = np.mean([predict(m, t, ic, lv, device="cpu")[0] for m in models], 0)
                acc.append(TCV.curve_and_well_mae(mu, t, it)[0])
        a.append(np.nanmean(ea)); b.append(np.nanmean(eb))
    a, b = np.array(a), np.array(b)
    d = a - b
    bs = np.array([d[i].mean() for i in (rng.integers(0, len(d), len(d)) for _ in range(R11.N_BOOT))])
    base = b.mean()
    lo, hi = np.percentile(bs, [2.5, 97.5]) / base * 100
    return {"k": 3, "n_chemicals": int(len(d)), "curve_mae_r11b": round(float(a.mean()), 4),
            "curve_mae_v1": round(float(base), 4), "rel_change_pct": round(float(d.mean() / base * 100), 2),
            "rel_change_ci95_pct": [round(float(lo), 2), round(float(hi), 2)],
            "noninferior_upper_le_2pct": bool(hi <= NONINF_MAX_PCT)}


def main():
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert sha == PROTOCOL_SHA, "protocol changed after registration"
    import torch
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "4")))
    from neurotwin.models.cnp import load_fold_models
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    tasks = [t for t in tasks if t.fold in (1, 2, 3, 4)]
    r11b = {f: load_fold_models(f, "cpu", R11B_DIR)[0] for f in (1, 2, 3, 4)}
    v1 = {f: load_fold_models(f, "cpu", V1_DIR)[0] for f in (1, 2, 3, 4)}
    assert all(len(r11b[f]) == 3 for f in r11b), "R11b models missing"
    rows = []
    for t in tasks:
        r = {"chemical": t.chem, "fold": int(t.fold), "epa_label": t.label}
        new = R11.scores_for(t, r11b[t.fold], "cpu")
        old = R11.scores_for(t, v1[t.fold], "cpu")
        r.update(new)
        r["twin_v1"] = old["twin"]
        r["late_reference_score"] = late_reference(t)
        r["late_reference_active"] = bool(r["late_reference_score"] >= 1.0)
        rows.append(r)
    folds = np.array([r["fold"] for r in rows])
    rng = np.random.default_rng(0)
    methods = R11.METHODS + ["twin_v1"]
    out = {"description": __doc__.strip().splitlines()[0], "protocol_file": "docs/prereg/r11b_early_exit_retrained.md",
           "protocol_sha256": sha, "protocol_registered_utc": "2026-09-26T13:15:00Z",
           "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "models": "R11b: data/processed/models_r11b (v1 recipe + DIV 9/12 context masking p=0.5); twin_v1: data/processed/models",
           "n_chemicals": len(rows), "references": {}}
    exits = {}
    for ref in ("reference_active", "late_reference_active"):
        active = np.array([r[ref] for r in rows])
        blk = {"n_active": int(active.sum()), "n_inactive": int((~active).sum()), "methods": {}}
        ex = {}
        for m in methods:
            s = np.array([r[m] for r in rows])
            ex[m], ent = R11.evaluate(s, active, folds)
            ent["auroc_ci95"] = R11.boot_auroc(active, s, rng)
            blk["methods"][m] = ent
        best = max(R11.BASELINES, key=lambda b: blk["methods"][b]["inactive_exit_rate"])
        prim = R11.paired_exit_diff(active, ex["twin"], ex[best], rng)
        safe = blk["methods"]["twin"]["sensitivity_ci95"][0] >= 0.85
        blk["twin_r11b_vs_best_baseline"] = {"best_baseline": best, **prim, "safety_ok": bool(safe),
                                             "twin_better": bool(prim["ci95"][0] > 0 and safe and blk["n_inactive"] >= 15)}
        blk["twin_r11b_vs_twin_v1"] = R11.paired_exit_diff(active, ex["twin"], ex["twin_v1"], rng)
        out["references"][ref] = blk
        exits[ref] = ex
    out["primary"] = out["references"]["reference_active"]["twin_r11b_vs_best_baseline"]
    out["noninferiority_dose_interpolation"] = noninferiority(tasks, v1, r11b, rng)
    (ROOT / "repo/results/r11b_early_exit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    import csv
    with open(ROOT / "repo/results/r11b_early_exit_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"primary": out["primary"], "noninf": out["noninferiority_dose_interpolation"],
                      "late": out["references"]["late_reference_active"]["twin_r11b_vs_best_baseline"]}, indent=1))
    for ref, blk in out["references"].items():
        for m, e in blk["methods"].items():
            print(ref, m, "sens", e["sensitivity"], "inactive exit", e["inactive_exit_rate"], "AUROC", e["auroc"])


if __name__ == "__main__":
    main()
