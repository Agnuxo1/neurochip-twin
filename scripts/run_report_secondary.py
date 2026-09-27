"""Post-hoc secondary analyses requested by the internal scientific and jury reviews (2026-09-26).

Not preregistered; reported as secondary. Inputs are existing per-chemical outputs only (no model is run):
  1. R11 on the external EPA DNT labels: paired exact McNemar test of which reference-positive (and
     reference-negative) chemicals exit at DIV 7 under persistence vs the twin's upper 90 % band (and the
     twin point forecast), from results/r11_early_exit_rows.csv.
  2. R4 stratified by cytotoxicity (same chemical-level AlamarBlue/LDH hit definition as R5): sensitivity,
     specificity and balanced accuracy of the k = 3 call and of EPA's rules among non-cytotoxic chemicals,
     from results/dnt_r4_calls.csv.
Output: results/report_secondary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "repo/results"


def cp_ci(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def mcnemar(a, b):
    """a, b: boolean arrays (event under method a / b). Exact two-sided test on discordant pairs."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"n": int(len(a)), "events_a": int(a.sum()), "events_b": int(b.sum()),
            "only_a": only_a, "only_b": only_b, "exact_p": round(float(p), 4)}


def r11_epa():
    r = pd.read_csv(RES / "r11_early_exit_rows.csv")
    out = {}
    for lab in ("positive", "negative"):
        g = r[r.epa_label == lab]
        pers = g.exit_persistence.astype(bool).values
        out[lab] = {"n": int(len(g)),
                    "persistence_vs_twin_upper90": mcnemar(pers, g.exit_twin_upper90.astype(bool).values),
                    "persistence_vs_twin": mcnemar(pers, g.exit_twin.astype(bool).values)}
    out["reading"] = ("events = chemicals exiting at DIV 7; for positives an exit is a missed neurotoxicant, "
                      "for negatives an exit is a saving")
    return out


def cytotoxic_chemicals():
    w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).dropna().drop_duplicates()
    dtx = w.groupby("chemical").dtxsid.apply(set)
    v = pd.read_parquet(ROOT / "data/processed/epa_nfa/viability.parquet")
    v = v[v.source.str.startswith("PubChem")]
    act = v[v.hit_call.astype(object).eq(True) & ~v.hit_call_conflict.astype(bool)]
    hit = set(act.dtxsid)
    return {c: bool(s & hit) for c, s in dtx.items()}


def rates(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, fn = int((y & yhat).sum()), int((y & ~yhat).sum())
    tn, fp = int((~y & ~yhat).sum()), int((~y & yhat).sum())
    se, sp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": round(se, 4), "sensitivity_ci95": cp_ci(tp, tp + fn),
            "specificity": round(sp, 4), "specificity_ci95": cp_ci(tn, tn + fp), "balanced_accuracy": round((se + sp) / 2, 4)}


def r4_by_cyto():
    d = pd.read_csv(RES / "dnt_r4_calls.csv")
    cy = cytotoxic_chemicals()
    d["cytotoxic"] = d.chemical.map(cy).fillna(False).astype(bool)
    out = {"definition": "cytotoxic = any active, non-conflicting PubChem AlamarBlue (viability) or LDH (membrane damage) "
                         "call of the NFA assay (same rule as R5)", "strata": {}}
    for name, g in (("all", d), ("not_cytotoxic", d[~d.cytotoxic]), ("cytotoxic", d[d.cytotoxic])):
        ent = {"n": int(len(g)), "n_positive": int(g.y.sum()), "n_negative": int((~g.y.astype(bool)).sum()),
               "twin_k3": rates(g.y, g.pred_x_k3)}
        for rule in ("DIV12", "AUC.3hit", "Top2"):
            col = "epa_" + rule
            ok = g[col].notna()
            ent[f"epa_{rule}"] = rates(g.y[ok], g[col][ok].astype(bool))
            ent[f"epa_{rule}"]["n"] = int(ok.sum())
        out["strata"][name] = ent
    return out


def main():
    out = {"description": __doc__.strip().splitlines()[0], "status": "post hoc, secondary, not preregistered",
           "r11_epa_labels": r11_epa(), "r4_by_cytotoxicity": r4_by_cyto()}
    (RES / "report_secondary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["r11_epa_labels"], indent=1))
    for k, v in out["r4_by_cytotoxicity"]["strata"].items():
        print(k, v["n"], v["n_positive"], v["n_negative"], "twin", v["twin_k3"]["sensitivity"], v["twin_k3"]["specificity"],
              v["twin_k3"]["balanced_accuracy"], "| DIV12", v["epa_DIV12"]["balanced_accuracy"], "AUC3", v["epa_AUC.3hit"]["balanced_accuracy"], "Top2", v["epa_Top2"]["balanced_accuracy"])


if __name__ == "__main__":
    main()
