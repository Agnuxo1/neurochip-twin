"""R6b DoseCompass-anchored: the second DoseCompass protocol, every strategy starts at the HIGHEST tested level.

Usage: python repo/scripts/run_dosecompass_anchored.py [--register-only] [--limit N --out <scratch.json>]
The protocol text (verbatim, with its SHA-256) and the implementation details are written to the output JSON
before any result is computed (--register-only stops there). v1 (run_dosecompass.py, r6_dosecompass.json) is
imported read-only: its planner, estimators, potency rule, composite error and calibrated (tau, ell) are reused
unchanged, and its results file is never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
import run_dosecompass as v1  # noqa: E402  (read-only reuse of the v1 functions)
from neurotwin.models.potency import BMR  # noqa: E402
from neurotwin.planner import full_information_potency, load_fold_models, sequential_design  # noqa: E402
from run_trajectory_cv import paired_boot  # noqa: E402

V1_JSON = ROOT / "repo/results/r6_dosecompass.json"

PROTOCOL = """- Every strategy starts by measuring the HIGHEST tested concentration (anchor), identical for all strategies. Budget B total levels (primary B=3; sensitivity B=2 and B=4).
- DoseCompass-anchored: after the anchor, chooses the remaining B-1 levels greedily by the same expected-information criterion and the same cross-fitted calibration (tau, ell) as v1, using only already-measured wells and the fold's own models (never hidden wells).
- Comparators: (a) fixed log-spaced (same as v1; it already contains the top level), (b) random-anchored: top + random B-1 of the remaining levels (200 seeded replicates), (c) unanchored random as in v1, (d) oracle-anchored best subset a posteriori (upper bound, labelled).
- Estimators: NeuroTrajectory forecast (ensemble of the fold) and log-linear interpolation, exactly as in v1; same reference potency definition and composite error as v1.
- Primary endpoint: B=3, NeuroTrajectory estimator, composite error, DoseCompass-anchored minus fixed log-spaced, paired bootstrap over chemicals (report mean diff, 95% CI). Secondary: vs random-anchored (isolates the value of the information criterion), activity accuracy/kappa, |dBMC| among actives, % within 0.3/0.5 log, wells saved.
- Multiplicity: state that this is the second DoseCompass protocol tested and that v1 is also reported."""

IMPLEMENTATION = [
    "Anchor = index L-1 of the chemical's sorted unique tested levels (every replicate well at that level, DIV 5-12).",
    "DoseCompass-anchored = neurotwin.planner.sequential_design(first_level = anchor, budget = 4) with the fold's "
    "NeuroTrajectory ensemble, S = 2000 samples, seed 0 (as v1); greedy choices do not depend on B, so the B = 2 and "
    "3 designs are prefixes of the B = 4 sequence. (tau, ell) per outer fold are read unchanged from "
    "r6_dosecompass.json planner.calibration_per_fold (v1 cross-fitting on other folds); no new calibration.",
    "Fixed log-spaced = v1 fixed_design(L, B) (log-spaced indices with the v1 median level substituted); it contains "
    "index L-1 for every L and B used here.",
    "Random-anchored = anchor + the first B-1 entries of a seeded permutation of the other levels; 200 replicates, "
    "seed = first 8 hex digits of SHA-256('random_anchored|<chemical>|<r>'); prefixes give B = 2, 3, 4. Metrics are "
    "reported as mean and 2.5 / 50 / 97.5 % percentiles over replicates; the per-chemical expected composite error "
    "(mean over replicates) enters the paired bootstrap.",
    "Unanchored random as in v1 = v1 'random' reproduced exactly (fixed median first level (L-1)//2 plus the v1 seeded "
    "orders SHA-256('random|<chemical>|<r>'), 200 replicates); 'unanchored' means not anchored at the top level.",
    "Oracle-anchored (upper bound, uses hidden wells) = per chemical, estimator and B, the B-subset containing the "
    "anchor with the lowest composite error against the reference.",
    "Estimators, reference potency, metrics and composite error = v1 functions estimate(), full_information_potency(), "
    "metrics(), composite(); paired bootstrap = run_trajectory_cv.paired_boot (4000 resamples, seed 0), "
    "DoseCompass-anchored minus each alternative.",
    "Wells saved = 1 - B / L (median %, same for every strategy) plus the actual fraction of wells not used by the "
    "DoseCompass-anchored and fixed designs (median %).",
    "Labelled non-preregistered additions (descriptive only): v1 DoseCompass (median first) recomputed from the "
    "orders stored in r6_dosecompass.json and compared with DoseCompass-anchored; recomputed v1 fixed / random / "
    "DoseCompass numbers checked against r6_dosecompass.json; planner Brier score of P(active) after B levels; "
    "design-position diagnostics.",
]

DEV_NOTE = ("The protocol and implementation details were written to this file (--register-only) and the protocol "
            "SHA-256 logged in colab/decisions.md before any R6b result was computed. A smoke test on 10 chemicals "
            "(--limit 2, scratch output outside repo/results) was then run to check code paths and runtime; the "
            "reported run (run_started_utc) keeps the registration time of the same protocol text (same SHA-256). The "
            "smoke test's console metrics were seen and its figure was viewed to check the layout; afterwards only "
            "the figure layout (panel spacing, legend columns) and this note were edited, not the planner, the "
            "strategies, the metrics, the implementation details or the protocol text.")
BUDGETS = v1.BUDGETS
PRIMARY_B = v1.PRIMARY_B
N_RANDOM = v1.N_RANDOM
S_PLAN = v1.S_PLAN
ESTIMATORS = v1.ESTIMATORS
PREREG = ["dosecompass_anchored", "fixed_logspaced", "random_anchored", "random_v1_median_first",
          "oracle_anchored_upper_bound"]
NONPREREG = ["dosecompass_v1_median_first"]


# ---------------------------------------------------------------- design rules (tested in tests/test_planner.py)
def anchor_index(L: int) -> int:
    return L - 1


def anchored_subsets(L: int, B: int) -> list:
    """Every B-subset of level indices that contains the anchor (top level)."""
    a = anchor_index(L)
    return [tuple(sorted((a,) + c)) for c in combinations([j for j in range(L) if j != a], B - 1)]


def random_anchored_orders(chem: str, L: int, n: int = N_RANDOM) -> list:
    """n seeded orders: anchor first, then a permutation of the other levels (prefix of length B = design)."""
    a = anchor_index(L)
    others = [j for j in range(L) if j != a]
    return [[a] + [int(j) for j in np.random.default_rng(v1.hseed("random_anchored", chem, r)).permutation(others)]
            for r in range(n)]


def random_v1_orders(chem: str, L: int, n: int = N_RANDOM) -> list:
    """Exactly the v1 random orders: median first level, then v1's seeded permutation of the others."""
    m0 = v1.first_index(L)
    others = [j for j in range(L) if j != m0]
    return [[m0] + [int(j) for j in np.random.default_rng(v1.hseed("random", chem, r)).permutation(others)]
            for r in range(n)]


def anchored_design(models, task, budget: int, device="cpu", tau=0.5, ell=1.0, n_samples=S_PLAN, seed=0):
    """DoseCompass-anchored: greedy EIG sequence starting at the top tested level. Only wells at levels already
    in the sequence are read when choosing the next one (sequential_design / plan_step)."""
    top = float(np.asarray(task.levels)[anchor_index(len(task.levels))])
    return sequential_design(models, task, top, budget, device, tau=tau, ell=ell, n_samples=n_samples, seed=seed)


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rep_distribution(rep_metrics: list) -> dict:
    dist = {}
    for key in rep_metrics[0]:
        v = np.array([m[key] for m in rep_metrics if m[key] is not None], float)
        dist[key] = {"mean": round(float(v.mean()), 4), "p2.5": round(float(np.percentile(v, 2.5)), 4),
                     "p50": round(float(np.percentile(v, 50)), 4), "p97.5": round(float(np.percentile(v, 97.5)), 4)}
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N chemicals of each fold")
    ap.add_argument("--out", default=str(ROOT / "repo/results/r6b_dosecompass_anchored.json"))
    ap.add_argument("--fig", default=str(ROOT / "repo/results/figures/dosecompass_r6b.png"))
    ap.add_argument("--register-only", action="store_true", help="write the protocol header and stop")
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from an existing JSON")
    a = ap.parse_args()
    out_path = Path(a.out)
    if a.figure_only:
        return make_figure(json.loads(out_path.read_text(encoding="utf-8")), Path(a.fig))
    t_start = time.time()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256(PROTOCOL.encode()).hexdigest()
    impl_sha = hashlib.sha256("\n".join(IMPLEMENTATION).encode()).hexdigest()
    v1_sha_start = file_sha256(V1_JSON)
    header = {"description": "R6b DoseCompass-anchored: second DoseCompass protocol; every strategy first measures "
                             "the highest tested concentration; retrospective validation on real EPA NFA wells",
              "protocol": PROTOCOL,
              "protocol_sha256": sha,
              "protocol_registered_utc": now,
              "implementation_details": IMPLEMENTATION,
              "implementation_details_sha256": impl_sha,
              "implementation_details_sha256_at_registration": impl_sha,
              "v1_reference": {"file": "repo/results/r6_dosecompass.json", "file_sha256_at_start": v1_sha_start},
              "status": "protocol_registered_results_pending"}
    if out_path.exists():                              # keep the first registration of the same protocol text
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if old.get("protocol_sha256") == sha and old.get("protocol_registered_utc"):
            header["protocol_registered_utc"] = old["protocol_registered_utc"]
            header["implementation_details_sha256_at_registration"] = old.get(
                "implementation_details_sha256_at_registration", old.get("implementation_details_sha256"))
            header["run_started_utc"] = now
    v1_res = json.loads(V1_JSON.read_text(encoding="utf-8"))
    assert v1_res["status"] == "complete", "v1 must be complete before R6b"
    header["v1_reference"]["protocol_sha256"] = v1_res["protocol_sha256"]
    v1.write_json(out_path, header)                    # protocol on disk before any result exists
    print("protocol registered", sha, flush=True)
    if a.register_only:
        return

    import torch
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "4")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    folds = sorted({t.fold for t in tasks})
    if a.limit:
        tasks = [t for f in folds for t in [x for x in tasks if x.fold == f][: a.limit]]
    models = {f: load_fold_models(ROOT / "data/processed/models", f, dev) for f in folds}
    ref = {t.chem: full_information_potency(t) for t in tasks}
    cal = {int(f): (float(v["tau"]), float(v["ell"])) for f, v in v1_res["planner"]["calibration_per_fold"].items()}
    v1_rows = {row["chemical"]: row for row in v1_res["per_chemical"]}
    print("calibration from v1", cal, flush=True)

    # ---------------------------------------------------------------- 1. estimates + DoseCompass-anchored design
    table, seqs, pact, eig1 = {}, {}, {}, {}
    for i, t in enumerate(tasks):
        L = len(t.levels)
        a_idx, m0 = anchor_index(L), v1.first_index(L)
        need = set()
        for B in BUDGETS:
            need.update(anchored_subsets(L, B))
            need.update(tuple(sorted((m0,) + c)) for c in combinations([j for j in range(L) if j != m0], B - 1))
        table[t.chem] = {idx: v1.estimate(t, idx, models[t.fold], dev) for idx in sorted(need)}
        tau, ell = cal[int(t.fold)]
        seq, st = anchored_design(models[t.fold], t, max(BUDGETS), dev, tau=tau, ell=ell, n_samples=S_PLAN, seed=0)
        order = [v1.level_index(t, v) for v in seq]
        assert order[0] == a_idx and len(set(order)) == len(order) == max(BUDGETS)
        seqs[t.chem] = order
        pact[t.chem] = [s.p_active for s in st]
        eig1[t.chem] = {f"{k:.4f}": round(v, 4) for k, v in st[0].eig_bits.items()}
        if i % 20 == 0:
            print(f"designs {i}/{len(tasks)} ({time.time()-t_start:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 2. strategies -> metrics
    chems = [t.chem for t in tasks]
    tk = {t.chem: t for t in tasks}
    ra = np.array([ref[c]["active"] for c in chems], bool)
    rb = np.array([ref[c]["bmc_log10"] for c in chems], float)
    Ls = np.array([len(tk[c].levels) for c in chems])
    ord_ra = {c: random_anchored_orders(c, len(tk[c].levels)) for c in chems}
    ord_v1 = {c: random_v1_orders(c, len(tk[c].levels)) for c in chems}
    have_v1 = all(c in v1_rows for c in chems)
    results, per_chem_comp = {}, {}
    for B in BUDGETS:
        eB = {}
        for est in ESTIMATORS:
            sel = {"dosecompass_anchored": [tuple(sorted(seqs[c][:B])) for c in chems],
                   "fixed_logspaced": [v1.fixed_design(len(tk[c].levels), B) for c in chems],
                   "oracle_anchored_upper_bound": []}
            for c in chems:
                cand = anchored_subsets(len(tk[c].levels), B)
                errs = [v1.composite(*table[c][k][est], ref[c]["active"], ref[c]["bmc_log10"]) for k in cand]
                sel["oracle_anchored_upper_bound"].append(cand[int(np.argmin(errs))])
            if have_v1:
                sel["dosecompass_v1_median_first"] = [tuple(sorted(v1_rows[c]["dosecompass_order_idx"][:B])) for c in chems]
            ent, comp = {}, {}
            for s, subsets in sel.items():
                ea = np.array([table[c][k][est][0] for c, k in zip(chems, subsets)], bool)
                eb = np.array([table[c][k][est][1] for c, k in zip(chems, subsets)], float)
                ent[s] = v1.metrics(ea, eb, ra, rb)
                comp[s] = v1.composite(ea, eb, ra, rb)
            for s, orders in (("random_anchored", ord_ra), ("random_v1_median_first", ord_v1)):
                rep_m, rep_c = [], []
                for r in range(N_RANDOM):
                    subsets = [tuple(sorted(orders[c][r][:B])) for c in chems]
                    ea = np.array([table[c][k][est][0] for c, k in zip(chems, subsets)], bool)
                    eb = np.array([table[c][k][est][1] for c, k in zip(chems, subsets)], float)
                    rep_m.append(v1.metrics(ea, eb, ra, rb))
                    rep_c.append(v1.composite(ea, eb, ra, rb))
                comp[s] = np.mean(rep_c, 0)
                ent[s] = {"over_200_replicates": rep_distribution(rep_m),
                          "expected_composite_error_mean": round(float(comp[s].mean()), 4)}
            boot = {f"dosecompass_anchored_minus_{s}": paired_boot(comp["dosecompass_anchored"], comp[s])
                    for s in ("fixed_logspaced", "random_anchored", "random_v1_median_first",
                              "oracle_anchored_upper_bound")}
            if have_v1:
                boot["NONPREREGISTERED_dosecompass_anchored_minus_dosecompass_v1_median_first"] = paired_boot(
                    comp["dosecompass_anchored"], comp["dosecompass_v1_median_first"])
            eB[est] = {"strategies": ent, "paired_bootstrap_composite": boot}
            per_chem_comp[(B, est)] = comp
        wells = {"dosecompass_anchored": [], "fixed_logspaced": []}
        for c in chems:
            t = tk[c]
            for s, idx in (("dosecompass_anchored", seqs[c][:B]), ("fixed_logspaced", v1.fixed_design(len(t.levels), B))):
                wells[s].append(1 - np.isin(t.logc, t.levels[list(idx)]).sum() / len(t.logc))
        eB["wells_saved_median_pct"] = round(float(100 * np.median(1 - B / Ls)), 1)
        eB["wells_saved_actual_median_pct"] = {s: round(float(100 * np.median(v)), 1) for s, v in wells.items()}
        pa = np.array([pact[c][B - 1] for c in chems])
        eB["planner_posterior_p_active_NONPREREGISTERED_diagnostic"] = {
            "brier_dosecompass_anchored": round(float(np.mean((pa - ra) ** 2)), 4),
            "brier_constant_base_rate": round(float(np.mean((ra.mean() - ra) ** 2)), 4),
            "mean_p_active": round(float(pa.mean()), 4), "reference_active_rate": round(float(ra.mean()), 4)}
        results[f"B={B}"] = eB
        cB = eB["neurotrajectory"]["strategies"]
        print(f"B={B}: composite NT " + ", ".join(
            f"{s}={cB[s]['composite_error_mean'] if 'composite_error_mean' in cB[s] else cB[s]['expected_composite_error_mean']}"
            for s in cB), flush=True)

    # ---------------------------------------------------------------- 3. endpoints and v1 cross-checks
    prim = results[f"B={PRIMARY_B}"]["neurotrajectory"]
    ps = prim["strategies"]
    secondary = {
        "definition": "B=3, NeuroTrajectory estimator unless stated; DoseCompass-anchored vs random-anchored (paired "
                      "bootstrap of composite error), activity accuracy / kappa, |dBMC| among actives, % within "
                      "0.3 / 0.5 log10, wells saved",
        "dosecompass_anchored_minus_random_anchored": prim["paired_bootstrap_composite"][
            "dosecompass_anchored_minus_random_anchored"],
        "random_anchored_expected_composite_error_mean": ps["random_anchored"]["expected_composite_error_mean"],
        "by_strategy": {s: {k: ps[s][k] for k in ("activity_accuracy", "activity_kappa", "bmc_abs_err_log10_median",
                                                   "bmc_abs_err_log10_mean", "frac_within_0p3_log",
                                                   "frac_within_0p5_log", "n_pairs_active_both")}
                        for s in ("dosecompass_anchored", "fixed_logspaced", "oracle_anchored_upper_bound")},
        "random_anchored_over_200_replicates_mean": {
            k: ps["random_anchored"]["over_200_replicates"][k]["mean"]
            for k in ("activity_accuracy", "activity_kappa", "bmc_abs_err_log10_median", "frac_within_0p3_log",
                      "frac_within_0p5_log")},
        "wells_saved_median_pct": results[f"B={PRIMARY_B}"]["wells_saved_median_pct"],
        "wells_saved_actual_median_pct": results[f"B={PRIMARY_B}"]["wells_saved_actual_median_pct"],
        "loglinear_interp_dosecompass_anchored_minus_fixed_logspaced": results[f"B={PRIMARY_B}"]["loglinear_interp"][
            "paired_bootstrap_composite"]["dosecompass_anchored_minus_fixed_logspaced"]}
    v1_prim = v1_res["primary_endpoint"]
    multiplicity = (f"R6b is the second DoseCompass protocol tested on these data (243 EPA NFA chemicals, same folds, "
                    f"same estimators and reference). The first, v1 / R6 (median first level; protocol SHA-256 "
                    f"{v1_res['protocol_sha256']}, repo/results/r6_dosecompass.json), is reported unchanged alongside: "
                    f"B=3 NeuroTrajectory composite error DoseCompass {v1_prim['dosecompass']} vs fixed log-spaced "
                    f"{v1_prim['fixed_logspaced']}, mean difference {v1_prim['paired_bootstrap']['mean_diff']} "
                    f"(95% CI {v1_prim['paired_bootstrap']['ci95']}). No multiplicity correction is applied; with two "
                    f"protocols a single nominal 95% CI excluding zero would be weaker evidence than it appears.")
    consistency = None
    if not a.limit:
        consistency = {"note": "NONPREREGISTERED check: v1 strategies recomputed in this run vs r6_dosecompass.json "
                               "(mean composite error; differences should be ~0 up to GPU nondeterminism)"}
        for B in BUDGETS:
            for est in ESTIMATORS:
                old = v1_res["results"][f"B={B}"][est]["strategies"]
                new = results[f"B={B}"][est]["strategies"]
                consistency[f"B={B}|{est}"] = {
                    "fixed_logspaced": [old["fixed_logspaced"]["composite_error_mean"],
                                        new["fixed_logspaced"]["composite_error_mean"]],
                    "random": [old["random"]["expected_composite_error_mean"],
                               new["random_v1_median_first"]["expected_composite_error_mean"]],
                    "dosecompass": [old["dosecompass"]["composite_error_mean"],
                                    new["dosecompass_v1_median_first"]["composite_error_mean"]]}
        consistency["max_abs_diff"] = round(float(max(abs(x - y) for k, v in consistency.items() if k.startswith("B=")
                                                      for x, y in v.values())), 4)
    v1_sha_end = file_sha256(V1_JSON)
    summary = {
        **header,
        "status": "complete" if not a.limit else f"smoke_test_limit_{a.limit}",
        "n_chemicals": len(chems), "n_reference_active": int(ra.sum()),
        "levels_per_chemical": {str(k): int(v) for k, v in zip(*np.unique(Ls, return_counts=True))},
        "bmr_vehicle_sd": BMR, "device": dev,
        "planner": {"samples": S_PLAN, "seed": 0, "calibration_source": "r6_dosecompass.json planner.calibration_per_fold",
                    "calibration_per_fold": {str(f): {"tau": v[0], "ell": v[1]} for f, v in cal.items()}},
        "primary_endpoint": {
            "definition": "B=3, NeuroTrajectory estimator, mean composite error, DoseCompass-anchored minus fixed "
                          "log-spaced, paired bootstrap over chemicals",
            "dosecompass_anchored": ps["dosecompass_anchored"]["composite_error_mean"],
            "fixed_logspaced": ps["fixed_logspaced"]["composite_error_mean"],
            "paired_bootstrap": prim["paired_bootstrap_composite"]["dosecompass_anchored_minus_fixed_logspaced"]},
        "secondary_endpoints": secondary,
        "multiplicity": multiplicity,
        "results": results,
        "v1_consistency_check": consistency,
        "development_note": DEV_NOTE,
        "per_chemical": [
            {"chemical": c, "fold": int(tk[c].fold), "label": tk[c].label, "n_levels": int(len(tk[c].levels)),
             "anchor_idx": anchor_index(len(tk[c].levels)),
             "anchor_level_log10uM": round(float(tk[c].levels[anchor_index(len(tk[c].levels))]), 4),
             "ref_active": bool(ref[c]["active"]),
             "ref_bmc_log10": None if not ref[c]["active"] else round(float(ref[c]["bmc_log10"]), 4),
             "dosecompass_anchored_order_idx": seqs[c],
             "planner_p_active_after_k_levels": [round(float(p), 4) for p in pact[c]],
             "first_step_eig_bits": eig1[c],
             "composite_B3": {est: {s: round(float(per_chem_comp[(3, est)][s][i]), 4) for s in per_chem_comp[(3, est)]}
                              for est in ESTIMATORS}}
            for i, c in enumerate(chems)],
        "limitations": [
            "Second protocol on the same 243 chemicals after v1 was seen (v1 showed that omitting the top level cost "
            "activity accuracy); R6b was motivated by that result, so it is not an independent confirmation.",
            "Retrospective: 'measuring' a level means revealing wells that were already recorded.",
            "Candidates are restricted to the concentrations EPA tested (usually 7); the oracle is the best subset of "
            "that grid containing the anchor, and uses hidden wells.",
            "(tau, ell) were calibrated in v1 with contexts starting at the median level; they are reused unchanged "
            "for top-anchored contexts (no recalibration).",
            "Greedy one-step EIG; no non-myopic planning. Reference potency is itself a noisy, interpolation-based "
            "call (3 replicate wells per level)."],
        "v1_reference": {**header["v1_reference"], "file_sha256_at_end": v1_sha_end,
                         "unchanged_during_run": v1_sha_end == v1_sha_start},
        "elapsed_s": round(time.time() - t_start, 1)}
    summary["diagnostics"] = diagnostics(json.loads(json.dumps(v1.clean(summary))))
    v1.write_json(out_path, summary)
    print(f"wrote {out_path} ({summary['elapsed_s']}s)", flush=True)
    if not a.limit:
        make_figure(json.loads(out_path.read_text(encoding="utf-8")), Path(a.fig))


def diagnostics(s: dict) -> dict:
    """Derived from the per-chemical rows (no new computation on data): where DoseCompass-anchored places its
    levels and the B = 3 composite error split by the reference activity call."""
    rows = s["per_chemical"]
    r = lambda v: round(float(v), 4)
    below = {}
    for k in (1, 2, 3):
        off = [row["anchor_idx"] - row["dosecompass_anchored_order_idx"][k] for row in rows]
        vals, cnt = np.unique(off, return_counts=True)
        below[f"pick_{k + 1}"] = {str(int(v)): int(c) for v, c in zip(vals, cnt)}
    ends = {}
    for B in BUDGETS:
        ds = [row["dosecompass_anchored_order_idx"][:B] for row in rows]
        ends[f"B={B}"] = {"frac_including_lowest_level": r(np.mean([0 in d for d in ds])),
                          "frac_including_median_level": r(np.mean([(row["n_levels"] - 1) // 2 in d
                                                                     for row, d in zip(rows, ds)])),
                          "frac_including_highest_level": r(np.mean([row["anchor_idx"] in d for row, d in zip(rows, ds)])),
                          "frac_identical_to_fixed_logspaced": r(np.mean([tuple(sorted(d)) == v1.fixed_design(row["n_levels"], B)
                                                                          for row, d in zip(rows, ds)]))}
    ra = np.array([row["ref_active"] for row in rows], bool)
    split = {}
    for est in ESTIMATORS:
        split[est] = {}
        for strat in rows[0]["composite_B3"][est]:
            v = np.array([row["composite_B3"][est][strat] for row in rows], float)
            split[est][strat] = {"reference_active": r(v[ra].mean()) if ra.any() else None,
                                 "reference_inactive": r(v[~ra].mean()) if (~ra).any() else None,
                                 "n_reference_active": int(ra.sum()), "n_reference_inactive": int((~ra).sum())}
    return {"source": "derived from per_chemical rows of this file (run_dosecompass_anchored.py diagnostics())",
            "levels_below_anchor_of_each_pick": below,
            "design_positions": ends,
            "composite_B3_by_reference_activity": split}


# ---------------------------------------------------------------- figure
INK, INK2, GRID, SURF = v1.INK, v1.INK2, v1.GRID, v1.SURF
COL = {"dosecompass_anchored": "#2a78d6", "fixed_logspaced": "#eb6834", "random_anchored": "#1baf7a",
       "random_v1_median_first": "#eda100", "dosecompass_v1_median_first": "#e87ba4",
       "oracle_anchored_upper_bound": "#52514e"}
LABEL = {"dosecompass_anchored": "DoseCompass-anchored", "fixed_logspaced": "Fixed log-spaced",
         "random_anchored": "Random-anchored (mean, 95% band)", "random_v1_median_first": "Random v1 (median first)",
         "dosecompass_v1_median_first": "DoseCompass v1 (median first, R6)",
         "oracle_anchored_upper_bound": "Oracle-anchored (upper bound)"}
STYLE = {"dosecompass_anchored": ("-", "o"), "fixed_logspaced": ("-", "s"), "random_anchored": ("-", "D"),
         "random_v1_median_first": (":", "v"), "dosecompass_v1_median_first": ("-.", "^"),
         "oracle_anchored_upper_bound": ("--", "o")}
SERIES = ["oracle_anchored_upper_bound", "random_v1_median_first", "random_anchored", "dosecompass_v1_median_first",
          "fixed_logspaced", "dosecompass_anchored"]


def _val(ent: dict, key: str):
    if "over_200_replicates" in ent:
        return ent["over_200_replicates"][key]["mean"]
    return ent[key]


def _curves(ax, res, est, key, band=True):
    for s in SERIES:
        if s not in res[f"B={BUDGETS[0]}"][est]["strategies"]:
            continue
        v = [_val(res[f"B={B}"][est]["strategies"][s], key) for B in BUDGETS]
        if band and s == "random_anchored":
            d = [res[f"B={B}"][est]["strategies"][s]["over_200_replicates"][key] for B in BUDGETS]
            ax.fill_between(BUDGETS, [x["p2.5"] for x in d], [x["p97.5"] for x in d], color=COL[s], alpha=0.18, lw=0)
        ls, mk = STYLE[s]
        thin = s in ("dosecompass_v1_median_first", "random_v1_median_first")
        ax.plot(BUDGETS, v, ls, color=COL[s], lw=1.4 if thin else 2, marker=mk, ms=6 if not thin else 5, mec=SURF,
                mew=1, label=LABEL[s], zorder=4 if s == "dosecompass_anchored" else 2)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("budget B (measured concentration levels, incl. the top anchor)", color=INK2, fontsize=8.5)


def make_figure(summary: dict, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 9.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.62, width_ratios=[1, 1, 1.1])
    res = summary["results"]
    axes, ymax = [], 0.0
    for k, est in enumerate(ESTIMATORS):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        v1._style(ax)
        _curves(ax, res, est, "composite_error_mean")
        ymax = max(ymax, ax.get_ylim()[1])
        ax.set_title(["A  Potency from the NeuroTrajectory forecast", "B  Potency from log-linear interpolation"][k],
                     loc="left", fontsize=10, color=INK)
        ax.set_ylabel("mean composite error vs full information\n(|dBMC| log10; activity miss = 1)", color=INK2, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, ymax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], frameon=False, fontsize=8.5, loc="lower center", ncol=3, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.94))
    ax = fig.add_subplot(gs[0, 2])
    v1._style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    rows = []
    for est, short in zip(ESTIMATORS, ("forecast", "interp")):
        bo = res[f"B={PRIMARY_B}"][est]["paired_bootstrap_composite"]
        for alt, name in (("fixed_logspaced", "fixed"), ("random_anchored", "random-anch."),
                          ("random_v1_median_first", "random v1"), ("oracle_anchored_upper_bound", "oracle-anch.")):
            b = bo[f"dosecompass_anchored_minus_{alt}"]
            rows.append((f"{short} - {name}", b["mean_diff"], b["ci95"], alt))
    for i, (lab_, m, ci, alt) in enumerate(rows[::-1]):
        ax.plot(ci, [i, i], color=COL[alt], lw=2, solid_capstyle="round")
        ax.plot([m], [i], STYLE[alt][1], color=COL[alt], ms=7, mec=SURF, mew=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8, color=INK)
    ax.set_xlabel("DoseCompass-anchored minus alternative\n(composite error; mean and paired bootstrap 95% CI;\n"
                  "< 0 favours DoseCompass-anchored)", color=INK2, fontsize=8.5)
    ax.set_title(f"C  Paired differences at B = {PRIMARY_B} (first row: primary)", loc="left", fontsize=10, color=INK)
    est = "neurotrajectory"
    for k, (key, title, ylab) in enumerate((
            ("activity_accuracy", "D  Activity call agreement (forecast)", "activity accuracy vs reference"),
            ("bmc_abs_err_log10_median", "E  Potency error among actives (forecast)",
             "median |dBMC| (log10), active in both"))):
        ax = fig.add_subplot(gs[1, k])
        v1._style(ax)
        _curves(ax, res, est, key)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    v1._style(ax)
    pos = summary["diagnostics"]["levels_below_anchor_of_each_pick"]
    xs = sorted({int(x) for p in pos.values() for x in p})
    w = 0.26
    for j, (pk, shade) in enumerate((("pick_2", "#2a78d6"), ("pick_3", "#86b6ef"), ("pick_4", "#cde2fb"))):
        ax.bar(np.array(xs) + (j - 1) * w, [pos[pk].get(str(x), 0) for x in xs], width=w - 0.03, color=shade, lw=0,
               label=f"level chosen {pk[-1]}{'nd' if pk == 'pick_2' else ('rd' if pk == 'pick_3' else 'th')}")
    ax.set_xticks(xs)
    ax.set_xlabel("levels below the top anchor", color=INK2, fontsize=9)
    ax.set_ylabel("chemicals", color=INK2, fontsize=9)
    ax.set_title("F  Where DoseCompass-anchored measures next", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
