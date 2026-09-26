"""R6 DoseCompass: choose the next concentration by expected information, validated retrospectively.

Usage: python repo/scripts/run_dosecompass.py [--limit N --out <scratch.json>]   (--limit = smoke test)
The protocol below is written to the output JSON before any result is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from neurotwin.models.cnp import predict  # noqa: E402
from neurotwin.models.potency import BMR, chemical_potency, dev_summary  # noqa: E402
from neurotwin.models.trajectory import predict_interp, split_context  # noqa: E402
from neurotwin.planner import (full_information_potency, load_fold_models, log_score_bits,  # noqa: E402
                               mask_divs, member_predictions, posterior_from_predictions,
                               potency_codes, reference_grid, sequential_design)
from run_trajectory_cv import paired_boot  # noqa: E402

PROTOCOL = """R6 DoseCompass - retrospective sequential concentration selection on the EPA NFA dose x DIV data
(243 chemicals, frozen 5-fold chemical split). Registered before any result is computed.

1. Budget. B = 3 measured concentration levels per chemical (primary); B = 2 and B = 4 are sensitivity
   analyses. Measuring a level means using every replicate well at that level (DIV 5, 7, 9, 12).
2. Fixed first level, identical for every strategy: the median tested level, index (L-1)//2 of the
   chemical's sorted unique tested levels (lower median when L is even).
3. Strategies (each adds B-1 levels chosen among the chemical's tested, not-yet-measured levels):
   (a) DoseCompass (neurotwin.planner): greedy; at each step the level with the largest expected
       information about the full-information potency P = (active?, BMC), computed ONLY from the wells at
       already-measured levels and the NeuroTrajectory ensemble of the chemical's own fold (models that
       never saw it). S = 2000 plausible full-information datasets are sampled: measured levels fixed to
       their observed developmental summary; unmeasured levels = ensemble-member mean + tau * member SD
       * GP(0, squared-exponential over log10 dose, length ell) residual, independent across features.
       P is computed per sample with the reference rule; EIG(u) = plug-in mutual information (bits)
       between P (inactive or BMC in 0.25-log10 bins) and the outcome at u (max_f |A_f(u)| / BMR binned
       at 0.5, 1, 2). Ties: farther from measured levels, then lower concentration. Greedy choices do not
       depend on B, so the B = 2 and 3 designs are prefixes of the B = 4 sequence.
       Planner hyperparameters (tau, ell) are chosen per outer fold by cross-fitting on a grid
       tau in {1/32, 1/16, 1/8, 1/4, 1/2, 1, 2} x ell in {0.25, 0.5, 1, 2}; criterion = mean log2
       predictive probability of the reference potency bin (S = 1000) with contexts of 1, 2 and 3 levels
       (first level + a seeded random order of the others), computed ONLY on chemicals of the other four
       folds, each predicted by its own fold's models. All seeds derive from SHA-256 of the chemical name.
   (b) Fixed log-spaced: indices round(linspace(0, L-1, B)) (numpy rounding); if the first-level index is
       not among them it replaces the nearest one (ties: the lower index is replaced).
   (c) Random: 200 seeded random orders of the non-first levels per chemical (prefixes give B = 2, 3, 4).
       Each metric is reported as its mean and 2.5 / 50 / 97.5 % percentiles over the 200 replicates; the
       per-chemical expected composite error (mean over replicates) enters the paired bootstrap.
   (d) Oracle (upper bound, not a strategy): per chemical, estimator and B, the B-subset containing the
       first level with the lowest composite error against the reference; it uses the hidden wells.
   (e) DIV-budget variant of (a): the planner sees only DIV 5, 7 and 9 of the measured wells (DIV 12
       masked; the model forecasts it); final estimation uses all DIVs at the chosen levels; same (tau,
       ell) as (a).
4. Estimation after B levels (context = every well at the B levels): (i) NeuroTrajectory forecast: mean of
   the 3-seed ensemble of the chemical's fold on the 120-point grid spanning the tested range; (ii)
   log-linear interpolation of the B measured level means (flat outside). Potency rule
   neurotwin.models.potency: developmental summary = mean over DIV, BMR = 3 vehicle robust SD, BMC = first
   crossing, most sensitive feature.
5. Reference (full information): the same rule on interp_rows(all tested levels, level means) on the
   same grid (R3 convention).
6. Metrics per strategy x estimator x B: median and mean |dBMC| (log10) over chemicals active in both;
   fraction of those within 0.3 and 0.5 log10; activity accuracy and Cohen's kappa; per-chemical
   composite error (|dBMC| if active in both, 1 on activity disagreement, 0 if both inactive) and its
   mean; paired bootstrap over chemicals (4000 resamples, seed 0) of the composite error, DoseCompass
   minus each alternative. Wells saved = 1 - B / L (median %), plus actual wells saved by DoseCompass.
7. Primary endpoint: B = 3, NeuroTrajectory estimator, mean composite error, DoseCompass minus fixed
   log-spaced (paired bootstrap 95 % CI). Every other comparison is secondary.
8. Diagnostic: Brier score of the planner's posterior P(active) after B levels vs the reference call."""

DEV_NOTE = ("Before the reported run, the pipeline was executed once as a smoke test on 10 chemicals (first 2 of "
            "each fold, --limit 2) to check code paths and runtime; its figure, which shows smoke-test metrics, was "
            "viewed to check the layout. Afterwards only the figure layout (legend position, panel spacing, label "
            "text) and this note were edited; the planner, the calibration grid, the strategies, the metrics and "
            "the protocol text were not changed. One planner diagnostic on a single fold-0 chemical at the default "
            "tau = 1 (all samples active for a reference-inactive chemical) was seen before the tau grid was "
            "fixed; it motivated extending the grid down to 1/32. A first full run, started when the protocol was "
            "registered (protocol_registered_utc), ended before writing any result and its console output was not "
            "kept; the run reported here (run_started_utc) uses the same protocol text (same SHA-256). The only code "
            "changes in between were loading the fold models through neurotwin.models.cnp.load_model (same "
            "architecture and weights) and an explicit error when no level has been measured; neither can change "
            "a number.")
TAUS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0]
ELLS = [0.25, 0.5, 1.0, 2.0]
BUDGETS = [2, 3, 4]
PRIMARY_B = 3
N_RANDOM = 200
S_PLAN = 2000
S_CAL = 1000
ESTIMATORS = ["neurotrajectory", "loglinear_interp"]
STRATEGIES = ["dosecompass", "dosecompass_div5to9", "fixed_logspaced", "random", "oracle_upper_bound"]


def hseed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def first_index(L: int) -> int:
    return (L - 1) // 2


def fixed_design(L: int, B: int) -> tuple:
    idx = sorted({int(i) for i in np.round(np.linspace(0, L - 1, B))})
    m0 = first_index(L)
    if m0 not in idx:
        j = min(range(len(idx)), key=lambda k: (abs(idx[k] - m0), idx[k]))
        idx[j] = m0
    assert len(set(idx)) == B
    return tuple(sorted(idx))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def composite(est_active, est_bmc, ref_active, ref_bmc):
    ea, ra = np.asarray(est_active, bool), np.asarray(ref_active, bool)
    d = np.abs(np.nan_to_num(np.asarray(est_bmc, float)) - np.nan_to_num(np.asarray(ref_bmc, float)))
    return np.where(ea & ra, d, np.where(ea == ra, 0.0, 1.0))


def metrics(ea, eb, ra, rb) -> dict:
    ea, ra = np.asarray(ea, bool), np.asarray(ra, bool)
    eb, rb = np.asarray(eb, float), np.asarray(rb, float)
    both = ea & ra
    err = np.abs(eb[both] - rb[both])
    r = lambda v: round(float(v), 4)
    return {"bmc_abs_err_log10_median": r(np.median(err)) if len(err) else None,
            "bmc_abs_err_log10_mean": r(err.mean()) if len(err) else None,
            "frac_within_0p3_log": r((err <= 0.3).mean()) if len(err) else None,
            "frac_within_0p5_log": r((err <= 0.5).mean()) if len(err) else None,
            "n_pairs_active_both": int(both.sum()),
            "activity_accuracy": r((ea == ra).mean()), "activity_kappa": r(kappa(ra, ea)),
            "composite_error_mean": r(composite(ea, eb, ra, rb).mean())}


def estimate(t, idx, models, dev):
    grid = reference_grid(t.levels)
    ic, _ = split_context(t, t.levels[list(idx)])
    mu = np.mean([predict(m, t, ic, grid, device=dev)[0] for m in models], 0)
    out = {}
    for name, P in (("neurotrajectory", mu), ("loglinear_interp", predict_interp(t, ic, grid))):
        p = chemical_potency(grid, dev_summary(P))
        out[name] = (bool(p["active"]), float(p["bmc_log10"]))
    return out


def level_index(t, lv) -> int:
    return int(np.argmin(np.abs(t.levels.astype(np.float64) - lv)))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N chemicals of each fold")
    ap.add_argument("--out", default=str(ROOT / "repo/results/r6_dosecompass.json"))
    ap.add_argument("--fig", default=str(ROOT / "repo/results/figures/dosecompass_r6.png"))
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from an existing JSON")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="recompute the 'diagnostics' block from the per-chemical rows of an existing JSON")
    a = ap.parse_args()
    out_path = Path(a.out)
    if a.figure_only:
        return make_figure(json.loads(out_path.read_text(encoding="utf-8")), Path(a.fig))
    if a.diagnostics_only:                             # derived from the per-chemical rows already in the JSON
        s = json.loads(out_path.read_text(encoding="utf-8"))
        s["diagnostics"] = diagnostics(s)
        write_json(out_path, s)
        return print(f"diagnostics added to {out_path}")
    t_start = time.time()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256(PROTOCOL.encode()).hexdigest()
    header = {"description": "R6 DoseCompass: next concentration by expected information about potency, "
                             "retrospective validation on real EPA NFA wells",
              "protocol": PROTOCOL,
              "protocol_sha256": sha,
              "protocol_registered_utc": now,
              "status": "protocol_registered_results_pending"}
    if out_path.exists():                              # keep the first registration of the same protocol text
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if old.get("protocol_sha256") == sha and old.get("protocol_registered_utc"):
            header["protocol_registered_utc"] = old["protocol_registered_utc"]
            header["run_started_utc"] = now
    write_json(out_path, header)                       # protocol on disk before any result exists
    print("protocol registered", header["protocol_sha256"][:12], flush=True)

    import torch
    torch.set_num_threads(int(__import__("os").environ.get("NT_THREADS", "4")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    folds = sorted({t.fold for t in tasks})
    if a.limit:
        tasks = [t for f in folds for t in [x for x in tasks if x.fold == f][: a.limit]]
    models = {f: load_fold_models(ROOT / "data/processed/models", f, dev) for f in folds}
    ref = {t.chem: full_information_potency(t) for t in tasks}

    # ---------------------------------------------------------------- 1. cross-fitted planner calibration
    combos = [(tau, ell) for tau in TAUS for ell in ELLS]
    cal = np.zeros((len(tasks), 3, len(combos)))
    for i, t in enumerate(tasks):
        L = len(t.levels)
        m0 = first_index(L)
        others = [j for j in range(L) if j != m0]
        perm = np.random.default_rng(hseed("calib", t.chem)).permutation(others)
        r = ref[t.chem]
        target, _ = potency_codes(np.array([r["active"]]), np.array([r["bmc_log10"]]),
                                  float(t.levels.min()), float(t.levels.max()))
        for k in range(3):
            idx = [m0] + [int(j) for j in perm[:k]]
            ic, _ = split_context(t, t.levels[idx])
            mu_m, sd_m = member_predictions(models[t.fold], t, ic, t.levels, dev)
            for c, (tau, ell) in enumerate(combos):
                post = posterior_from_predictions(t, ic, t.levels, mu_m, sd_m, tau, ell, S_CAL, seed=1)
                cal[i, k, c] = log_score_bits(post.codes, int(target[0]), post.n_codes)
        if i % 40 == 0:
            print(f"calibration {i}/{len(tasks)} ({time.time()-t_start:.0f}s)", flush=True)
    fold_arr = np.array([t.fold for t in tasks])
    per_fold = {}
    for f in folds:
        score = cal[fold_arr != f].mean((0, 1))
        b = int(np.argmax(score))
        per_fold[f] = {"tau": combos[b][0], "ell": combos[b][1], "mean_log2_score_other_folds": round(float(score[b]), 4),
                       "on_grid_boundary": bool(combos[b][0] in (TAUS[0], TAUS[-1]) or combos[b][1] in (ELLS[0], ELLS[-1])),
                       "table_mean_log2_score": {f"tau={c[0]:g},ell={c[1]:g}": round(float(s), 4)
                                                 for c, s in zip(combos, score)}}
    print("calibration", {f: (v["tau"], v["ell"]) for f, v in per_fold.items()}, flush=True)

    # ---------------------------------------------------------------- 2. every design containing the first level
    table, seqs, steps_div, pact = {}, {}, {}, {}
    for i, t in enumerate(tasks):
        L = len(t.levels)
        m0 = first_index(L)
        others = [j for j in range(L) if j != m0]
        tab = {}
        for B in BUDGETS:
            for comb in combinations(others, B - 1):
                idx = tuple(sorted((m0,) + comb))
                tab[idx] = estimate(t, idx, models[t.fold], dev)
        table[t.chem] = tab
        # ------------------------------------------------------------ 3. DoseCompass (full data and DIV budget)
        kw = dict(tau=per_fold[t.fold]["tau"], ell=per_fold[t.fold]["ell"], n_samples=S_PLAN, seed=0)
        seq, st = sequential_design(models[t.fold], t, float(t.levels[m0]), max(BUDGETS), dev, **kw)
        seq_d, st_d = sequential_design(models[t.fold], t, float(t.levels[m0]), max(BUDGETS), dev,
                                        planner_task=mask_divs(t), **kw)
        seqs[t.chem] = {"dosecompass": [level_index(t, v) for v in seq],
                        "dosecompass_div5to9": [level_index(t, v) for v in seq_d]}
        pact[t.chem] = {"dosecompass": [s.p_active for s in st], "dosecompass_div5to9": [s.p_active for s in st_d],
                        "first_step_eig_bits": {f"{k:.4f}": round(v, 4) for k, v in st[0].eig_bits.items()}}
        if i % 20 == 0:
            print(f"designs {i}/{len(tasks)} ({time.time()-t_start:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 4. strategies -> per-chemical estimates
    chems = [t.chem for t in tasks]
    tk = {t.chem: t for t in tasks}
    ra = np.array([ref[c]["active"] for c in chems], bool)
    rb = np.array([ref[c]["bmc_log10"] for c in chems], float)
    Ls = np.array([len(tk[c].levels) for c in chems])
    results = {}
    per_chem_comp = {}
    orders = {c: [np.random.default_rng(hseed("random", c, r)).permutation(
        [j for j in range(len(tk[c].levels)) if j != first_index(len(tk[c].levels))]) for r in range(N_RANDOM)]
        for c in chems}
    for B in BUDGETS:
        eB = {}
        for est in ESTIMATORS:
            sel = {}
            for s in ("dosecompass", "dosecompass_div5to9"):
                sel[s] = [tuple(sorted(seqs[c][s][:B])) for c in chems]
            sel["fixed_logspaced"] = [fixed_design(len(tk[c].levels), B) for c in chems]
            sel["oracle_upper_bound"] = []
            for c in chems:
                cand = [k for k in table[c] if len(k) == B]
                errs = [composite(*table[c][k][est], ref[c]["active"], ref[c]["bmc_log10"]) for k in cand]
                sel["oracle_upper_bound"].append(cand[int(np.argmin(errs))])
            ent, comp = {}, {}
            for s, subsets in sel.items():
                ea = np.array([table[c][k][est][0] for c, k in zip(chems, subsets)], bool)
                eb = np.array([table[c][k][est][1] for c, k in zip(chems, subsets)], float)
                ent[s] = metrics(ea, eb, ra, rb)
                comp[s] = composite(ea, eb, ra, rb)
            # random: 200 seeded orders per chemical
            rep_metrics, rep_comp = [], []
            for r in range(N_RANDOM):
                subsets = [tuple(sorted([first_index(len(tk[c].levels))] + [int(j) for j in orders[c][r][:B - 1]])) for c in chems]
                ea = np.array([table[c][k][est][0] for c, k in zip(chems, subsets)], bool)
                eb = np.array([table[c][k][est][1] for c, k in zip(chems, subsets)], float)
                rep_metrics.append(metrics(ea, eb, ra, rb))
                rep_comp.append(composite(ea, eb, ra, rb))
            comp["random"] = np.mean(rep_comp, 0)
            dist = {}
            for key in rep_metrics[0]:
                v = np.array([m[key] for m in rep_metrics if m[key] is not None], float)
                dist[key] = {"mean": round(float(v.mean()), 4),
                             "p2.5": round(float(np.percentile(v, 2.5)), 4),
                             "p50": round(float(np.percentile(v, 50)), 4),
                             "p97.5": round(float(np.percentile(v, 97.5)), 4)}
            ent["random"] = {"over_200_replicates": dist,
                             "expected_composite_error_mean": round(float(comp["random"].mean()), 4)}
            boot = {f"dosecompass_minus_{s}": paired_boot(comp["dosecompass"], comp[s])
                    for s in ("fixed_logspaced", "random", "oracle_upper_bound")}
            boot["dosecompass_div5to9_minus_fixed_logspaced"] = paired_boot(comp["dosecompass_div5to9"], comp["fixed_logspaced"])
            boot["dosecompass_div5to9_minus_dosecompass"] = paired_boot(comp["dosecompass_div5to9"], comp["dosecompass"])
            eB[est] = {"strategies": ent, "paired_bootstrap_composite": boot}
            per_chem_comp[(B, est)] = comp
        wells_dc = []
        for c in chems:
            t = tk[c]
            used = np.isin(t.logc, t.levels[list(seqs[c]["dosecompass"][:B])]).sum()
            wells_dc.append(1 - used / len(t.logc))
        eB["wells_saved_median_pct"] = round(float(100 * np.median(1 - B / Ls)), 1)
        eB["wells_saved_actual_dosecompass_median_pct"] = round(float(100 * np.median(wells_dc)), 1)
        pa = np.array([pact[c]["dosecompass"][B - 1] for c in chems])
        pd_ = np.array([pact[c]["dosecompass_div5to9"][B - 1] for c in chems])
        eB["planner_posterior_p_active"] = {
            "brier_dosecompass": round(float(np.mean((pa - ra) ** 2)), 4),
            "brier_dosecompass_div5to9": round(float(np.mean((pd_ - ra) ** 2)), 4),
            "brier_constant_base_rate": round(float(np.mean((ra.mean() - ra) ** 2)), 4),
            "mean_p_active": round(float(pa.mean()), 4), "reference_active_rate": round(float(ra.mean()), 4)}
        results[f"B={B}"] = eB
        c = eB["neurotrajectory"]["strategies"]
        print(f"B={B}: composite NT " + ", ".join(f"{s}={c[s]['composite_error_mean']}" for s in c if s != "random")
              + f", random={c['random']['expected_composite_error_mean']}", flush=True)

    prim = results[f"B={PRIMARY_B}"]["neurotrajectory"]
    same_first = sum(seqs[c]["dosecompass"][1] == seqs[c]["dosecompass_div5to9"][1] for c in chems)
    summary = {
        **header,
        "status": "complete" if not a.limit else f"smoke_test_limit_{a.limit}",
        "n_chemicals": len(chems), "n_reference_active": int(ra.sum()),
        "levels_per_chemical": {str(k): int(v) for k, v in zip(*np.unique(Ls, return_counts=True))},
        "bmr_vehicle_sd": BMR, "device": dev,
        "planner": {"samples": S_PLAN, "potency_bin_log10": 0.25, "outcome_edges_x_bmr": [0.5, 1.0, 2.0],
                    "calibration_samples": S_CAL, "calibration_grid": {"tau": TAUS, "ell": ELLS},
                    "calibration_per_fold": {str(f): v for f, v in per_fold.items()}},
        "primary_endpoint": {
            "definition": "B=3, NeuroTrajectory estimator, mean composite error, DoseCompass minus fixed log-spaced",
            "dosecompass": prim["strategies"]["dosecompass"]["composite_error_mean"],
            "fixed_logspaced": prim["strategies"]["fixed_logspaced"]["composite_error_mean"],
            "paired_bootstrap": prim["paired_bootstrap_composite"]["dosecompass_minus_fixed_logspaced"]},
        "results": results,
        "development_note": DEV_NOTE,
        "div_budget_note": ("Variant (e) decides with DIV 5-9 only; its second level equals the full-data "
                            f"DoseCompass choice for {same_first}/{len(chems)} chemicals."),
        "per_chemical": [
            {"chemical": c, "fold": int(tk[c].fold), "label": tk[c].label, "n_levels": int(len(tk[c].levels)),
             "first_level_log10uM": round(float(tk[c].levels[first_index(len(tk[c].levels))]), 4),
             "ref_active": bool(ref[c]["active"]),
             "ref_bmc_log10": None if not ref[c]["active"] else round(float(ref[c]["bmc_log10"]), 4),
             "dosecompass_order_idx": seqs[c]["dosecompass"], "dosecompass_div5to9_order_idx": seqs[c]["dosecompass_div5to9"],
             "first_step_eig_bits": pact[c]["first_step_eig_bits"],
             "composite_B3": {est: {s: round(float(per_chem_comp[(3, est)][s][i]), 4) for s in per_chem_comp[(3, est)]}
                              for est in ESTIMATORS}}
            for i, c in enumerate(chems)],
        "limitations": [
            "Retrospective: 'measuring' a level means revealing wells that were already recorded; in a real campaign each step would need a new culture (about 12 days) unless the DIV-budget variant is used to decide early.",
            "The candidate set is restricted to the concentrations EPA tested (usually 7), so wells saved are bounded by that grid and the oracle is the best achievable subset of it.",
            "Sampled full-information datasets treat features as independent given the NeuroTrajectory predictive; tau/ell calibration on the potency log score absorbs part of this misspecification but not all of it.",
            "Planner calibration uses other folds' chemicals scored by their own fold models; those models were trained with the test fold's chemicals, a global two-parameter indirect dependence (no test-chemical well is read).",
            "The DIV-budget variant feeds the NeuroTrajectory model contexts with DIV 12 masked, which it was not trained on specifically.",
            "Greedy one-step EIG; no non-myopic planning. Reference potency is itself a noisy, interpolation-based call (3 replicate wells per level)."],
        "elapsed_s": round(time.time() - t_start, 1)}
    if not a.limit:
        summary["example_chemical"] = example_panel(tasks, models, per_fold, ref, dev)
    summary["diagnostics"] = diagnostics(json.loads(json.dumps(clean(summary))))
    write_json(out_path, summary)
    print(f"wrote {out_path} ({summary['elapsed_s']}s)", flush=True)
    if not a.limit:
        make_figure(json.loads(out_path.read_text(encoding="utf-8")), Path(a.fig))


def diagnostics(s: dict) -> dict:
    """Failure analysis derived only from the per-chemical rows of the results (no new computation on data):
    where DoseCompass puts its levels, and the B = 3 composite error split by the reference activity call."""
    rows = s["per_chemical"]
    r = lambda v: round(float(v), 4)
    off = [row["dosecompass_order_idx"][1] - row["dosecompass_order_idx"][0] for row in rows]
    vals, cnt = np.unique(off, return_counts=True)
    ends = {}
    for B in BUDGETS:
        designs_B = {"dosecompass": [row["dosecompass_order_idx"][:B] for row in rows],
                     "dosecompass_div5to9": [row["dosecompass_div5to9_order_idx"][:B] for row in rows],
                     "fixed_logspaced": [fixed_design(row["n_levels"], B) for row in rows]}
        ends[f"B={B}"] = {name: {
            "frac_including_highest_level": r(np.mean([row["n_levels"] - 1 in d for row, d in zip(rows, ds)])),
            "frac_including_lowest_level": r(np.mean([0 in d for d in ds]))} for name, ds in designs_B.items()}
    ra = np.array([row["ref_active"] for row in rows], bool)
    split = {}
    for est in ESTIMATORS:
        split[est] = {}
        for strat in rows[0]["composite_B3"][est]:
            v = np.array([row["composite_B3"][est][strat] for row in rows], float)
            split[est][strat] = {"reference_active": r(v[ra].mean()), "reference_inactive": r(v[~ra].mean()),
                                 "n_reference_active": int(ra.sum()), "n_reference_inactive": int((~ra).sum())}
    return {"source": "derived from per_chemical rows of this file (run_dosecompass.py diagnostics())",
            "second_level_index_offset_from_first": {str(int(k)): int(c) for k, c in zip(vals, cnt)},
            "design_ends": ends,
            "composite_B3_by_reference_activity": split,
            "notes": [
                "The fixed log-spaced design always contains the highest tested level; DoseCompass concentrates on the "
                "predicted BMR crossing and often leaves it out, which matters most for the interpolation estimator "
                "(flat outside the measured range, so activity reached only at the top is missed).",
                "For the interpolation estimator the oracle is degenerate (it can reproduce the reference exactly once the "
                "levels bracketing the crossing are measured), so 'rel_change_pct' against the oracle is not meaningful.",
            ]}


EXAMPLE_RULE = ("first chemical, alphabetically, with EPA DNT label 'positive', reference-active and 7 tested "
                "levels; DoseCompass posterior after measuring its first (median) level")


def example_panel(tasks, models, per_fold, ref, dev):
    """Worked example for the figure; the selection rule (EXAMPLE_RULE) is fixed in advance."""
    from neurotwin.models.trajectory import interp_rows
    from neurotwin.planner.eig import eig_of_candidates
    t = sorted([x for x in tasks if x.label == "positive" and ref[x.chem]["active"] and len(x.levels) == 7],
               key=lambda x: x.chem)[0]
    m0 = first_index(len(t.levels))
    ic, _ = split_context(t, t.levels[[m0]])
    pf = per_fold[t.fold]
    mu_m, sd_m = member_predictions(models[t.fold], t, ic, t.levels, dev)
    post = posterior_from_predictions(t, ic, t.levels, mu_m, sd_m, pf["tau"], pf["ell"], S_PLAN, seed=0)
    cand = [j for j in range(len(t.levels)) if j != m0]
    eig = eig_of_candidates(post, cand)
    resp = np.abs(post.samples).max(-1)                                     # (S, L)
    truth = np.abs(dev_summary(interp_rows(t.levels, t._mu, t._ok, t.levels))).max(-1)
    order = np.lexsort((t.levels[cand], -np.abs(t.levels[cand][:, None] - t.levels[m0]).min(1), -np.round(eig, 9)))
    return {"rule": EXAMPLE_RULE, "chemical": t.chem, "fold": int(t.fold),
            "levels_log10uM": [round(float(v), 4) for v in t.levels], "first_level_idx": m0,
            "tau": pf["tau"], "ell": pf["ell"],
            "sampled_max_abs_summary_pctl": {str(q): [round(float(v), 4) for v in np.percentile(resp, q, 0)]
                                             for q in (5, 25, 50, 75, 95)},
            "observed_max_abs_summary_all_levels": [round(float(v), 4) for v in truth],
            "eig_bits": {str(j): round(float(e), 4) for j, e in zip(cand, eig)},
            "chosen_idx": int(cand[int(order[0])]),
            "posterior_p_active": round(float(post.active.mean()), 4),
            "reference_bmc_log10": round(float(ref[t.chem]["bmc_log10"]), 4)}


# ---------------------------------------------------------------- figure
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"dosecompass": "#2a78d6", "fixed_logspaced": "#eb6834", "dosecompass_div5to9": "#1baf7a",
       "random": "#8c8b86", "oracle_upper_bound": "#52514e"}
LABEL = {"dosecompass": "DoseCompass", "fixed_logspaced": "Fixed log-spaced",
         "dosecompass_div5to9": "DoseCompass, decides on DIV 5-9", "random": "Random (mean, 95% band)",
         "oracle_upper_bound": "Oracle subset (upper bound)"}


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def make_figure(summary: dict, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 8.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.5, width_ratios=[1, 1, 1.05])
    res = summary["results"]
    axes, ymax = [], 0.0
    for k, est in enumerate(ESTIMATORS):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        _style(ax)
        rnd = [res[f"B={B}"][est]["strategies"]["random"]["over_200_replicates"]["composite_error_mean"] for B in BUDGETS]
        ax.fill_between(BUDGETS, [r["p2.5"] for r in rnd], [r["p97.5"] for r in rnd], color=COL["random"], alpha=0.2, lw=0)
        series = {"random": [r["mean"] for r in rnd]}
        for s in ("oracle_upper_bound", "fixed_logspaced", "dosecompass_div5to9", "dosecompass"):
            series[s] = [res[f"B={B}"][est]["strategies"][s]["composite_error_mean"] for B in BUDGETS]
        for s, v in series.items():
            ax.plot(BUDGETS, v, "--" if s == "oracle_upper_bound" else "-", color=COL[s], lw=2, marker="o", ms=5,
                    mec=SURF, mew=1, label=LABEL[s], zorder=3 if s == "dosecompass" else 2)
            ymax = max(ymax, max(v), max(r["p97.5"] for r in rnd))
        ax.set_xticks(BUDGETS)
        ax.set_xlabel("budget B (measured concentration levels)", color=INK2, fontsize=9)
        ax.set_title(["A  Potency from the NeuroTrajectory forecast", "B  Potency from log-linear interpolation"][k],
                     loc="left", fontsize=10, color=INK)
        ax.set_ylabel("mean composite error vs full information\n(|dBMC| log10; activity miss = 1)", color=INK2, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], frameon=False, fontsize=8.5, loc="lower center", ncol=5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955))
    ax = fig.add_subplot(gs[0, 2])
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    rows = []
    for est, short in zip(ESTIMATORS, ("forecast", "interp")):
        bo = res[f"B={PRIMARY_B}"][est]["paired_bootstrap_composite"]
        for alt, name in (("fixed_logspaced", "fixed"), ("random", "random"), ("oracle_upper_bound", "oracle")):
            b = bo[f"dosecompass_minus_{alt}"]
            rows.append((f"{short} - {name}", b["mean_diff"], b["ci95"], alt))
    for i, (lab, m, ci, alt) in enumerate(rows[::-1]):
        ax.plot(ci, [i, i], color=COL[alt], lw=2, solid_capstyle="round")
        ax.plot([m], [i], "o", color=COL[alt], ms=7, mec=SURF, mew=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8, color=INK)
    ax.set_xlabel("DoseCompass minus alternative\n(composite error; mean and paired\nbootstrap 95% CI; < 0 favours DoseCompass)",
                  color=INK2, fontsize=8.5)
    ax.set_title(f"C  Paired differences at B = {PRIMARY_B}", loc="left", fontsize=10, color=INK)
    ex = summary.get("example_chemical")
    if ex:
        lv = np.array(ex["levels_log10uM"])
        pc = ex["sampled_max_abs_summary_pctl"]
        ax = fig.add_subplot(gs[1, :2])
        _style(ax)
        ax.fill_between(lv, pc["5"], pc["95"], color=COL["dosecompass"], alpha=0.15, lw=0, label="predictive 5-95%")
        ax.fill_between(lv, pc["25"], pc["75"], color=COL["dosecompass"], alpha=0.3, lw=0, label="predictive 25-75%")
        ax.plot(lv, pc["50"], color=COL["dosecompass"], lw=2, label="predictive median")
        ax.axhline(BMR, color=INK2, lw=1, ls=":")
        ax.text(lv[0], BMR, " BMR = 3 vehicle SD", va="bottom", fontsize=8, color=INK2)
        truth = np.array(ex["observed_max_abs_summary_all_levels"])
        m0, ch = ex["first_level_idx"], ex["chosen_idx"]
        ax.plot(lv, truth, "o", mfc="none", mec=INK, ms=7, label="observed level means (hidden from planner)")
        ax.plot([lv[m0]], [truth[m0]], "o", color=INK, ms=8, label="measured first level")
        ax.annotate("chosen next", (lv[ch], truth[ch]), xytext=(0, 22), textcoords="offset points", ha="center",
                    fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
        ax.set_xlabel("log10 concentration (uM)", color=INK2, fontsize=9)
        ax.set_ylabel("max over features of |developmental effect|\n(vehicle robust SD)", color=INK2, fontsize=9)
        ax.set_title(f"D  Worked example ({ex['chemical']}): posterior after the median level, "
                     f"P(active) = {ex['posterior_p_active']:.2f}", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
        ax2 = fig.add_subplot(gs[1, 2])
        _style(ax2)
        idx = sorted(int(k) for k in ex["eig_bits"])
        vals = [ex["eig_bits"][str(k)] for k in idx]
        ax2.bar(lv[idx], vals, width=0.3, color=[COL["dosecompass"] if k == ch else "#a9c8f0" for k in idx], lw=0)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel("candidate log10 concentration (uM)", color=INK2, fontsize=9)
        ax2.set_ylabel("expected information about potency (bits)", color=INK2, fontsize=9)
        ax2.set_title("E  EIG of each unmeasured level", loc="left", fontsize=10, color=INK)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
