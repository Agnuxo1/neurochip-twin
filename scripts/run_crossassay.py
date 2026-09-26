"""G5 cross-assay corroboration of the EPA NFA developmental reference (X2 Harrill imaging, X3 Kosnik acute MEA).

Both analyses are UNPAIRED cross-assay comparisons: different plates, cultures, exposure windows
and readouts. Nothing is fitted on the external data: the NFA call/potency is the a-priori rule of
neurotwin.models.potency (BMR = 3 vehicle robust SD) applied to the full-information NFA reference
(interp_rows of the observed level means on a 120-point grid spanning the tested range), so every
chemical is an out-of-sample test. The secondary "twin" predictor is the NeuroTrajectory ensemble of
the chemical's own CV fold (models that never saw it) fed only k = 3 measured concentrations (the 5
seeded designs of run_trajectory_cv.py); its call is the majority over designs.

X2 (Harrill 2018 high-content imaging, doi:10.23719/1407642): curated with neurotwin.data.harrill;
per chemical, the most sensitive neurite/synapse endpoint potency, the neuron-count potency and the
same-well 'morphological alteration without neuron-count loss' window. Concordance with NFA:
activity agreement (Cohen's kappa), Spearman of potencies (active in both) and effect direction,
with chemical-level bootstrap CIs. Primary overlap = identity-level name matches (exact or synonym);
salt/hydrate/stereo-form matches are added in a sensitivity set.

X3 (Kosnik 2020 acute MEA, doi:10.23719/1504294): does NFA developmental activity/potency predict
acute neuroactivity? Targets: (a) the published acute MEA chemical call (measured), (b) the curated
41/32 truth set (literature NEUROACTIVITY, not a molecular target, not DNT). Balanced accuracy of the
binary call and AUROC of the potency score, stratified chemical-level bootstrap CIs and label-
permutation p-values against chance (0.5).

Usage: python repo/scripts/run_crossassay.py [--nboot 4000] [--nperm 10000] [--no-twin]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.data import harrill as H  # noqa: E402
from neurotwin.data import kosnik as K  # noqa: E402
from neurotwin.models.potency import BMR, chemical_potency, dev_summary  # noqa: E402
from neurotwin.models.trajectory import interp_rows, split_context  # noqa: E402

SEED = 20260926
CENSOR_LOG10 = 4.0            # inactive chemicals ranked after every active one (10 mM)
INTERVAL_FEATURES = {"per_burst_interspike_interval", "interburst_interval_mean",
                     "inter_network_spike_interval_mean"}
MIN_CLASS = 10                # below this many chemicals in a class, BA/AUROC go to the appendix
RES = ROOT / "repo/results"
CW_CACHE = ROOT / "data/processed/crossassay/cas_dtxsid_crosswalk.parquet"
UNPAIRED = ("UNPAIRED cross-assay comparison: different plates, cultures, exposure windows and "
            "readouts; no well or plate is shared with the EPA NFA data.")


# ------------------------------------------------------------------ statistics
def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_acc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    if y.all() or (~y).all():
        return float("nan")
    return float(0.5 * (p[y].mean() + (~p[~y]).mean()))


def boot_ci(fn, arrays, n, seed=SEED, strata=None):
    """Percentile 95% CI of fn over chemical-level resamples (stratified by class if strata given)."""
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    groups = [np.where(strata == g)[0] for g in np.unique(strata)] if strata is not None else [np.arange(m)]
    vals = []
    for _ in range(n):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n // 2:
        return [None, None]
    return [r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]


def perm_p(fn, y, s, n, seed=SEED):
    """One-sided label-permutation p-value for fn(y, s) > chance."""
    rng = np.random.default_rng(seed + 1)
    obs = fn(y, s)
    cnt = sum(fn(rng.permutation(y), s) >= obs for _ in range(n))
    return round(float((1 + cnt) / (1 + n)), 5)


def agreement_block(a, b, n, name_a, name_b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"n": int(len(a)), f"{name_a}_active": int(a.sum()), f"{name_b}_active": int(b.sum()),
            "table": {"both_active": int((a & b).sum()), f"{name_a}_only": int((a & ~b).sum()),
                      f"{name_b}_only": int((~a & b).sum()), "both_inactive": int((~a & ~b).sum())},
            "agreement": r4((a == b).mean()), "agreement_ci95": boot_ci(lambda x, y: (x == y).mean(), [a, b], n),
            "kappa": r4(kappa(a, b)), "kappa_ci95": boot_ci(kappa, [a, b], n),
            f"p_{name_a}_active_given_{name_b}_active": r4(a[b].mean()) if b.any() else None,
            f"p_{name_a}_active_given_{name_b}_active_ci95": boot_ci(np.mean, [a[b]], n) if b.sum() >= 3 else None,
            f"p_{name_b}_active_given_{name_a}_active": r4(b[a].mean()) if a.any() else None,
            f"p_{name_b}_active_given_{name_a}_active_ci95": boot_ci(np.mean, [b[a]], n) if a.sum() >= 3 else None}


def spearman_block(x, y, n):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y)
    return {"n": int(len(x)), "spearman": r4(rho), "ci95": boot_ci(spearman, [x, y], n),
            "perm_p_one_sided": perm_p(spearman, x, y, min(n, 4000)) if len(x) >= 5 else None}


def classify_block(y, call, score, n, nperm):
    y = np.asarray(y, bool)
    call = np.asarray(call, bool)
    score = np.asarray(score, float)
    npos, nneg = int(y.sum()), int((~y).sum())
    out = {"n": int(len(y)), "n_positive": npos, "n_negative": nneg,
           "sensitivity": r4(call[y].mean()) if npos else None,
           "sensitivity_ci95": boot_ci(lambda c: c.mean(), [call[y]], n) if npos else None,
           "specificity": r4((~call[~y]).mean()) if nneg else None,
           "specificity_ci95": boot_ci(lambda c: (~c).mean(), [call[~y]], n) if nneg else None}
    if min(npos, nneg) < 2:
        out["status"] = "appendix_only"
        out["reason"] = f"only {min(npos, nneg)} chemical(s) in the minority class; BA/AUROC undefined"
        return out
    out.update({"balanced_accuracy": r4(bal_acc(y, call)),
                "balanced_accuracy_ci95": boot_ci(bal_acc, [y, call], n, strata=y),
                "balanced_accuracy_perm_p": perm_p(bal_acc, y, call, nperm),
                "auroc": r4(auroc(y, score)),
                "auroc_ci95": boot_ci(auroc, [y, score], n, strata=y),
                "auroc_perm_p": perm_p(auroc, y, score, nperm),
                "chance": 0.5})
    if min(npos, nneg) < MIN_CLASS:
        out["status"] = "appendix_only"
        out["reason"] = (f"minority class has {min(npos, nneg)} < {MIN_CLASS} chemicals: CIs are too wide "
                         "for a headline claim")
    else:
        out["status"] = "ok"
    return out


# ------------------------------------------------------------------ NFA side
def nfa_reference(tasks):
    rows = []
    for t in tasks:
        grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
        ref = chemical_potency(grid, dev_summary(interp_rows(t.levels, t._mu, t._ok, grid)))
        d = int(ref.get("direction", 0))
        hypo = None
        if ref["active"]:
            hypo = bool((-d if ref["feature"] in INTERVAL_FEATURES else d) < 0)
        rows.append({"chem": t.chem, "fold": int(t.fold), "dnt_label": t.label, "nfa_active": bool(ref["active"]),
                     "nfa_bmc_log10": ref["bmc_log10"], "nfa_feature": ref["feature"],
                     "nfa_activity_loss": hypo, "nfa_top_logc": float(t.levels.max()),
                     "nfa_bottom_logc": float(t.levels.min()), "nfa_n_levels": int(len(t.levels))})
    return pd.DataFrame(rows)


def twin_calls(tasks, chems, k=3, device="cpu"):
    """NeuroTrajectory few-shot call per chemical, using ONLY the models of the chemical's fold."""
    import torch
    from neurotwin.models.cnp import load_fold_models, predict
    from run_trajectory_cv import designs
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "4")))
    rows = []
    sel = [t for t in tasks if t.chem in chems and len(t.levels) > k]
    for f in sorted({t.fold for t in sel}):
        models, _ = load_fold_models(f, device)
        if not models:
            raise FileNotFoundError(f"no NeuroTrajectory models for fold {f}")
        for t in [x for x in sel if x.fold == f]:
            grid = np.linspace(t.levels.min(), t.levels.max(), 120).astype(np.float32)
            acts, bmcs = [], []
            for ctx in designs(t, k):
                ic, _ = split_context(t, ctx)
                mu = np.mean([predict(m, t, ic, grid, device=device)[0] for m in models], 0)
                est = chemical_potency(grid, dev_summary(mu))
                acts.append(est["active"])
                bmcs.append(est["bmc_log10"] if est["active"] else CENSOR_LOG10)
            rows.append({"chem": t.chem, "twin_k": k, "twin_n_designs": len(acts),
                         "twin_frac_active": float(np.mean(acts)), "twin_active": bool(np.mean(acts) > 0.5),
                         "twin_bmc_log10_median": float(np.median(bmcs))})
    return pd.DataFrame(rows)


def score_from(bmc, active):
    """Potency score for AUROC: -BMC, inactive chemicals below every active one."""
    b = np.where(np.asarray(active, bool), np.asarray(bmc, float), CENSOR_LOG10)
    return -np.nan_to_num(b, nan=CENSOR_LOG10)


# ------------------------------------------------------------------ X2
def x2_concordance(df, n):
    """df: one row per overlapping chemical with NFA and Harrill columns."""
    out = {"n_chemicals": int(len(df))}
    nfa = df.nfa_active.to_numpy(bool)
    at_bottom = df.nfa_active & np.isclose(df.nfa_bmc_log10, df.nfa_bottom_logc, atol=1e-3)
    out["nfa_active_at_lowest_tested_conc"] = {
        "n": int(at_bottom.sum()), "of_nfa_active": int(nfa.sum()),
        "harrill_morph_active_among_them": int((at_bottom & df.morph_active).sum()),
        "note": "NFA reference already at BMR at its lowest tested concentration (no lower bound on the BMC); "
                "descriptive, helps read the 'nfa_only' cell"}
    out["activity_nfa_vs_harrill_morph"] = agreement_block(nfa, df.morph_active, n, "nfa", "harrill_morph")
    out["activity_nfa_vs_harrill_selective_morph"] = agreement_block(nfa, df.selective_morph, n, "nfa",
                                                                     "harrill_selective_morph")
    out["activity_nfa_vs_harrill_any_endpoint"] = agreement_block(nfa, df.any_active, n, "nfa", "harrill_any")
    both = df[df.nfa_active & df.morph_active]
    out["potency_spearman_both_active"] = spearman_block(both.nfa_bmc_log10, both.morph_bmc_log10, n)
    d = (both.nfa_bmc_log10 - both.morph_bmc_log10).to_numpy()
    out["potency_offset_nfa_minus_morph_log10"] = {
        "median": r4(np.median(d)) if len(d) else None,
        "ci95": boot_ci(np.median, [d], n) if len(d) >= 3 else None,
        "frac_nfa_more_sensitive": r4((d < 0).mean()) if len(d) else None,
        "note": "negative = network function (NFA) responds at lower concentration than morphology"}
    cx = np.where(df.nfa_active, df.nfa_bmc_log10, CENSOR_LOG10)
    cy = np.where(df.morph_active, df.morph_bmc_log10, CENSOR_LOG10)
    out["potency_spearman_censored_all"] = spearman_block(cx, cy, n)
    out["potency_spearman_censored_all"]["censoring"] = f"inactive = {CENSOR_LOG10} log10 uM (ties ranked last)"
    anyb = df[df.nfa_active & df.any_active]
    out["potency_spearman_nfa_vs_harrill_any_both_active"] = spearman_block(anyb.nfa_bmc_log10,
                                                                          anyb.any_bmc_log10, n)
    # direction (bidirectional Harrill analysis vs NFA activity-loss direction)
    dd = df[df.nfa_active & df.morph_active_bidir]
    if len(dd) >= 3:
        hl = (dd.morph_direction_bidir < 0).to_numpy()
        nl = dd.nfa_activity_loss.astype(bool).to_numpy()
        exp = hl.mean() * nl.mean() + (1 - hl.mean()) * (1 - nl.mean())
        out["direction_concordance"] = {
            "definition": ("among chemicals active in both (bidirectional rule on both sides): Harrill most "
                           "sensitive neurite/synapse effect is a LOSS and the NFA most sensitive feature "
                           "indicates LESS network activity (decrease, or increase of an interval feature), or "
                           "both the opposite"),
            "n": int(len(dd)), "harrill_loss": int(hl.sum()), "nfa_activity_loss": int(nl.sum()),
            "concordant": int((hl == nl).sum()), "rate": r4((hl == nl).mean()),
            "ci95": boot_ci(lambda a, b: (a == b).mean(), [hl, nl], n),
            "expected_under_independence": r4(exp)}
    # common concentration range: both calls truncated at the shared top concentration
    top = np.minimum(df.nfa_top_logc, df.top_logc)
    a = df.nfa_active & (df.nfa_bmc_log10 <= top)
    b = df.morph_active & (df.morph_bmc_log10 <= top)
    out["activity_common_range_sensitivity"] = agreement_block(a, b, n, "nfa", "harrill_morph")
    # what the window means for NFA: NFA effect below the Harrill neuron-count potency
    act = df[df.nfa_active]
    cnt = np.where(act.count_active, act.count_bmc_log10, np.inf)
    out["nfa_active_below_harrill_neuron_count_loss"] = {
        "n_nfa_active": int(len(act)), "frac": r4((act.nfa_bmc_log10 < cnt).mean()) if len(act) else None,
        "note": "fraction of NFA-active chemicals whose NFA BMC is below the Harrill neuron-count BMC "
                "(count inactive = infinitely high); cross-assay, descriptive only"}
    sel_m = df[df.morph_active]
    out["window_vs_nfa"] = {
        "n_harrill_morph_active": int(len(sel_m)),
        "nfa_active_rate_if_selective": r4(sel_m[sel_m.selective_morph].nfa_active.mean())
        if sel_m.selective_morph.any() else None,
        "n_selective": int(sel_m.selective_morph.sum()),
        "nfa_active_rate_if_not_selective": r4(sel_m[~sel_m.selective_morph].nfa_active.mean())
        if (~sel_m.selective_morph).any() else None,
        "n_not_selective": int((~sel_m.selective_morph).sum())}
    return out


def x2_family_spearman(df, n):
    out = {}
    for fam in H.FAMILIES:
        c = f"{fam}_morph_bmc"
        if c not in df:
            continue
        s = df[df.nfa_active & df[c].notna()]
        out[fam] = spearman_block(s.nfa_bmc_log10, s[c], n) if len(s) >= 5 else {"n": int(len(s))}
        out[fam]["cells"] = H.FAMILIES[fam]["cells"]
    return out


def run_x2(tasks, nfa, twin, n):
    xlsx = ROOT / "data/raw/epa_dnt/Harrill_DNT_Assay_Dataset.xlsx"
    cache = ROOT / "data/processed/crossassay/harrill_raw.parquet"
    raw = H.load_raw(xlsx, cache)
    sheet = H.check_endpoint_sheet(xlsx)
    cur, noise, audit = H.curate(raw)
    lv = H.level_table(cur, "z")
    ct = H.chemical_table(lv)
    bid = H.chemical_table(lv, bidirectional=True)[["harrill_name", "morph_active", "morph_direction"]]
    bid.columns = ["harrill_name", "morph_active_bidir", "morph_direction_bidir"]
    alt_bmad = H.chemical_table(H.level_table(cur, "z_bmad"))[["harrill_name", "morph_active", "morph_bmc_log10"]]
    ct = ct.merge(bid, on="harrill_name")
    match = H.match_nfa(ct.harrill_name, nfa.chem)
    ct = ct.merge(match, on="harrill_name")
    w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).drop_duplicates()
    dtx = dict(zip(w.chemical, w.dtxsid))
    ov = ct[ct.nfa_chem.notna()].merge(nfa, left_on="nfa_chem", right_on="chem")
    ov["dtxsid"] = ov.chem.map(dtx)
    if twin is not None and len(twin):
        ov = ov.merge(twin, on="chem", how="left")
    primary = ov[ov.match_type.isin(["exact", "synonym"])]
    res = {
        "status": "ok",
        "label": UNPAIRED,
        "question": ("Do chemicals that alter neurite/synapse morphology in Harrill high-content imaging "
                     "(without loss of neuron count) agree in activity, potency ranking and direction with the "
                     "EPA NFA developmental network-formation reference call?"),
        "source": {"dataset": "Harrill et al. 2018 DNT assay dataset (US EPA ScienceHub)",
                   "doi": "10.23719/1407642", "file": "data/raw/epa_dnt/Harrill_DNT_Assay_Dataset.xlsx"},
        "curation": {**audit, "endpoint_sheet_check": sheet,
                     "normalisation": "pct = 100*(raw/median vehicle of same plate & endpoint - 1); z = pct / s_e, "
                                      "s_e = pooled robust SD of leave-one-out-normalised vehicle wells",
                     "level_summary": "median z over replicate wells per chemical x endpoint x concentration"},
        "noise_scale_per_endpoint": {int(a): {"endpoint": r.endpoint, "role": r.role, "direction": int(r.direction),
                                              "sd_vehicle_pct": r4(r.sd_vehicle_pct),
                                              "n_vehicle_wells": int(r.n_vehicle_wells),
                                              "sd_bmad_pct": r4(r.sd_bmad_pct)} for a, r in noise.iterrows()},
        "potency_rule": {"bmr_vehicle_sd": BMR, "grid_points": H.GRID_N,
                         "rule": "first crossing of BMR by the interpolated median response in the endpoint's "
                                 "declared direction (same arithmetic as neurotwin.models.potency.bmc_from_curve)",
                         "baseline_exceedance_lowest_conc": r4(H.baseline_exceedance(lv)),
                         "baseline_exceedance_note": "fraction of chemical x endpoint pairs already >= BMR at the "
                                                     "lowest tested concentration (false-positive proxy)"},
        "window_definition": ("per imaging family (same wells): W = BMC(neuron count) - min BMC(neurite/synapse); "
                              "count never reaching BMR -> W censored at the top tested concentration; a chemical "
                              "is 'selective' if in at least one family the neurite/synapse effect precedes any "
                              "neuron-count loss"),
        "harrill_summary": {"n_chemicals": int(len(ct)), "morph_active": int(ct.morph_active.sum()),
                            "count_active": int(ct.count_active.sum()), "any_active": int(ct.any_active.sum()),
                            "selective_morph": int(ct.selective_morph.sum()),
                            "selective_window_censored": int((ct.selective_morph & ct.window_censored).sum())},
        "identity_resolution": {
            "method": "Harrill ships names only: curated alias table -> EPA NFA preferred name -> NFA DTXSID",
            "counts": {k: int(v) for k, v in match.match_type.value_counts().items()},
            "unmatched": match[match.match_type == "unmatched"][["harrill_name", "unmatched_reason"]].to_dict("records"),
            "salt_or_form_pairs": match[match.match_type == "salt_or_form"][["harrill_name", "nfa_chem"]].to_dict("records"),
            "synonym_pairs": match[match.match_type == "synonym"][["harrill_name", "nfa_chem"]].to_dict("records")},
        "nfa_reference": "neurotwin.models.potency rule on interp_rows(levels, level means) over a 120-point grid "
                         "(full-information reference, run_potency.py convention)",
        "primary_overlap": {"definition": "identity-level matches (exact name or synonym)",
                            **x2_concordance(primary, n)},
        "primary_by_family_spearman": x2_family_spearman(primary, n),
        "sensitivity_exact_name_only": {"definition": "only chemicals whose Harrill and NFA names are identical up to "
                                                      "case/whitespace (no synonym or salt/form resolution)",
                                        **x2_concordance(ov[ov.match_type == "exact"], n)},
        "sensitivity_overlap_incl_salt_or_form": {"definition": "adds salt/hydrate/stereo-form parent-moiety matches",
                                                  **x2_concordance(ov, n)},
    }
    # alternative Harrill rules on the primary overlap
    alt = primary[["harrill_name", "nfa_active", "nfa_bmc_log10"]].merge(alt_bmad, on="harrill_name")
    both = alt[alt.nfa_active & alt.morph_active]
    lv_conf = lv.copy()
    conf = confirmed_table(lv_conf)
    alt2 = primary[["harrill_name", "nfa_active", "nfa_bmc_log10"]].merge(conf, on="harrill_name")
    both2 = alt2[alt2.nfa_active & alt2.morph_active]
    res["sensitivity_rules_primary_overlap"] = {
        "bmad_noise_scale": {"definition": "z scaled by tcpl-style BMAD (robust SD of test wells at each chemical's "
                                           "two lowest concentrations) instead of the vehicle SD",
                             "activity": agreement_block(alt.nfa_active, alt.morph_active, n, "nfa", "harrill_morph"),
                             "potency_spearman_both_active": spearman_block(both.nfa_bmc_log10, both.morph_bmc_log10, n)},
        "confirmed_crossing": {"definition": "an endpoint counts as active only if its median response reaches BMR at "
                                             ">= 2 consecutive tested concentrations or at the top concentration",
                               "activity": agreement_block(alt2.nfa_active, alt2.morph_active, n, "nfa", "harrill_morph"),
                               "potency_spearman_both_active": spearman_block(both2.nfa_bmc_log10,
                                                                              both2.morph_bmc_log10, n)}}
    if twin is not None and "twin_active" in primary:
        p = primary.dropna(subset=["twin_active"])
        tb = p[p.twin_active.astype(bool) & p.morph_active]
        res["twin_k3_vs_harrill_primary_overlap"] = {
            "definition": "NeuroTrajectory ensemble of the chemical's own CV fold, k=3 measured NFA concentrations, "
                          "majority call over 5 seeded designs; median BMC over designs",
            "activity": agreement_block(p.twin_active.astype(bool), p.morph_active, n, "twin", "harrill_morph"),
            "potency_spearman_both_active": spearman_block(tb.twin_bmc_log10_median, tb.morph_bmc_log10, n),
            "twin_vs_nfa_reference_activity": agreement_block(p.twin_active.astype(bool), p.nfa_active, n, "twin", "nfa")}
    lab = primary[primary.dnt_label.isin(["positive", "negative"])]
    if len(lab):
        y = (lab.dnt_label == "positive").to_numpy()
        res["context_epa_dnt_reference_labels"] = {
            "n": int(len(lab)), "n_positive": int(y.sum()), "n_negative": int((~y).sum()),
            "nfa_sensitivity": r4(lab.nfa_active[y].mean()) if y.any() else None,
            "nfa_specificity": r4((~lab.nfa_active[~y]).mean()) if (~y).any() else None,
            "harrill_morph_sensitivity": r4(lab.morph_active[y].mean()) if y.any() else None,
            "harrill_morph_specificity": r4((~lab.morph_active[~y]).mean()) if (~y).any() else None,
            "note": "descriptive; negatives are few, no CI claimed"}
    cols = ["harrill_name", "nfa_chem", "dtxsid", "match_type", "fold", "dnt_label", "nfa_active", "nfa_bmc_log10",
            "nfa_feature", "nfa_activity_loss", "morph_active", "morph_bmc_log10", "morph_endpoint", "count_active",
            "count_bmc_log10", "selective_morph", "window_log10", "window_family", "window_censored", "any_active",
            "any_bmc_log10", "morph_direction_bidir"] + (["twin_active", "twin_bmc_log10_median"] if "twin_active" in ov else [])
    per = ov[cols].sort_values("harrill_name")
    res["per_chemical"] = json.loads(per.round(4).to_json(orient="records"))
    res["limitations"] = [
        UNPAIRED,
        "Harrill cells (hN2 human iPSC-derived and rat cortical neurons, single imaging time point) differ from the "
        "NFA rat cortical networks recorded over DIV 5-12; potency offsets mix biology with assay sensitivity.",
        "Identity is resolved by name (Harrill has no identifiers); salt/form pairs are kept out of the primary set.",
        "No curve fitting: BMCs are first crossings of interpolated replicate medians, so a single noisy "
        "concentration can create a call; see baseline_exceedance and the confirmed-crossing sensitivity.",
        "The Harrill concentration range (typically 0.001-100 uM) differs from the NFA range; see the common-range "
        "sensitivity."]
    return res, ov


def confirmed_table(lv):
    """Chemical table where an endpoint is active only if BMR is reached at >= 2 consecutive tested
    concentrations or at the top concentration (robustness sensitivity)."""
    keep = []
    for (chem, aid), g in lv.groupby(["sample", "aid"]):
        g = g.sort_values("logc")
        e = (g.resp * H.ENDPOINTS[aid][2]).to_numpy() >= BMR
        ok = e[-1] or np.any(e[:-1] & e[1:])
        if not ok:
            g = g.assign(resp=0.0)
        keep.append(g)
    ct = H.chemical_table(pd.concat(keep))
    return ct[["harrill_name", "morph_active", "morph_bmc_log10"]]


# ------------------------------------------------------------------ X3
def run_x3(tasks, nfa, twin, n, nperm):
    kdir = ROOT / "data/neurotox_probe/kosnik2020"
    k = K.load(kdir, with_input=True)
    aud = K.audit(k)
    calls = K.acute_calls(k)
    cw = K.cas_crosswalk(ROOT, CW_CACHE)
    w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).drop_duplicates()
    ids = nfa[["chem"]].merge(w, left_on="chem", right_on="chemical")[["chem", "dtxsid"]]
    m = K.match_nfa(ids, calls, cw).merge(nfa, on="chem")
    tr = K.match_truth(ids, k["truth"], cw).merge(nfa, on="chem")
    if twin is not None and len(twin):
        m = m.merge(twin, on="chem", how="left")
        tr = tr.merge(twin, on="chem", how="left")
        tr = tr.merge(m[["chem", "acute_final_active"]], on="chem", how="left")
    else:
        tr = tr.merge(m[["chem", "acute_final_active"]], on="chem", how="left")
    res = {
        "status": "ok",
        "label": UNPAIRED + " Exposure is developmental (NFA, DIV 0-12) vs acute on mature networks (Kosnik).",
        "question": "Does EPA NFA developmental activity/potency predict acute neuroactivity?",
        "label_semantics": ("Kosnik truth set: 41 'Neuroactive' / 32 'Negative' chemicals curated from the literature "
                            "for acute NEUROACTIVITY (functional effects on neurons). It is not a molecular-target "
                            "label and not a developmental-neurotoxicity label. The measured target is the authors' "
                            "published acute MEA chemical call (hits in >= 3 of 15 ML-ranked parameters after their "
                            "cytotoxicity filter); that parameter set was selected with the truth set."),
        "source": {"dataset": "Kosnik et al. 2020 acute MEA (US EPA ScienceHub)", "doi": "10.23719/1504294",
                   "files": "data/neurotox_probe/kosnik2020/MEA_Data/*.csv (published tcpl outputs; R scripts not run)"},
        "parse_audit": aud,
        "identity_resolution": {"method": "Kosnik CASRN -> DTXSID (local EPA crosswalk) -> NFA DTXSID; else normalised name",
                                "crosswalk_pairs": int(len(cw)),
                                "n_nfa_matched": int(len(m)),
                                "by": {k_: int(v) for k_, v in m.match.value_counts().items()},
                                "n_nfa_truth_matched": int(len(tr))},
        "no_fitting": ("No parameter is fitted on Kosnik data: the NFA call is the a-priori BMR = 3 rule and the score is "
                       "-BMC (inactive ranked last), so every chemical is an out-of-sample (chemical-split) test; twin "
                       "predictions use only CV-fold models that never saw the chemical."),
    }
    y = m.acute_final_active.to_numpy(bool)
    res["measured_acute_call"] = {
        "target": "Kosnik published acute MEA chemical call (final >=3-parameter hit)",
        "nfa_reference": classify_block(y, m.nfa_active, score_from(m.nfa_bmc_log10, m.nfa_active), n, nperm)}
    if "twin_active" in m:
        mm = m.dropna(subset=["twin_active"])
        res["measured_acute_call"]["twin_k3"] = classify_block(
            mm.acute_final_active.to_numpy(bool), mm.twin_active.astype(bool),
            -mm.twin_bmc_log10_median.to_numpy(float), n, nperm)
    at_bottom = m.nfa_active & np.isclose(m.nfa_bmc_log10, m.nfa_bottom_logc, atol=1e-3)
    res["measured_acute_call"]["nfa_active_at_lowest_tested_conc"] = {
        "n": int(at_bottom.sum()), "of_nfa_active": int(m.nfa_active.sum()),
        "acute_active_among_them": int((at_bottom & m.acute_final_active).sum()),
        "note": "NFA reference already at BMR at its lowest tested concentration; descriptive"}
    both = m[m.nfa_active & m.acute_final_active]
    res["measured_acute_call"]["potency_spearman_both_active"] = spearman_block(both.nfa_bmc_log10,
                                                                              both.acute_bmc_log10, n)
    res["measured_acute_call"]["potency_spearman_both_active"]["note"] = (
        "NFA BMC (BMR crossing) vs Kosnik min AC50 (modl_ga, the authors' potency); only ranks are compared "
        "because the two potency definitions differ")
    ya = m.acute_any_hit_active.to_numpy(bool)
    res["sensitivity_any_filtered_hit"] = {
        "target": "acute MEA active if any cytotoxicity-filtered tcpl hit (liberal)",
        "nfa_reference": classify_block(ya, m.nfa_active, score_from(m.nfa_bmc_log10, m.nfa_active), n, nperm)}
    yt = tr.neuroactive.to_numpy(bool)
    ts = {"target": "Kosnik curated truth label (Neuroactive=1, Negative=0)",
          "nfa_reference": classify_block(yt, tr.nfa_active, score_from(tr.nfa_bmc_log10, tr.nfa_active), n, nperm)}
    if "twin_active" in tr:
        tt = tr.dropna(subset=["twin_active"])
        ts["twin_k3"] = classify_block(tt.neuroactive.to_numpy(bool), tt.twin_active.astype(bool),
                                       -tt.twin_bmc_log10_median.to_numpy(float), n, nperm)
    ta = tr.dropna(subset=["acute_final_active"])
    ts["context_acute_mea_call_same_chemicals"] = classify_block(
        ta.neuroactive.to_numpy(bool), ta.acute_final_active.astype(bool),
        ta.acute_final_active.astype(float).to_numpy(), n, nperm)
    ts["status"] = "appendix_only" if min(yt.sum(), (~yt).sum()) < MIN_CLASS else "ok"
    if ts["status"] == "appendix_only":
        ts["reason"] = (f"the NFA cohort overlaps the truth set in {int(yt.sum())} neuroactive but only "
                        f"{int((~yt).sum())} negative chemicals; specificity, BA and AUROC are not estimable "
                        "with useful precision")
    res["truth_set"] = ts
    cols = ["chem", "dtxsid", "match", "kosnik_name", "kosnik_cas", "fold", "nfa_active", "nfa_bmc_log10",
            "acute_final_active", "acute_bmc_log10", "acute_n_final_endpoints", "acute_any_hit_active"] + \
           (["twin_active", "twin_bmc_log10_median"] if "twin_active" in m else [])
    res["per_chemical"] = json.loads(m[cols].sort_values("chem").round(4).to_json(orient="records"))
    res["per_chemical_truth"] = json.loads(tr[["chem", "truth_name", "truth_label", "truth_match", "nfa_active",
                                               "nfa_bmc_log10", "acute_final_active"]].sort_values("chem")
                                           .round(4).to_json(orient="records"))
    res["limitations"] = [
        res["label"],
        "The NFA cohort was assembled around DNT reference chemicals, so it overlaps the Kosnik truth set mostly in "
        "neuroactive chemicals; the truth-set analysis is therefore appendix-only.",
        "The acute MEA call used as measured target was defined with ML-ranked parameters chosen on the truth set.",
        "Acute neuroactivity and developmental network disruption are different biological questions; agreement is "
        "corroboration of shared neuroactive chemistry, not validation of a DNT prediction."]
    return res, m, tr


# ------------------------------------------------------------------ figure
def make_figure(x2ov, x3m, x2, x3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid_c = "#1f2328", "#6e7781", "#d0d7de"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#199e70"   # reference categorical slots 1-3 (validated)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4))
    prim = x2ov[x2ov.match_type.isin(["exact", "synonym"])]
    # (a) X2 potency scatter
    a = ax[0, 0]
    b = prim[prim.nfa_active & prim.morph_active]
    for sel, col, lab in [(b.selective_morph, c1, "morphology before neuron-count loss"),
                          (~b.selective_morph, c2, "neuron-count loss first")]:
        a.scatter(b.nfa_bmc_log10[sel], b.morph_bmc_log10[sel], s=28, color=col, label=lab, alpha=.9, lw=0)
    def lims(*cols):
        v = np.concatenate([np.asarray(c, float) for c in cols])
        v = v[np.isfinite(v)]
        return [float(np.floor(v.min() * 2) / 2 - 0.25), float(np.ceil(v.max() * 2) / 2 + 0.25)]
    lim = lims(b.nfa_bmc_log10, b.morph_bmc_log10)
    a.plot(lim, lim, color=grid_c, lw=1, zorder=0)
    a.set_xlim(lim), a.set_ylim(lim)
    s = x2["primary_overlap"]["potency_spearman_both_active"]
    a.set_title(f"X2  NFA vs Harrill imaging potency (both active, n={s['n']})\nSpearman {s['spearman']} "
                f"[{s['ci95'][0]}, {s['ci95'][1]}]", fontsize=9, color=ink, loc="left")
    a.set_xlabel("NFA developmental BMC (log10 µM)"), a.set_ylabel("Harrill neurite/synapse BMC (log10 µM)")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    # (b) X2 activity table
    a = ax[0, 1]
    t = x2["primary_overlap"]["activity_nfa_vs_harrill_morph"]["table"]
    M = np.array([[t["both_active"], t["nfa_only"]], [t["harrill_morph_only"], t["both_inactive"]]])
    a.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.3)
    for i in range(2):
        for j in range(2):
            a.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=14, color=ink)
    a.set_xticks([0, 1], ["Harrill morph active", "Harrill morph inactive"])
    a.set_yticks([0, 1], ["NFA active", "NFA inactive"])
    a.spines[:].set_visible(False)
    ag = x2["primary_overlap"]["activity_nfa_vs_harrill_morph"]
    a.set_title(f"X2  activity agreement (n={ag['n']}): {ag['agreement']} "
                f"[{ag['agreement_ci95'][0]}, {ag['agreement_ci95'][1]}]\nCohen's kappa {ag['kappa']} "
                f"[{ag['kappa_ci95'][0]}, {ag['kappa_ci95'][1]}]", fontsize=9, color=ink, loc="left")
    # (c) X3 ROC
    a = ax[1, 0]

    def roc(y, s):
        o = np.argsort(-s, kind="mergesort")
        y = np.asarray(y, bool)[o]
        tp = np.concatenate([[0], np.cumsum(y)]) / max(y.sum(), 1)
        fp = np.concatenate([[0], np.cumsum(~y)]) / max((~y).sum(), 1)
        return fp, tp
    y = x3m.acute_final_active.to_numpy(bool)
    fp, tp = roc(y, score_from(x3m.nfa_bmc_log10, x3m.nfa_active))
    r = x3["measured_acute_call"]["nfa_reference"]
    a.step(fp, tp, where="post", color=c1, lw=1.8, label=f"NFA reference potency  AUROC {r.get('auroc')}")
    if "twin_k3" in x3["measured_acute_call"]:
        mm = x3m.dropna(subset=["twin_active"])
        fp, tp = roc(mm.acute_final_active.to_numpy(bool), -mm.twin_bmc_log10_median.to_numpy(float))
        rt = x3["measured_acute_call"]["twin_k3"]
        a.step(fp, tp, where="post", color=c3, lw=1.5, label=f"twin, 3 concentrations  AUROC {rt.get('auroc')}")
    a.plot([0, 1], [0, 1], color=grid_c, lw=1, ls="--", zorder=0)
    a.set_xlabel("1 - specificity"), a.set_ylabel("sensitivity")
    a.set_title(f"X3  NFA developmental potency -> acute MEA call (n={r['n']}, {r['n_positive']} active)\n"
                f"BA {r.get('balanced_accuracy')} [{r['balanced_accuracy_ci95'][0]}, {r['balanced_accuracy_ci95'][1]}]"
                f"; AUROC CI [{r['auroc_ci95'][0]}, {r['auroc_ci95'][1]}]", fontsize=9, color=ink, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    # (d) X3 potency scatter
    a = ax[1, 1]
    b = x3m[x3m.nfa_active & x3m.acute_final_active]
    a.scatter(b.nfa_bmc_log10, b.acute_bmc_log10, s=26, color=c1, lw=0, alpha=.9)
    lim = lims(b.nfa_bmc_log10, b.acute_bmc_log10)
    a.plot(lim, lim, color=grid_c, lw=1, zorder=0)
    a.set_xlim(lim), a.set_ylim(lim)
    s = x3["measured_acute_call"]["potency_spearman_both_active"]
    a.set_title(f"X3  developmental vs acute potency (both active, n={s['n']})\nSpearman {s['spearman']} "
                f"[{s['ci95'][0]}, {s['ci95'][1]}]", fontsize=9, color=ink, loc="left")
    a.set_xlabel("NFA developmental BMC (log10 µM)"), a.set_ylabel("Kosnik acute MEA min AC50 (log10 µM)")
    fig.suptitle("Cross-assay corroboration of the NFA reference (unpaired: different plates, cultures and readouts)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--device", default=os.environ.get("NT_DEVICE", "cpu"))
    args = ap.parse_args()
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    nfa = nfa_reference(tasks)
    twin = None
    if not args.no_twin:
        # chemicals needed by either analysis (computed once, fold-restricted)
        hn = H.match_nfa(H.curate(H.load_raw(ROOT / "data/raw/epa_dnt/Harrill_DNT_Assay_Dataset.xlsx",
                                             ROOT / "data/processed/crossassay/harrill_raw.parquet"))[0]["sample"].unique(),
                         nfa.chem)
        k = K.load(ROOT / "data/neurotox_probe/kosnik2020")
        cw = K.cas_crosswalk(ROOT, CW_CACHE)
        w = pd.read_parquet(ROOT / "data/processed/epa_nfa/wells.parquet", columns=["chemical", "dtxsid"]).drop_duplicates()
        ids = nfa[["chem"]].merge(w, left_on="chem", right_on="chemical")[["chem", "dtxsid"]]
        need = set(hn.nfa_chem.dropna()) | set(K.match_nfa(ids, K.acute_calls(k), cw).chem) | \
            set(K.match_truth(ids, k["truth"], cw).chem)
        twin = twin_calls(tasks, need, 3, args.device)
        print(f"twin predictions for {len(twin)} chemicals", flush=True)
    x2, x2ov = run_x2(tasks, nfa, twin, args.nboot)
    x2["settings"] = {"seed": SEED, "n_bootstrap": args.nboot, "bootstrap": "chemical-level percentile 95% CI"}
    po = x2["primary_overlap"]
    x2["headline"] = {"n_overlap_primary": po["n_chemicals"],
                      "activity_agreement": po["activity_nfa_vs_harrill_morph"]["agreement"],
                      "activity_agreement_ci95": po["activity_nfa_vs_harrill_morph"]["agreement_ci95"],
                      "kappa": po["activity_nfa_vs_harrill_morph"]["kappa"],
                      "kappa_ci95": po["activity_nfa_vs_harrill_morph"]["kappa_ci95"],
                      "p_nfa_active_given_harrill_morph_active":
                          po["activity_nfa_vs_harrill_morph"]["p_nfa_active_given_harrill_morph_active"],
                      "spearman_potency_both_active": po["potency_spearman_both_active"]["spearman"],
                      "spearman_ci95": po["potency_spearman_both_active"]["ci95"],
                      "n_both_active": po["potency_spearman_both_active"]["n"],
                      "potency_offset_nfa_minus_morph_log10_median": po["potency_offset_nfa_minus_morph_log10"]["median"],
                      "potency_offset_ci95": po["potency_offset_nfa_minus_morph_log10"]["ci95"],
                      "direction_concordance_rate": po.get("direction_concordance", {}).get("rate"),
                      "direction_expected_under_independence":
                          po.get("direction_concordance", {}).get("expected_under_independence")}
    (RES / "x2_harrill.json").write_text(json.dumps(x2, indent=2), encoding="utf-8")
    print("wrote x2_harrill.json", json.dumps(x2["primary_overlap"]["activity_nfa_vs_harrill_morph"]),
          json.dumps(x2["primary_overlap"]["potency_spearman_both_active"]), flush=True)
    x3, x3m, _ = run_x3(tasks, nfa, twin, args.nboot, args.nperm)
    x3["settings"] = {"seed": SEED, "n_bootstrap": args.nboot, "n_permutations": args.nperm,
                      "bootstrap": "chemical-level percentile 95% CI, stratified by class for BA/AUROC"}
    r = x3["measured_acute_call"]["nfa_reference"]
    x3["headline"] = {"n_overlap_measured": r["n"], "n_acute_active": r["n_positive"],
                      "balanced_accuracy": r.get("balanced_accuracy"), "balanced_accuracy_ci95": r.get("balanced_accuracy_ci95"),
                      "balanced_accuracy_perm_p": r.get("balanced_accuracy_perm_p"),
                      "auroc": r.get("auroc"), "auroc_ci95": r.get("auroc_ci95"), "auroc_perm_p": r.get("auroc_perm_p"),
                      "status": r.get("status"),
                      "n_overlap_truth_set": x3["truth_set"]["nfa_reference"]["n"],
                      "truth_set_status": x3["truth_set"]["status"]}
    (RES / "x3_kosnik.json").write_text(json.dumps(x3, indent=2), encoding="utf-8")
    print("wrote x3_kosnik.json", json.dumps(x3["measured_acute_call"]["nfa_reference"]), flush=True)
    make_figure(x2ov, x3m, x2, x3, RES / "figures/crossassay.png")
    print("wrote figures/crossassay.png")


if __name__ == "__main__":
    main()
