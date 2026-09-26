"""EPA Harrill et al. (2018) developmental-neurotoxicity high-content imaging battery (X2).

Source: "Data for Harrill et al.: Testing for developmental neurotoxicity using a suite of assays
for key cellular events in neurodevelopment", US EPA ScienceHub, doi:10.23719/1407642
(public domain unless noted). One XLSX, sheet ``Data``: 48,856 assay rows, 21 endpoints,
14 nominal concentrations 0.001-3000 uM, ~3 replicate wells per concentration and 3+ DMSO
vehicle wells per plate and endpoint. No images, no chemical identifiers (names only).

Curation SOP (audited earlier by the team, reproduced here and checked by tests):
1. Exclude every row with ``well quality == 0`` (107 rows; 11 of them are test-chemical rows on
   5 physical wells, the others are blanks).
2. Resolve sample aliases: whitespace ("DMSO "), Valproate == Valproic Acid, and the positive
   control spellings BIs1/Bis1, Rac I/Rac In.
3. Normalise each well to the median of the vehicle (DMSO) wells of the SAME plate and endpoint:
       pct = 100 * (raw / median_vehicle(plate, endpoint) - 1)
   and express it in vehicle robust-SD units z = pct / s_e, where s_e is the pooled robust SD
   (1.4826 * MAD) of vehicle wells, each normalised leave-one-out to the median of the other
   vehicle wells of its plate (so that s_e is not shrunk by self-normalisation).
4. Concentration-response per chemical and endpoint: median z over replicate wells at each tested
   concentration (log10 uM), linearly interpolated on a 120-point grid spanning the tested range.
5. Potency: the same benchmark-response rule as neurotwin.models.potency (BMR = 3 vehicle robust
   SD; BMC = lowest log10 concentration at which the effect first reaches BMR, linear
   interpolation), applied in the endpoint's declared direction (loss for all imaging endpoints,
   gain for hNP1 caspase apoptosis). A bidirectional variant (|z|, exactly the NFA rule) is also
   provided.

"Morphological alteration without loss of neuron count": inside each imaging assay family the
neurite/synapse endpoints and the neuron count are measured in the SAME wells. The family window
is  W = BMC(neuron count) - min BMC(neurite/synapse endpoints)  in log10 units; when the neuron
count never reaches BMR, W is right-censored at the top tested concentration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BMR = 3.0                      # same a-priori benchmark response as neurotwin.models.potency
GRID_N = 120
RAW_COLUMNS = ["dbid", "aid", "sample", "plate", "row", "col", "wtype", "wq", "conc", "raw"]
N_ROWS_EXPECTED = 48856

# aid -> (component name, role, declared signal direction +1 gain / -1 loss, family)
ENDPOINTS = {
    1: ("hNP1_Caspase_Apop", "hnp1_apoptosis", +1, "hNP1"),
    2: ("hN2_NOG_NeuronCount", "count", -1, "hN2_NOG"),
    3: ("hN2_NOG_NeuriteLength", "morph", -1, "hN2_NOG"),
    4: ("hN2_NOG_NeuriteCount", "morph", -1, "hN2_NOG"),
    5: ("hN2_NOG_BPCount", "morph", -1, "hN2_NOG"),
    6: ("Cortical_NOG_NeuronCount", "count", -1, "Cortical_NOG"),
    7: ("Cortical_NOG_NeuriteLength", "morph", -1, "Cortical_NOG"),
    8: ("Cortical_NOG_NeuriteCount", "morph", -1, "Cortical_NOG"),
    9: ("Cortical_NOG_BPCount", "morph", -1, "Cortical_NOG"),
    11: ("hNP1_CellTiter_Lum", "hnp1_viability", -1, "hNP1"),
    12: ("Cortical_Synaptogenesis_NeuronCount", "count", -1, "Cortical_Synaptogenesis"),
    13: ("Cortical_Synaptogenesis_CellBodySpotCount", "morph", -1, "Cortical_Synaptogenesis"),
    14: ("Cortical_Synaptogenesis_NeuriteSpotCountPerNeuron", "morph", -1, "Cortical_Synaptogenesis"),
    15: ("Cortical_Synaptogenesis_NeuriteSpotCountPerNeuriteLength", "morph", -1, "Cortical_Synaptogenesis"),
    16: ("Cortical_Synaptogenesis_NeuriteLength", "morph", -1, "Cortical_Synaptogenesis"),
    17: ("Cortical_Synaptogenesis_NeuriteCount", "morph", -1, "Cortical_Synaptogenesis"),
    18: ("Cortical_Synaptogenesis_BPCount", "morph", -1, "Cortical_Synaptogenesis"),
    19: ("Cortical_Synaptogenesis_SynapseCount", "morph", -1, "Cortical_Synaptogenesis"),
    21: ("hNP1_Pro_ObjectCount", "hnp1_proliferation", -1, "hNP1"),
    22: ("hNP1_Pro_ResponderAvgInten", "hnp1_proliferation", -1, "hNP1"),
    23: ("hNP1_Pro_MeanAvgInten", "hnp1_proliferation", -1, "hNP1"),
}
FAMILIES = {  # imaging families where neurites/synapses and neuron count come from the same wells
    "hN2_NOG": {"count": 2, "morph": [3, 4, 5], "cells": "human iPSC-derived neurons (hN2)"},
    "Cortical_NOG": {"count": 6, "morph": [7, 8, 9], "cells": "rat primary cortical neurons"},
    "Cortical_Synaptogenesis": {"count": 12, "morph": [13, 14, 15, 16, 17, 18, 19],
                                "cells": "rat primary cortical neurons"},
}
SAMPLE_ALIASES = {"Valproate": "Valproic Acid", "BIs1": "Bis1", "Rac I": "Rac In"}

# Harrill sample name -> (EPA NFA / DSSTox preferred name, match type). Harrill ships names only,
# so identity is resolved by this curated table and then inherits the NFA DTXSID.
#   exact        identical name up to case/whitespace
#   synonym      same substance under another name (same DSSTox record expected)
#   salt_or_form same parent moiety, different salt / hydrate / stereo form (different DTXSID)
NFA_ALIASES = {
    "Aminonicotinimide": ("6-aminopyridine-3-carboxamide", "synonym"),
    "Bis(tri-n-butyltin) oxide": ("Bis(tributyltin)oxide", "synonym"),
    "Cocaine Bases": ("Cocaine", "synonym"),
    "Cytosine Arabinoside": ("Cytarabine", "synonym"),
    "DEHP": ("Di(2-ethylhexyl) phthalate", "synonym"),
    "Diphenylhydantoin": ("5,5-Diphenylhydantoin", "synonym"),
    "Heptachlor epoxide": ("Heptachlor epoxide B", "synonym"),   # CAS 1024-57-3 in both sources
    "Nicotine": ("(-)-Nicotine", "synonym"),
    "PBDE-47": ("2,2',4,4'-Tetrabromodiphenyl ether", "synonym"),
    "Sorbitol": ("D-Glucitol", "synonym"),
    "trans-Retinoic Acid": ("Retinoic acid", "synonym"),
    "Amphetamine": ("Dextroamphetamine sulfate", "salt_or_form"),
    "Cadmium Chloride": ("Cadmium(II) chloride hydrate (2:5)", "salt_or_form"),
    "Chlordiazepoxide": ("Chlordiazepoxide hydrochloride", "salt_or_form"),
    "Chlorpromazine": ("Chlorpromazine hydrochloride", "salt_or_form"),
    "Cyclophosphamide": ("Cyclophosphamide monohydrate", "salt_or_form"),
    "Fluoxetine": ("Fluoxetine hydrochloride", "salt_or_form"),
    "Ketamine": ("Esketamine hydrochloride", "salt_or_form"),
    "Lead Acetate": ("Lead(II) acetate trihydrate", "salt_or_form"),
    "Methylmercury": ("Methylmercuric(II) chloride", "salt_or_form"),
    "Naloxone": ("Naloxone hydrochloride dihydrate", "salt_or_form"),
    "Paraquat": ("Paraquat dichloride", "salt_or_form"),
    "Saccharin": ("Sodium saccharin hydrate", "salt_or_form"),
    "Terbutaline": ("Terbutaline hemisulfate", "salt_or_form"),
    "Triethyltin": ("Triethyltin bromide", "salt_or_form"),
    "Trimethyltin": ("Trimethyltin hydroxide", "salt_or_form"),
    "Valproic Acid": ("Sodium valproate", "salt_or_form"),
}
UNMATCHED_REASONS = {
    "Heptachlor": "not tested in the EPA NFA cohort (only heptachlor epoxide B is)",
    "Manganese": "ambiguous salt: the NFA cohort has both manganese dichloride and manganese(II) acetate",
    "Thiouracil": "2-thiouracil is not in the NFA cohort (only 6-methyl- and 6-propyl-2-thiouracil)",
}


def name_key(x) -> str:
    return " ".join(str(x).casefold().split())


# ------------------------------------------------------------------ loading and curation
def load_raw(xlsx: Path, cache: Path | None = None) -> pd.DataFrame:
    """Read sheet ``Data`` with short column names (optionally cached as parquet)."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    d = pd.read_excel(xlsx, sheet_name="Data")
    if d.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"unexpected Harrill layout: {d.shape}")
    d.columns = RAW_COLUMNS
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache)
    return d


def check_endpoint_sheet(xlsx: Path) -> dict:
    """Verify ENDPOINTS against the workbook's own endpoint annotation sheet."""
    a = pd.read_excel(xlsx, sheet_name="assay_component_endpoint")
    bad = []
    for aid, name, direc in a[["Assay id", "assay__name", "signal_direction"]].itertuples(index=False):
        comp, _, sign, _ = ENDPOINTS[int(aid)]
        if not str(name).startswith(comp) or (sign > 0) != (str(direc).strip() == "gain"):
            bad.append(int(aid))
    return {"n_endpoints_sheet": int(len(a)), "mismatches": bad}


def _robust_sd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def curate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply the SOP. Returns (test-chemical wells with pct/z, per-endpoint noise table, audit)."""
    d = raw.copy()
    d["sample"] = d["sample"].astype(str).str.strip().replace(SAMPLE_ALIASES)
    q0 = d[d.wq == 0]
    q0_test = q0[q0.wtype == "t"]
    audit = {"n_rows": int(len(d)), "n_rows_quality0": int(len(q0)),
             "n_test_rows_quality0": int(len(q0_test)),
             "n_test_wells_quality0": int(q0_test[["plate", "row", "col"]].drop_duplicates().shape[0]),
             "n_endpoints": int(d.aid.nunique()), "n_plates": int(d.plate.nunique()),
             "concentrations_uM": sorted(float(c) for c in d.loc[d.wtype == "t", "conc"].unique()),
             "n_sample_names_test_raw": int(raw.loc[raw.wtype == "t", "sample"].astype(str).str.strip().nunique()),
             "n_chemicals_test_after_alias": int(d.loc[d.wtype == "t", "sample"].nunique()),
             "aliases_resolved": SAMPLE_ALIASES}
    d = d[d.wq == 1]
    veh = d[d.wtype == "n"]
    g = veh.groupby(["aid", "plate"]).raw
    base = g.median().rename("veh_median")
    nveh = g.size().rename("n_vehicle")
    # leave-one-out vehicle normalisation for the noise scale
    loo = []
    for (aid, plate), v in veh.groupby(["aid", "plate"]).raw:
        v = v.to_numpy(float)
        if len(v) < 2:
            continue
        for i in range(len(v)):
            b = np.median(np.delete(v, i))
            if b > 0:
                loo.append((aid, 100.0 * (v[i] / b - 1.0)))
    loo = pd.DataFrame(loo, columns=["aid", "pct"])
    t = d[d.wtype == "t"].join(base, on=["aid", "plate"]).join(nveh, on=["aid", "plate"])
    bad_plate = t.veh_median.isna() | (t.veh_median <= 0)
    audit["n_test_rows_dropped_no_vehicle"] = int(bad_plate.sum())
    t = t[~bad_plate].copy()
    t["pct"] = 100.0 * (t.raw / t.veh_median - 1.0)
    t["logc"] = np.log10(t.conc.astype(float))
    # alternative noise scale (tcpl-style BMAD): test wells at each chemical's two lowest concentrations
    low = t.groupby(["sample", "aid"]).conc.transform(lambda c: c.isin(np.sort(c.unique())[:2]))
    noise = pd.DataFrame({
        "aid": sorted(ENDPOINTS),
        "endpoint": [ENDPOINTS[a][0] for a in sorted(ENDPOINTS)],
        "role": [ENDPOINTS[a][1] for a in sorted(ENDPOINTS)],
        "direction": [ENDPOINTS[a][2] for a in sorted(ENDPOINTS)],
        "sd_vehicle_pct": [_robust_sd(loo.loc[loo.aid == a, "pct"]) for a in sorted(ENDPOINTS)],
        "n_vehicle_wells": [int((loo.aid == a).sum()) for a in sorted(ENDPOINTS)],
        "sd_bmad_pct": [_robust_sd(t.loc[low & (t.aid == a), "pct"]) for a in sorted(ENDPOINTS)],
    }).set_index("aid")
    t["z"] = t.pct / t.aid.map(noise.sd_vehicle_pct)
    t["z_bmad"] = t.pct / t.aid.map(noise.sd_bmad_pct)
    audit["n_test_rows_used"] = int(len(t))
    return t.reset_index(drop=True), noise, audit


def level_table(t: pd.DataFrame, col: str = "z") -> pd.DataFrame:
    """Median response over replicate wells per (chemical, endpoint, concentration)."""
    g = t.groupby(["sample", "aid", "logc"])[col]
    return pd.DataFrame({"resp": g.median(), "n_wells": g.size()}).reset_index()


# ------------------------------------------------------------------ potency rule
def first_crossing(grid: np.ndarray, e: np.ndarray, bmr: float = BMR) -> float:
    """Lowest grid log-concentration where e first reaches bmr (linear interpolation); nan if never.
    Identical arithmetic to neurotwin.models.potency.bmc_from_curve for one column."""
    hit = np.where(e >= bmr)[0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(grid[0])
    x0, x1, y0, y1 = grid[i - 1], grid[i], e[i - 1], e[i]
    return float(x0 + (bmr - y0) * (x1 - x0) / max(y1 - y0, 1e-9))


def endpoint_curve(lv: np.ndarray, resp: np.ndarray, n: int = GRID_N):
    grid = np.linspace(lv.min(), lv.max(), n)
    return grid, np.interp(grid, lv, resp)


def endpoint_potency(lv, resp, sign: int, bmr: float = BMR, bidirectional: bool = False):
    """(bmc_log10 or nan, effect direction +1/-1/0) for one chemical x endpoint."""
    lv, resp = np.asarray(lv, float), np.asarray(resp, float)
    o = np.argsort(lv)
    grid, a = endpoint_curve(lv[o], resp[o])
    e = np.abs(a) if bidirectional else sign * a
    bmc = first_crossing(grid, e, bmr)
    if not np.isfinite(bmc):
        return bmc, 0
    i = int(np.where(e >= bmr)[0][0])
    return bmc, int(np.sign(a[i]))


def chemical_table(levels: pd.DataFrame, bmr: float = BMR, bidirectional: bool = False) -> pd.DataFrame:
    """One row per Harrill chemical: endpoint BMCs, family windows and chemical-level calls."""
    rows = []
    for chem, gc in levels.groupby("sample"):
        r = {"harrill_name": chem}
        bmc, top, dirn = {}, {}, {}
        for aid, ge in gc.groupby("aid"):
            b, s = endpoint_potency(ge.logc.to_numpy(), ge.resp.to_numpy(), ENDPOINTS[aid][2], bmr, bidirectional)
            bmc[aid], top[aid], dirn[aid] = b, float(ge.logc.max()), s
            r[f"bmc_{ENDPOINTS[aid][0]}"] = b
        best_w, best_fam, any_sel = -np.inf, None, False
        for fam, spec in FAMILIES.items():
            ms = [a for a in spec["morph"] if a in bmc]
            c = spec["count"]
            if not ms or c not in bmc:
                r[f"{fam}_tested"] = False
                continue
            r[f"{fam}_tested"] = True
            mb = np.array([bmc[a] for a in ms])
            active = np.isfinite(mb)
            r[f"{fam}_morph_bmc"] = float(np.nanmin(mb)) if active.any() else np.nan
            r[f"{fam}_count_bmc"] = bmc[c]
            if active.any():
                a_best = ms[int(np.nanargmin(mb))]
                cnt = bmc[c] if np.isfinite(bmc[c]) else top[c]
                w = cnt - r[f"{fam}_morph_bmc"]
                r[f"{fam}_window_log10"] = float(w)
                r[f"{fam}_window_censored"] = bool(not np.isfinite(bmc[c]))
                sel = (not np.isfinite(bmc[c])) or (bmc[c] > r[f"{fam}_morph_bmc"])
                r[f"{fam}_selective"] = bool(sel)
                any_sel |= sel
                if w > best_w:
                    best_w, best_fam = w, fam
                    r["window_endpoint"] = ENDPOINTS[a_best][0]
                    r["window_direction"] = dirn[a_best]
        morph = [(bmc[a], a) for a in bmc if ENDPOINTS[a][1] == "morph" and np.isfinite(bmc[a])]
        count = [bmc[a] for a in bmc if ENDPOINTS[a][1] == "count" and np.isfinite(bmc[a])]
        anyb = [bmc[a] for a in bmc if np.isfinite(bmc[a])]
        r["morph_active"] = bool(morph)
        r["morph_bmc_log10"] = min(morph)[0] if morph else np.nan
        r["morph_endpoint"] = ENDPOINTS[min(morph)[1]][0] if morph else None
        r["morph_family"] = ENDPOINTS[min(morph)[1]][3] if morph else None
        r["morph_direction"] = dirn[min(morph)[1]] if morph else 0
        r["count_active"] = bool(count)
        r["count_bmc_log10"] = min(count) if count else np.nan
        r["any_active"] = bool(anyb)
        r["any_bmc_log10"] = min(anyb) if anyb else np.nan
        r["selective_morph"] = bool(any_sel)
        r["window_log10"] = float(best_w) if best_fam else np.nan
        r["window_family"] = best_fam
        r["window_censored"] = bool(r.get(f"{best_fam}_window_censored", False)) if best_fam else False
        r["top_logc"] = float(gc.logc.max())
        r["bottom_logc"] = float(gc.logc.min())
        rows.append(r)
    return pd.DataFrame(rows)


def baseline_exceedance(levels: pd.DataFrame, bmr: float = BMR) -> float:
    """QC false-positive proxy: fraction of (chemical, endpoint) whose median response at the lowest
    tested concentration already reaches BMR in the declared direction."""
    lo = levels.loc[levels.groupby(["sample", "aid"]).logc.idxmin()]
    e = lo.resp * lo.aid.map(lambda a: ENDPOINTS[a][2])
    return float((e >= bmr).mean())


# ------------------------------------------------------------------ identity resolution
def match_nfa(harrill_names, nfa_names) -> pd.DataFrame:
    """Map Harrill sample names to NFA chemical names (exact key, else curated alias)."""
    nfa_by_key = {name_key(n): n for n in nfa_names}
    rows = []
    for h in sorted(set(harrill_names)):
        if h in NFA_ALIASES and NFA_ALIASES[h][0] in set(nfa_names):
            rows.append((h, NFA_ALIASES[h][0], NFA_ALIASES[h][1], None))
        elif name_key(h) in nfa_by_key:
            rows.append((h, nfa_by_key[name_key(h)], "exact", None))
        else:
            rows.append((h, None, "unmatched", UNMATCHED_REASONS.get(h, "no NFA record")))
    return pd.DataFrame(rows, columns=["harrill_name", "nfa_chem", "match_type", "unmatched_reason"])
