"""Kosnik et al. (2020) acute MEA concentration-response screen of 384 ToxCast chemicals (X3).

Source: "Data set for Kosnik et al.", US EPA ScienceHub, doi:10.23719/1504294
(MEA_All_Data_Scripts.zip; EPA public-domain data; the bundled R scripts carry no licence and are
NOT redistributed or executed here, only their published outputs are read).

Mature rat primary cortical networks are exposed ACUTELY (not during development); 43 MEA
parameters plus LDH and AlamarBlue cytotoxicity (45 assay components, 90 up/down endpoints) are
fitted with tcpl. We use the authors' own processed outputs:
- MEA_tcpl_Hits.csv: tcpl level-5 hits after the authors' cytotoxicity filter (hits whose AC50 is
  at or above the AlamarBlue cytotoxicity AC50 removed).
- MEA_Final_Chemical_Parameter_Hits_tcpl_Data.csv: the published chemical-level call, i.e. chemicals
  with hits in >= 3 of the 15 machine-learning-ranked parameters ("acute MEA active").
  Potency = min modl_ga (log10 uM) over those final hits, as in the authors' script 7.
- MEA_Truth_Chemicals.csv: 73 curated chemicals, 41 'Neuroactive' / 32 'Negative'. These labels
  encode literature NEUROACTIVITY (acute effects on neuronal function), NOT a molecular target and
  NOT developmental neurotoxicity. The ML-ranked parameter set behind the final call was chosen
  with this truth set, so the acute MEA call is not an independent test of it.

Identity: Kosnik ships CASRN + ToxCast names. CASRN is mapped to DSSTox substance IDs with a local
crosswalk built only from EPA files already in the workspace; NFA chemicals are matched by DTXSID
(through CASRN) and, failing that, by normalised preferred name.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SUB = "MEA_Data"
N_CHEM_EXPECTED = 384
TRUTH_EXPECTED = {"Neuroactive": 41, "Negative": 32}
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
CYTOTOX_AEIDS = {1, 2, 3, 4}          # LDH_up/dn, AB_up/dn


def name_key(x) -> str:
    return " ".join(str(x).casefold().split())


def clean_cas(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    return s if CAS_RE.match(s) else None


def load(kdir: Path, with_input: bool = False) -> dict:
    """Read the published Kosnik tables. kdir is the extracted folder that contains MEA_Data/."""
    p = Path(kdir) / SUB
    out = {
        "chemicals": pd.read_csv(p / "MEA_Chemicals.csv"),
        "endpoints": pd.read_csv(p / "MEA_Endpoints.csv"),
        "truth": pd.read_csv(p / "MEA_Truth_Chemicals.csv"),
        "hits": pd.read_csv(p / "MEA_tcpl_Hits.csv"),
        "final": pd.read_csv(p / "MEA_Final_Chemical_Parameter_Hits_tcpl_Data.csv"),
    }
    if with_input:
        out["input"] = pd.read_csv(p / "MEA_tcpl_Input.csv", usecols=["spid", "apid", "coli", "rowi",
                                                                        "conc", "acsn", "wllt", "wllq"])
    return out


def audit(k: dict) -> dict:
    ch, tr = k["chemicals"], k["truth"]
    a = {"n_spids": int(ch.spid.nunique()), "n_casrn": int(ch.casn.nunique()),
         "n_endpoints_aeid": int(len(k["endpoints"])),
         "truth_counts": {str(t): int(n) for t, n in tr.Type.value_counts().items()},
         "n_final_active_spids": int(k["final"].spid.nunique()),
         "n_any_filtered_hit_spids": int(k["hits"].loc[~k["hits"].aeid.isin(CYTOTOX_AEIDS), "spid"].nunique())}
    if "input" in k:
        i = k["input"]
        a.update({"input_rows": int(len(i)), "input_spids": int(i.spid.nunique()),
                  "input_parameters": int(i.acsn.nunique()), "input_plates": int(i.apid.nunique()),
                  "input_concentrations": int(i.conc.nunique())})
    a["checks_pass"] = bool(a["n_spids"] == N_CHEM_EXPECTED and a["truth_counts"] == TRUTH_EXPECTED)
    return a


def acute_calls(k: dict) -> pd.DataFrame:
    """Per spid: published final call/potency, liberal any-hit call, cytotoxicity."""
    ch = k["chemicals"].copy()
    fin = k["final"]
    fcount = fin.groupby("spid").endpoint.nunique()
    fbmc = fin.groupby("spid").modl_ga.min()
    hits = k["hits"]
    nonc = hits[~hits.aeid.isin(CYTOTOX_AEIDS)]
    ch["final_active"] = ch.spid.isin(fin.spid)
    ch["n_final_endpoints"] = ch.spid.map(fcount).fillna(0).astype(int)
    ch["acute_bmc_log10"] = ch.spid.map(fbmc)
    ch["any_hit_active"] = ch.spid.isin(nonc.spid)
    ch["any_hit_bmc_log10"] = ch.spid.map(nonc.groupby("spid").modl_ga.min())
    ch["cas"] = ch.casn.map(clean_cas)
    ch["nk"] = ch.chnm.map(name_key)
    return ch


def cas_crosswalk(root: Path, cache: Path | None = None) -> pd.DataFrame:
    """CASRN <-> DTXSID pairs from EPA files already in the workspace (no network).
    The Excel sources are slow to parse, so the result can be cached as parquet."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    P = Path(root) / "data/neurotox_probe"
    parts = []
    f = P / "nfa_refine/annotate_dnt_ref_chems.csv"
    if f.exists():
        a = pd.read_csv(f)
        parts.append(a[["dsstox_substance_id", "casn"]].set_axis(["dtxsid", "casrn"], axis=1))
    f = Path(root) / "data/processed/epa_nfa/upstream/Compare_ref_chems_23mar26_edit.xlsx"
    if f.exists():
        a = pd.read_excel(f)
        parts.append(a[["dsstox_substance_id", "CASRN"]].set_axis(["dtxsid", "casrn"], axis=1))
    f = P / "nfa_refine/DNT_ref_list_17sept2025.xlsx"
    if f.exists():
        a = pd.read_excel(f)
        parts.append(a[["DTXSID", "CASRN"]].set_axis(["dtxsid", "casrn"], axis=1))
    f = P / "shafer2019_mea_toxcast_aeds.xlsx"
    if f.exists():
        a = pd.read_excel(f, sheet_name="MEA_data")
        parts.append(a[["DTXSID", "Casrn"]].set_axis(["dtxsid", "casrn"], axis=1))
    f = Path(root) / "data/processed/epa_nfa/viability.parquet"
    if f.exists():
        parts.append(pd.read_parquet(f, columns=["dtxsid", "casrn"]))
    cw = pd.concat(parts, ignore_index=True).dropna()
    cw["dtxsid"] = cw.dtxsid.astype(str).str.strip()
    cw["casrn"] = cw.casrn.map(clean_cas)
    cw = cw.dropna()
    cw = cw[cw.dtxsid.str.fullmatch(r"DTXSID\d+")].drop_duplicates().reset_index(drop=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        cw.to_parquet(cache)
    return cw


def match_nfa(nfa: pd.DataFrame, calls: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    """nfa: columns chem, dtxsid. Returns one row per matched NFA chemical with the Kosnik call
    aggregated over its matched spids (active if any spid active; potency = min)."""
    cas_of = cw.groupby("dtxsid").casrn.apply(set).to_dict()
    rows = []
    for chem, dtx in nfa[["chem", "dtxsid"]].itertuples(index=False):
        cas = cas_of.get(dtx, set())
        by_cas = calls[calls.cas.isin(cas)]
        how = "dtxsid_via_casrn"
        sel = by_cas
        if sel.empty:
            sel = calls[calls.nk == name_key(chem)]
            how = "name"
        if sel.empty:
            continue
        rows.append({"chem": chem, "dtxsid": dtx, "match": how, "kosnik_spids": ";".join(sel.spid),
                     "kosnik_name": ";".join(sorted(set(sel.chnm))), "kosnik_cas": ";".join(sorted(set(sel.casn))),
                     "acute_final_active": bool(sel.final_active.any()),
                     "acute_bmc_log10": float(sel.acute_bmc_log10.min()) if sel.final_active.any() else np.nan,
                     "acute_n_final_endpoints": int(sel.n_final_endpoints.max()),
                     "acute_any_hit_active": bool(sel.any_hit_active.any())})
    return pd.DataFrame(rows)


def match_truth(nfa: pd.DataFrame, truth: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    """Kosnik truth-set label for NFA chemicals (by DTXSID through CASRN, else by name)."""
    cas_of = cw.groupby("dtxsid").casrn.apply(set).to_dict()
    tr = truth.assign(cas=truth.CASRN.map(clean_cas), nk=truth.Name.map(name_key))
    rows = []
    for chem, dtx in nfa[["chem", "dtxsid"]].itertuples(index=False):
        sel = tr[tr.cas.isin(cas_of.get(dtx, set()))]
        how = "dtxsid_via_casrn"
        if sel.empty:
            sel = tr[tr.nk == name_key(chem)]
            how = "name"
        if sel.empty:
            continue
        if sel.Type.nunique() > 1:
            continue
        rows.append({"chem": chem, "dtxsid": dtx, "truth_match": how, "truth_name": sel.Name.iloc[0],
                     "truth_label": sel.Type.iloc[0], "neuroactive": sel.Type.iloc[0] == "Neuroactive"})
    return pd.DataFrame(rows)
