"""Build NFA chemical-level viability anchors and independent potency tables.

Only local EPA PubChem exports, pinned EPA fit tables, and NTP DNT-DIVER 2018
release are read. NTP wells are from different experiments than NFA wells:
their join resolution is chemical, never physical well or plate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from neurotwin.data.toxcast_viability import AIDS, read_aid

NTP_PROTOCOL = "USEPA 91 neuron firing"
NTP_TO_NFA = {
    "bursts per minute": "burst_rate",
    "mean correlation": "correlation_coefficient_mean",
    "mean firing rate": "firing_rate_mean",
    "mutual information": "mutual_information_norm",
    "number active electrodes": "active_electrodes_number",
    "number actively bursting electrodes": "bursting_electrodes_number",
    "number network spikes": "network_spike_number",
    "percent spikes in burst": "per_burst_spike_percent",
    "percent spikes in network spike": "per_network_spike_spike_percent",
}
ZIP_MEMBERS = {
    "chemicals": "Data Release version 2/List_of_Chemicals.xlsx",
    "nested": "Data Release version 2/neurotox-data.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _cas_map(outer: zipfile.ZipFile) -> tuple[pd.DataFrame, dict]:
    listing = pd.read_excel(io.BytesIO(outer.read(ZIP_MEMBERS["chemicals"])), header=2)
    names = listing.iloc[:, [1, 2, 9]].copy()
    names.columns = ["ntp_chemical", "casrn", "dtxsid"]
    names = names[names.dtxsid.astype(str).str.fullmatch(r"DTXSID\d+")]
    names = names.dropna(subset=["casrn"])
    names.casrn = names.casrn.astype(str).str.strip()
    conflicts = names.groupby("casrn").dtxsid.nunique()
    bad = sorted(conflicts[conflicts.gt(1)].index.tolist())
    names = names[~names.casrn.isin(bad)].drop_duplicates(subset=["casrn", "dtxsid"])
    return names[["casrn", "dtxsid"]], {"casrn_conflicts": bad, "mapped_casrn": int(names.casrn.nunique())}


def _read_ntp(nested: zipfile.ZipFile, member: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(nested.open("neurotox-data/" + member), sep="\t", **kwargs)


def _ntp_viability(nested: zipfile.ZipFile, cas: pd.DataFrame, ids: set[str]) -> tuple[pd.DataFrame, dict]:
    cols = ["protocol", "endpoint", "is viability", "chemical name", "chemical casrn",
            "plate id", "plate row", "plate column", "concentration (µM)", "id",
            "response raw", "response normalized"]
    rows = _read_ntp(nested, "well-response.tsv", usecols=cols)
    rows = rows[rows.protocol.eq(NTP_PROTOCOL) & rows["is viability"].eq(True)].copy()
    rows = rows.rename(columns={"chemical casrn": "casrn", "chemical name": "ntp_chemical",
                                "plate id": "ntp_plate", "plate row": "ntp_row",
                                "plate column": "ntp_column", "concentration (µM)": "concentration_uM",
                                "id": "ntp_measurement_id", "response raw": "response_raw",
                                "response normalized": "response_normalized",
                                "endpoint": "ntp_endpoint"})
    rows.casrn = rows.casrn.astype(str).str.strip()
    rows = rows.merge(cas, on="casrn", how="left", validate="many_to_one")
    n_unmapped = int(rows.dtxsid.isna().sum())
    n_other = int(rows.dtxsid.notna().sum() - rows.dtxsid.isin(ids).sum())
    rows = rows[rows.dtxsid.isin(ids)].copy()
    rows["source"] = "NTP DNT-DIVER 2018"
    rows["resolution"] = "NTP well; NFA join by chemical"
    rows["endpoint"] = "viability_" + rows.ntp_endpoint.astype(str).str.replace(" ", "_", regex=False)
    rows["hitc"] = np.nan
    rows["hit_call"] = pd.NA
    rows["reported_outcome"] = pd.NA
    rows["ac50_uM"] = np.nan
    rows["ac50_log10_uM"] = np.nan
    rows["bmd_uM"] = np.nan
    rows["aid"] = pd.NA
    audit = {"all_protocol_viability_rows": int(n_unmapped + n_other + len(rows)),
             "unmapped_casrn_rows": n_unmapped, "mapped_outside_cohort_rows": n_other,
             "joined_rows": int(len(rows)), "joined_chemicals": int(rows.dtxsid.nunique()),
             "joined_by_endpoint": {str(k): int(v) for k, v in rows.groupby("ntp_endpoint").dtxsid.nunique().items()}}
    return rows, audit


def _ntp_potency(nested: zipfile.ZipFile, cas: pd.DataFrame, ids: set[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    counts = {}
    for method in ("hill", "curvep"):
        rows = _read_ntp(nested, f"bmc-{method}.tsv")
        rows.columns = [str(col).strip().replace(" ", "_") for col in rows.columns]
        rows = rows[rows.protocol.eq(NTP_PROTOCOL)].copy()
        rows = rows.rename(columns={"chemical_casrn": "casrn", "chemical_name": "ntp_chemical",
                                    "is_active": "active", "bmd": "bmd_uM",
                                    "bmdl": "bmdl_uM", "bmdu": "bmdu_uM"})
        rows.casrn = rows.casrn.astype(str).str.strip()
        rows = rows.merge(cas, on="casrn", how="left", validate="many_to_one")
        rows = rows[rows.dtxsid.isin(ids)].copy()
        rows["nfa_endpoint"] = rows.endpoint.map(NTP_TO_NFA)
        rows = rows[rows.nfa_endpoint.notna()]
        rows["source"] = "NTP DNT-DIVER 2018"
        rows["assay"] = "USEPA 91 neuron firing"
        rows["method"] = method
        rows["analysis"] = "cross_assay"
        rows["ac50_uM"] = np.nan
        rows["bmd_log10_uM"] = np.log10(pd.to_numeric(rows.bmd_uM, errors="coerce").where(lambda x: x.gt(0)))
        rows["hitc"] = np.nan
        rows["spid"] = pd.NA
        frames.append(rows[["source", "analysis", "method", "assay", "dtxsid", "spid", "casrn",
                            "endpoint", "nfa_endpoint", "active", "hitc", "ac50_uM", "bmd_uM",
                            "bmd_log10_uM", "bmdl_uM", "bmdu_uM"]])
        counts[method] = {"rows": int(len(rows)), "chemicals": int(rows.dtxsid.nunique()),
                          "active_rows": int(rows.active.fillna(False).sum())}
    return pd.concat(frames, ignore_index=True), counts


def _epa_potency(cache: Path, cohort: list[dict]) -> tuple[pd.DataFrame, dict]:
    by_name: dict[str, set[str]] = {}
    spids: dict[str, set[str]] = {}
    for item in cohort:
        key = _name(item["chemical"])
        by_name.setdefault(key, set()).add(item["dtxsid"])
        spids.setdefault(key, set()).add(item["spid"])
    frames = []
    counts = {}
    for analysis, filename in (
        ("DIV12", "DIV12_analysis_tcplfit2_res_02May2025.xlsx"),
        ("AUC", "AUC_analysis_tcplfit2_res_30Apr2025.xlsx"),
    ):
        rows = pd.read_excel(cache / filename, usecols=["chnm", "assay", "ac50", "bmd", "bmdl", "bmdu", "hitc"])
        rows["key"] = rows.chnm.map(_name)
        rows["dtxsid"] = rows.key.map(lambda k: next(iter(by_name[k])) if k in by_name and len(by_name[k]) == 1 else None)
        rows["spid"] = rows.key.map(lambda k: next(iter(spids[k])) if k in spids and len(spids[k]) == 1 else None)
        unmatched = int(rows.dtxsid.isna().sum())
        rows = rows[rows.dtxsid.notna()].copy()
        rows["nfa_endpoint"] = rows.assay.str.removeprefix("CCTE_Shafer_MEA_dev_")
        rows["endpoint"] = rows.nfa_endpoint
        rows["source"] = "EPA NFA tcplfit2 published fit"
        rows["analysis"] = analysis
        rows["method"] = "tcplfit2"
        rows["active"] = pd.to_numeric(rows.hitc, errors="coerce").ge(0.90)
        rows["ac50_uM"] = pd.to_numeric(rows.ac50, errors="coerce")
        rows["bmd_uM"] = pd.to_numeric(rows.bmd, errors="coerce")
        rows["bmdl_uM"] = pd.to_numeric(rows.bmdl, errors="coerce")
        rows["bmdu_uM"] = pd.to_numeric(rows.bmdu, errors="coerce")
        rows["bmd_log10_uM"] = np.log10(rows.bmd_uM.where(rows.bmd_uM.gt(0)))
        rows["casrn"] = pd.NA
        frames.append(rows[["source", "analysis", "method", "assay", "dtxsid", "spid", "casrn",
                            "endpoint", "nfa_endpoint", "active", "hitc", "ac50_uM", "bmd_uM",
                            "bmd_log10_uM", "bmdl_uM", "bmdu_uM"]])
        counts[analysis] = {"rows": int(len(rows)), "chemicals": int(rows.dtxsid.nunique()),
                            "unmatched_name_rows": unmatched, "active_rows": int(rows.active.sum())}
    return pd.concat(frames, ignore_index=True), counts


def build(root: Path) -> tuple[dict, dict]:
    root = root.resolve()
    probe = root / "data" / "neurotox_probe"
    out = root / "data" / "processed" / "epa_nfa"
    results = root / "repo" / "results"
    cohort = json.loads((results / "cohort_freeze.json").read_text(encoding="utf-8"))["cohort"]
    ids = {item["dtxsid"] for item in cohort}
    source_files = [probe / f"aid{aid}.csv" for aid in AIDS]
    ntp_path = probe / "ntp_dntdiver_Data_Release.zip"
    source_files.append(ntp_path)
    cache = out / "upstream"
    source_files += [cache / "DIV12_analysis_tcplfit2_res_02May2025.xlsx",
                     cache / "AUC_analysis_tcplfit2_res_30Apr2025.xlsx",
                     results / "cohort_freeze.json"]
    pub = pd.concat([read_aid(probe / f"aid{aid}.csv", aid) for aid in AIDS], ignore_index=True)
    duplicate_aid_ids = pub.groupby(["aid", "dtxsid"]).size()
    duplicate_aid_ids = int(duplicate_aid_ids.gt(1).sum())
    pub = pub[pub.dtxsid.isin(ids)].copy()
    with zipfile.ZipFile(ntp_path) as outer:
        cas, map_audit = _cas_map(outer)
        with zipfile.ZipFile(io.BytesIO(outer.read(ZIP_MEMBERS["nested"]))) as nested:
            ntp_wells, well_audit = _ntp_viability(nested, cas, ids)
            ntp_bmc, bmc_counts = _ntp_potency(nested, cas, ids)
    epa_bmc, epa_counts = _epa_potency(cache, cohort)
    viability = pd.concat([pub, ntp_wells], ignore_index=True)
    potency = pd.concat([epa_bmc, ntp_bmc], ignore_index=True)
    from neurotwin.models.trajectory import FEATURES
    unexpected = sorted(set(potency.nfa_endpoint) - set(FEATURES))
    if unexpected:
        raise ValueError(f"Unmapped NFA endpoints: {unexpected}")
    if duplicate_aid_ids:
        raise ValueError(f"Conflicting PubChem AID/DTXSID rows: {duplicate_aid_ids}")
    out.mkdir(parents=True, exist_ok=True)
    viability.to_parquet(out / "viability.parquet", index=False)
    potency.to_parquet(out / "reference_potency.parquet", index=False)
    pub_coverage = {endpoint: int(part.dtxsid.nunique()) for endpoint, part in pub.groupby("endpoint")}
    missing = {endpoint: sorted(ids - set(part.dtxsid)) for endpoint, part in pub.groupby("endpoint")}
    ntp_ids = set(ntp_wells.dtxsid)
    ntp_catalog_ids = set(cas.dtxsid)
    label_by_id: dict[str, set[str]] = {}
    for item in cohort:
        label_by_id.setdefault(item["dtxsid"], set()).add(item["label"] or "Unknown")
    label_by_id = {key: values for key, values in label_by_id.items()}
    label_strata = {"Positive", "Negative", "Unknown"}
    stratified_coverage = {}
    for endpoint, part in pub.groupby("endpoint"):
        present = set(part.dtxsid)
        stratified_coverage[endpoint] = {
            label: {"cohort": sum(label in label_by_id[d] for d in ids),
                    "joined": sum(label in label_by_id[d] for d in present)}
            for label in sorted(label_strata)
        }
    stratified_coverage["NTP_USEPA_91_viability"] = {
        label: {"cohort": sum(label in label_by_id[d] for d in ids),
                "joined": sum(label in label_by_id[d] for d in ntp_ids)}
        for label in sorted(label_strata)
    }
    pub_conflicts = {endpoint: {"hit_call_conflict_chemicals": int(part.hit_call_conflict.sum()),
                                "multiple_sid_chemicals": int(part.n_source_records.gt(1).sum()),
                                "active_chemicals": int(part.hit_call.fillna(False).sum()),
                                "inactive_chemicals": int(part.hit_call.eq(False).sum())}
                     for endpoint, part in pub.groupby("endpoint")}
    viability_audit = {
        "cohort_spids": len(cohort), "cohort_unique_dtxsid": len(ids),
        "join_resolution": {"PubChem": "chemical DTXSID", "NTP": "NTP well to NFA chemical via CASRN/DTXSID"},
        "same_physical_well_as_nfa": False,
        "pubchem_chemical_coverage": pub_coverage, "pubchem_missing_dtxsid": missing,
        "coverage_by_dnt_label": stratified_coverage,
        "pubchem_call_strata": pub_conflicts,
        "pubchem_duplicate_aid_dtxsid_groups": duplicate_aid_ids,
        "ntp": {**map_audit, **well_audit},
        "ntp_unjoined_cohort_dtxsid": sorted(ids - ntp_ids),
        "ntp_unjoined_by_reason": {
            "absent_from_ntp_chemical_catalog": sorted(ids - ntp_catalog_ids),
            "in_catalog_but_no_matching_USEPA_91_viability_well": sorted((ids & ntp_catalog_ids) - ntp_ids),
        },
        "ntp_unjoined_rule": "Exact CASRN/DTXSID only; no name or salt guessing.",
        "source_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in source_files},
    }
    potency_audit = {
        "units": "AC50, BMD/BMC, BMDL, BMDU in micromolar; bmd_log10_uM is log10(numeric micromolar).",
        "inactive_rule": "Retain numeric fits for audit; only active=True and positive finite potency is a reference call.",
        "epa_fits": epa_counts, "ntp_bmc": bmc_counts,
        "endpoint_map_ntp_to_nfa": NTP_TO_NFA,
        "nfa_feature_count": len(FEATURES),
        "ntp_mapped_endpoint_count": len(NTP_TO_NFA),
        "ntp_cross_assay_caveat": "NTP USEPA 91 neuron firing is a distinct assay, not NFA DIV12 or AUC wells.",
        "source_sha256": viability_audit["source_sha256"],
    }
    for path, audit in ((results / "viability_join_audit.json", viability_audit),
                        (results / "reference_potency_audit.json", potency_audit)):
        path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return viability_audit, potency_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    viability, potency = build(args.root)
    print(json.dumps({"pubchem": viability["pubchem_chemical_coverage"],
                      "ntp_well_chemicals": viability["ntp"]["joined_chemicals"],
                      "ntp_well_rows": viability["ntp"]["joined_rows"],
                      "epa_fit_rows": potency["epa_fits"], "ntp_bmc_rows": potency["ntp_bmc"]},
                     indent=2))


if __name__ == "__main__":
    main()
