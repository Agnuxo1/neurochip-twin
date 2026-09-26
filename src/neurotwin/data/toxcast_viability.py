"""Local PubChem CCTE_Shafer_MEA_dev AB/LDH viability calls.

PubChem exports put five metadata rows after the CSV header. AC50 and BMD
units are explicitly MICROMOLAR in RESULT_UNIT. HITC >= 0.90 is the export's
documented active threshold; an AC50 on an inactive row is not a potency call.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

AIDS = {"2284083": "viability_AB", "2284068": "viability_LDH"}
DTXSID = re.compile(r"(DTXSID\d+)")


def read_aid(path: Path, aid: str) -> pd.DataFrame:
    if aid not in AIDS:
        raise ValueError(f"Unsupported viability AID: {aid}")
    with path.open(encoding="utf-8-sig") as stream:
        rows = [next(stream).rstrip("\n") for _ in range(5)]
    if "MICROMOLAR" not in rows[3] or "hitc < 0.90" not in rows[2]:
        raise ValueError(f"Unexpected PubChem units or hit threshold in {path}")
    frame = pd.read_csv(path, skiprows=range(1, 5))
    frame["dtxsid"] = frame.PUBCHEM_ACTIVITY_URL.str.extract(DTXSID, expand=False)
    frame = frame[frame.dtxsid.notna()].copy()
    frame["hitc"] = pd.to_numeric(frame.HITC, errors="coerce")
    frame["ac50_uM"] = pd.to_numeric(frame.AC50, errors="coerce")
    frame["bmd_uM"] = pd.to_numeric(frame.BMD, errors="coerce")
    frame["hit_call"] = frame.hitc.ge(0.90)
    frame["aid"] = aid
    frame["endpoint"] = AIDS[aid]
    frame["source"] = "PubChem CCTE_Shafer_MEA_dev"
    frame["resolution"] = "chemical"
    frame = frame.rename(columns={"PUBCHEM_ACTIVITY_OUTCOME": "reported_outcome"})
    records = []
    for dtxsid, group in frame.groupby("dtxsid", sort=True):
        conflict = group.hit_call.nunique() > 1
        active = bool(group.hit_call.all()) and not conflict
        ac50 = group.ac50_uM[group.ac50_uM.gt(0)] if active else pd.Series(dtype=float)
        bmd = group.bmd_uM[group.bmd_uM.gt(0)] if active else pd.Series(dtype=float)
        records.append({
            "source": "PubChem CCTE_Shafer_MEA_dev", "aid": aid, "endpoint": AIDS[aid],
            "dtxsid": dtxsid, "resolution": "chemical; aggregated PubChem SIDs",
            "hitc": float(group.hitc.median()) if group.hitc.notna().any() else np.nan,
            "hit_call": pd.NA if conflict else active,
            "hit_call_conflict": conflict, "reported_outcome": "discordant" if conflict else
                ("Active" if active else "Inactive/Inconclusive"),
            "ac50_uM": float(ac50.median()) if len(ac50) else np.nan,
            "ac50_min_uM": float(ac50.min()) if len(ac50) else np.nan,
            "ac50_max_uM": float(ac50.max()) if len(ac50) else np.nan,
            "bmd_uM": float(bmd.median()) if len(bmd) else np.nan,
            "n_source_records": len(group),
            "pubchem_sids": ";".join(sorted(group.PUBCHEM_SID.dropna().astype(str))),
        })
    result = pd.DataFrame(records)
    result["hit_call"] = result.hit_call.astype("boolean")
    result["ac50_log10_uM"] = np.log10(result.ac50_uM.where(result.ac50_uM.gt(0)))
    return result
