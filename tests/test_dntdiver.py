"""G2b integrity checks; production test uses the locally rebuilt tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurotwin.data.dntdiver import NTP_TO_NFA
from neurotwin.data.toxcast_viability import read_aid
from neurotwin.models.trajectory import FEATURES

ROOT = Path(__file__).resolve().parents[2]


def test_pubchem_conflicting_sids_abstain(tmp_path: Path) -> None:
    header = ("PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,"
              "PUBCHEM_EXT_DATASOURCE_SMILES,PUBCHEM_ACTIVITY_OUTCOME,"
              "PUBCHEM_ACTIVITY_SCORE,PUBCHEM_ACTIVITY_URL,PUBCHEM_ASSAYDATA_COMMENT,AC50,HITC,BMD")
    rows = [header, "RESULT_TYPE,,,,,,,,FLOAT,FLOAT,FLOAT",
            'RESULT_DESCR,,,,,,,,"hitc < 0.90 are inactive",,',
            "RESULT_UNIT,,,,,,,,MICROMOLAR,,MICROMOLAR",
            "RESULT_IS_ACTIVE_CONCENTRATION,,,,,,,,TRUE,,",
            "1,101,,,Active,,https://example.org/DTXSID123,,1,1,2",
            "2,102,,,Inactive,,https://example.org/DTXSID123,,3,0.5,4",
            "3,103,,,Active,,https://example.org/DTXSID456,,10,0.95,12"]
    path = tmp_path / "aid2284083.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = read_aid(path, "2284083").set_index("dtxsid")
    assert len(result) == 2
    assert pd.isna(result.loc["DTXSID123", "hit_call"])
    assert result.loc["DTXSID123", "hit_call_conflict"]
    assert pd.isna(result.loc["DTXSID123", "ac50_uM"])
    assert bool(result.loc["DTXSID456", "hit_call"])
    assert result.loc["DTXSID456", "ac50_uM"] == 10


def test_production_joins_and_units() -> None:
    base = ROOT / "data" / "processed" / "epa_nfa"
    viability = pd.read_parquet(base / "viability.parquet")
    potency = pd.read_parquet(base / "reference_potency.parquet")
    audit = json.loads((ROOT / "repo" / "results" / "viability_join_audit.json").read_text())
    assert audit["same_physical_well_as_nfa"] is False
    assert audit["pubchem_chemical_coverage"] == {"viability_AB": 243, "viability_LDH": 243}
    pub = viability[viability.source.eq("PubChem CCTE_Shafer_MEA_dev")]
    assert not pub.duplicated(["dtxsid", "aid"]).any()
    assert pub.loc[pub.hit_call.eq(False).fillna(False), "ac50_uM"].isna().all()
    assert pub.loc[pub.hit_call_conflict, "hit_call"].isna().all()
    ntp = viability[viability.source.eq("NTP DNT-DIVER 2018")]
    assert ntp.dtxsid.nunique() == 64
    assert len(ntp) == 4032
    assert ntp.ntp_measurement_id.notna().all()
    assert ntp.resolution.eq("NTP well; NFA join by chemical").all()
    assert len(potency[potency.analysis.eq("DIV12")]) == 1785
    assert len(potency[potency.analysis.eq("AUC")]) == 4335
    assert potency[potency.analysis.eq("cross_assay")].dtxsid.nunique() == 64
    finite = potency.bmd_uM.gt(0) & potency.bmd_log10_uM.notna()
    assert np.allclose(potency.loc[finite, "bmd_log10_uM"], np.log10(potency.loc[finite, "bmd_uM"]))


def test_endpoint_correspondence_is_complete_and_explicit() -> None:
    assert len(FEATURES) == 17
    assert len(NTP_TO_NFA) == 9
    assert set(NTP_TO_NFA.values()).issubset(set(FEATURES))
    potency = pd.read_parquet(ROOT / "data" / "processed" / "epa_nfa" / "reference_potency.parquet")
    assert set(potency.nfa_endpoint) == set(FEATURES)
    assert set(potency.loc[potency.analysis.eq("cross_assay"), "nfa_endpoint"]) == set(NTP_TO_NFA.values())
