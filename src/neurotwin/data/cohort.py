"""Freeze the EPA NFA cohort and G0 evaluation splits without model outputs.

Run from the workspace root after ``python -m neurotwin.data.epa_nfa``.
This module reads its tidy tables and existing chemical split; it never changes
either artifact. The manifest has a canonical payload digest, and a sidecar
holds the SHA256 of the complete JSON file (a self-hash cannot be embedded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260926
FOLDS = 5
SOURCE_FILES = (
    "All_DIV_Data.Rdata",
    "annotate_dnt_ref_chems.csv",
    "dnt_ref_tbl.xlsx",
    "dnt_ref_oecd_culbreth_curate.xlsx",
    "DNT_ref_list_17sept2025.xlsx",
    "EFSA_Mundy_2024_list_DNT%20IVB.xlsx",
    "Results_sens_spec_BA_refine.csv",
    "rval_zval_table.csv",
)
SPLIT_FILES = (
    "splits_epa_nfa.json", "splits_dose.json", "splits_plate.json", "splits_family.json"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _key(name: str) -> str:
    return " ".join(str(name).casefold().split())


def _read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    processed = root / "data" / "processed" / "epa_nfa"
    wells = pd.read_parquet(processed / "wells.parquet")
    labels = pd.read_parquet(processed / "dnt_labels.parquet")
    existing_path = root / "repo" / "results" / "splits_epa_nfa.json"
    original = existing_path.read_bytes()
    chemical_split = json.loads(original)
    if len(wells) != 31_756 or len(labels) != 267 or labels.spid.nunique() != 267:
        raise ValueError("EPA cohort counts differ from audited ingestion")
    if set(wells.spid) != set(labels.spid) or wells.dtxsid.isna().any():
        raise ValueError("Tidy wells and DNT identity table disagree")
    if len(chemical_split["assignments"]) != 267 or chemical_split["n_folds"] != FOLDS:
        raise ValueError("Existing chemical split differs from G0 cohort")
    if wells.wllq.eq(0).any():
        raise ValueError("wllq=0 wells must be excluded before G0 freeze")
    if {item["spid"] for item in chemical_split["assignments"]} != set(labels.spid):
        raise ValueError("Existing chemical split SPIDs differ from G0 cohort")
    if existing_path.read_bytes() != original:
        raise ValueError("Existing chemical split changed during freeze")
    return wells, labels, chemical_split


def cohort_document(wells: pd.DataFrame, labels: pd.DataFrame) -> dict:
    records = []
    for row in labels.sort_values("spid").itertuples(index=False):
        records.append(dict(spid=row.spid, dtxsid=row.dtxsid, chemical=row.chemical,
                            label=None if pd.isna(row.label) else row.label,
                            id_source=row.id_source))
    aliases = [
        dict(group="valproate_active_moiety",
             members=[dict(name="Sodium valproate", dtxsid="DTXSID5037072", in_nfa=True),
                      dict(name="Valproic acid", dtxsid="DTXSID6023733", in_nfa=False)],
             relation="salt and parent acid", collapse_identity=False,
             decision="Keep distinct DTXSIDs; link for future analog leakage checks.",
             evidence="dnt_ref_oecd_culbreth_curate.xlsx, DNT_ref_list_17sept2025.xlsx"),
        dict(group="heptachlor_metabolite_pair",
             members=[dict(name="Heptachlor epoxide B", dtxsid="DTXSID1024126", in_nfa=True),
                      dict(name="Heptachlor", dtxsid="DTXSID3020679", in_nfa=False)],
             relation="parent pesticide and epoxide metabolite", collapse_identity=False,
             decision="Do not treat as exact aliases; family holdout groups both as organochlorines.",
             evidence="annotate_dnt_ref_chems.csv, dnt_ref_oecd_culbreth_curate.xlsx"),
    ]
    qc = {
        "exclude_wllq_zero": True,
        "observed_wllq_values": sorted(int(value) for value in wells.wllq.unique()),
        "n_rows_wllq_zero": int(wells.wllq.eq(0).sum()),
        "valid_div": [5, 7, 9, 12],
        "concentration_unit": "uM",
        "vehicle_definition": "concentration_uM == 0",
        "n_source_rows": len(wells),
        "n_imputed_rows_retained": int(wells.was_imputed.sum()),
        "imputed_policy": "Retain with was_imputed flag; analyze exclusion as sensitivity test.",
        "physical_duplicates": "Retain distinct source chemical records; count identical physical vehicle wells once in normalization.",
        "missing_raw_features": {feature: int(wells[feature].isna().sum())
                                 for feature in sorted(column[len("z_vehicle_"):] for column in wells if column.startswith("z_vehicle_"))},
    }
    normalization = dict(group_by=["plate", "date", "div"],
                         statistic="median and 1.4826 * median absolute deviation of unique physical vehicle wells",
                         formula="z_vehicle_feature = (feature - vehicle_median_feature) / vehicle_mad_feature",
                         zero_mad="z is missing; raw feature retained",
                         n_vehicle_source_rows=int(wells.well_type.eq("vehicle").sum()),
                         use_labels=False,
                         epa_preprocessed_z_equivalence=False)
    return dict(schema_version=1, cohort=records, aliases=aliases, qc=qc,
                normalization=normalization,
                summary=dict(n_spids=len(records), n_dtxsid=labels.dtxsid.nunique(),
                             n_names=labels.chemical.nunique(), n_wells=len(wells)),
                component_sha256=dict(cohort=_digest(records), aliases=_digest(aliases),
                                      qc=_digest(qc), normalization=_digest(normalization)))


def dose_split(wells: pd.DataFrame) -> dict:
    treated = wells[wells.concentration_uM.gt(0)][["dtxsid", "concentration_uM"]].drop_duplicates()
    assignments = []
    for dtxsid, group in treated.groupby("dtxsid", sort=True):
        levels = sorted(float(value) for value in group.concentration_uM.unique())
        # Per-chemical seed prevents a new chemical from reshuffling earlier ones.
        salt = int.from_bytes(hashlib.sha256(dtxsid.encode()).digest()[:8], "big")
        rng = np.random.default_rng(SEED ^ salt)
        shuffled = list(rng.permutation(levels))
        offset = salt % FOLDS
        for position, concentration in enumerate(shuffled):
            assignments.append(dict(dtxsid=dtxsid, concentration_uM=float(concentration),
                                    test_fold=(position + offset) % FOLDS))
    assignments.sort(key=lambda item: (item["dtxsid"], item["concentration_uM"]))
    if len(assignments) != len(treated):
        raise ValueError("Dose-level assignment is not one-to-one")
    return dict(schema_version=1, seed=SEED, n_folds=FOLDS,
                rule="For each DTXSID, shuffle unique nonzero concentrations with a stable per-chemical seed and assign levels round-robin to held-out folds with a per-chemical starting offset.",
                vehicle_policy="All zero-concentration wells remain available as controls in every fold; never scored as held-out dose.",
                test_unit="Complete DTXSID x concentration level, across all plates, DIVs and SPIDs",
                n_chemicals=treated.dtxsid.nunique(), n_levels=len(assignments),
                assignments=assignments)


def _balanced_group_folds(sizes: dict[str, int]) -> dict[str, int]:
    rng = np.random.default_rng(SEED)
    names = sorted(sizes)
    rng.shuffle(names)
    ordered = sorted(names, key=lambda name: sizes[name], reverse=True)
    load = [0] * FOLDS
    groups = [0] * FOLDS
    assignment = {}
    for name in ordered:
        fold = min(range(FOLDS), key=lambda i: (load[i], groups[i], i))
        assignment[name] = fold
        load[fold] += sizes[name]
        groups[fold] += 1
    return assignment


def plate_split(wells: pd.DataFrame) -> dict:
    sizes = wells.groupby("plate").size().to_dict()
    assigned = _balanced_group_folds(sizes)
    records = []
    for (plate, date), group in wells.groupby(["plate", "date"], sort=True):
        records.append(dict(plate=plate, date=int(date), test_fold=assigned[plate],
                            n_well_rows=len(group)))
    return dict(schema_version=1, seed=SEED, n_folds=FOLDS,
                rule="Assign entire plate IDs, including every date and DIV, to one fold; balance source-row counts greedily without model outcomes.",
                test_unit="plate (all associated dates and DIVs)",
                n_plates=len(assigned), n_plate_dates=len(records),
                fold_row_counts=[sum(row["n_well_rows"] for row in records if row["test_fold"] == fold)
                                 for fold in range(FOLDS)],
                assignments=records)


def _family_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = _key(str(value).replace("\xa0", " ")).replace(" ", "_")
    aliases = {"op": "organophosphate", "pyrethroid2": "pyrethroid",
               "pyrethoid": "pyrethroid", "drug": "pharmaceutical",
               "food_additive": "food_additive", "metal,_organic": "metal"}
    return aliases.get(text, text) if text else None


def family_split(labels: pd.DataFrame, annotation_path: Path) -> dict:
    annotation = pd.read_csv(annotation_path)
    if annotation.dsstox_substance_id.duplicated().any():
        raise ValueError("Annotated DTXSID is not unique")
    by_id = annotation.set_index("dsstox_substance_id")
    chemicals = labels.groupby("dtxsid").agg(spids=("spid", lambda s: sorted(s)),
                                              names=("chemical", lambda s: sorted(set(s)))).reset_index()
    records = []
    for row in chemicals.itertuples(index=False):
        if row.dtxsid in by_id.index:
            note = by_id.loc[row.dtxsid]
            fields = {"class_shafer": _family_name(note["class.shafer"]),
                      "neuro_class": _family_name(note["neuro.class"]),
                      "class_general": _family_name(note["Class"])}
        else:
            fields = {"class_shafer": None, "neuro_class": None, "class_general": None}
        family = fields["class_shafer"] or fields["neuro_class"] or fields["class_general"]
        records.append(dict(dtxsid=row.dtxsid, spids=row.spids, names=row.names,
                            family=family, family_source=("class.shafer" if fields["class_shafer"] else
                                                          "neuro.class" if fields["neuro_class"] else
                                                          "Class" if fields["class_general"] else None),
                            **fields))
    sizes = Counter(row["family"] for row in records if row["family"] is not None)
    family_folds = _balanced_group_folds(dict(sizes))
    for row in records:
        row["test_fold"] = family_folds.get(row["family"])
    eligible = [row for row in records if row["test_fold"] is not None]
    return dict(schema_version=1, seed=SEED, n_folds=FOLDS,
                rule="Prefer class.shafer, then neuro.class, then Class; normalize known synonyms and keep every annotated family wholly within one fold.",
                missing_class_policy="Exclude unannotated chemicals from both train and test in this family extrapolation experiment.",
                rdkit_note="No SMILES column exists in the supplied annotation file; RDKit scaffolds are not inferred.",
                n_chemicals=len(records), n_eligible_chemicals=len(eligible),
                n_excluded_chemicals=len(records) - len(eligible),
                n_families=len(family_folds), family_to_fold=family_folds,
                fold_chemical_counts=[sum(row["test_fold"] == fold for row in eligible)
                                      for fold in range(FOLDS)],
                assignments=records)


def _git_head(repo: Path) -> tuple[str | None, str]:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "HEAD"],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip(), "committed"
    return None, "unborn_repository_no_HEAD"


def _provenance_md(root: Path) -> str:
    raw = root / "data" / "neurotox_probe" / "nfa_refine"
    upstream = root / "data" / "processed" / "epa_nfa" / "upstream"
    rows = []
    for name in SOURCE_FILES:
        path = raw / name
        rows.append((f"data/neurotox_probe/nfa_refine/{name}", _file_sha256(path),
                     "[USEPA NFA repository](https://github.com/USEPA/CompTox-DNT-NFA-Refinement)",
                     "US EPA work; confirm any third-party contribution before redistribution"))
    from neurotwin.data.epa_nfa import EPA_COMMIT, UPSTREAM

    for name in UPSTREAM:
        path = upstream / name
        rows.append((f"data/processed/epa_nfa/upstream/{name}", _file_sha256(path),
                     f"[USEPA commit {EPA_COMMIT[:7]}](https://github.com/USEPA/CompTox-DNT-NFA-Refinement/tree/{EPA_COMMIT})",
                     "US EPA work; pinned by commit and SHA256"))
    lines = ["# Data provenance and reuse inventory", "",
             "G0 uses EPA NFA well trajectories, DNT references and EPA-fitted endpoint calls. The exact files and hashes below are inputs to `freeze_g0.json`.",
             "", "| Input | SHA256 | Source | Reuse status |", "| --- | --- | --- | --- |"]
    for name, hash_, source, license_ in rows:
        lines.append(f"| `{name}` | `{hash_}` | {source} | {license_} |")
    lines.extend(["", "## Related sources", "",
                  "- Brewer 4-compartment MEA: [Zenodo record 10257483](https://zenodo.org/records/10257483), CC0. Its pinned hashes are in `repo/scripts/brewer_sha256sums.txt`; this is the input for ChipLayer, not for the NFA G0 cohort.",
                  "- NTP DNT-DIVER, PubChem and Kosnik are reserved for later validation tasks. Their licenses and source hashes must be added before those artifacts enter a release.",
                  "", "## Scope and limits", "",
                  "The EPA repository does not expose a machine-readable license in the checked-out source; the project identifies these as US Government works. This inventory does not assert rights over third-party material that might be embedded in an upstream spreadsheet. Original files are cached under `data/processed/` and are not added to the public code repository.",
                  "The five-fold chemical split is the pre-existing `splits_epa_nfa.json` and is unchanged. Dose, plate and family tests use only cohort identities, concentrations and annotations; no model predictions determine their folds.", ""])
    return "\n".join(lines)


def build(root: Path) -> dict:
    root = root.resolve()
    results = root / "repo" / "results"
    docs = root / "repo" / "docs"
    results.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    wells, labels, chemical_split = _read_inputs(root)
    original_split_hash = _file_sha256(results / "splits_epa_nfa.json")
    cohort = cohort_document(wells, labels)
    dose = dose_split(wells)
    plate = plate_split(wells)
    family = family_split(labels, root / "data" / "neurotox_probe" / "nfa_refine" / "annotate_dnt_ref_chems.csv")
    _write_json(results / "cohort_freeze.json", cohort)
    _write_json(results / "splits_dose.json", dose)
    _write_json(results / "splits_plate.json", plate)
    _write_json(results / "splits_family.json", family)
    (docs / "PROVENANCE.md").write_text(_provenance_md(root), encoding="utf-8")
    if _file_sha256(results / "splits_epa_nfa.json") != original_split_hash:
        raise ValueError("Existing chemical split changed during G0 generation")
    from neurotwin.data.epa_nfa import UPSTREAM

    inputs = [f"data/neurotox_probe/nfa_refine/{name}" for name in SOURCE_FILES]
    inputs.extend(f"data/processed/epa_nfa/upstream/{name}" for name in UPSTREAM)
    derived = ["data/processed/epa_nfa/wells.parquet",
               "data/processed/epa_nfa/vehicle_stats.parquet",
               "data/processed/epa_nfa/dnt_labels.parquet",
               "repo/results/epa_baseline_repro.json", "repo/results/cohort_freeze.json",
               *(f"repo/results/{name}" for name in SPLIT_FILES),
               "repo/docs/PROVENANCE.md", "repo/docs/epa_nfa.md",
               "repo/src/neurotwin/data/cohort.py",
               "repo/src/neurotwin/data/epa_nfa.py", "repo/scripts/check_freeze_g0.py",
               "repo/tests/test_cohort.py"]
    hashes = {path: _file_sha256(root / path) for path in sorted(set(inputs + derived))}
    commit, git_state = _git_head(root / "repo")
    payload = dict(schema_version=1, freeze_id="G0-2026-09-26",
                   git_commit_at_freeze=commit, git_state=git_state,
                   git_note="No initial commit exists yet; content SHA256 below is the effective freeze. Refresh git_commit_at_freeze after the first reviewed commit." if commit is None else None,
                   x1_targets="deferred_by_team_decision; no preregistered X1 targets in G0",
                   source_of_truth="repo/results/cohort_freeze.json",
                   cohort_component_sha256=cohort["component_sha256"],
                   chemical_split_unchanged_sha256=original_split_hash,
                   split_files=list(SPLIT_FILES),
                   files=hashes,
                   hash_rule="payload_sha256 is SHA256 of canonical JSON excluding payload_sha256; freeze_g0.sha256 is the complete JSON file SHA256")
    manifest = {**payload, "payload_sha256": _digest(payload)}
    manifest_path = results / "freeze_g0.json"
    _write_json(manifest_path, manifest)
    (results / "freeze_g0.sha256").write_text(
        f"{_file_sha256(manifest_path)}  freeze_g0.json\n", encoding="ascii")
    return dict(n_spids=len(cohort["cohort"]), n_chemicals=cohort["summary"]["n_dtxsid"],
                n_dose_levels=dose["n_levels"], n_plate_dates=plate["n_plate_dates"],
                n_family_eligible=family["n_eligible_chemicals"],
                n_family_excluded=family["n_excluded_chemicals"],
                git_state=git_state, manifest_sha256=_file_sha256(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))


if __name__ == "__main__":
    main()
