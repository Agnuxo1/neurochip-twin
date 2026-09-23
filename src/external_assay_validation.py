"""Grouped external assay audit using the U.S. EPA public-domain DRG workbook.

This audit evaluates tabular assay response models on real acute rat DRG data.
It does not validate the synthetic microscopy pipeline or organ-on-chip
generalization. The workbook is downloaded by the user/reviewer and is never
bundled with this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ooc_metadata_audit import _cell_value, _column_number, _shared_strings


SOURCE_CATALOG_URL = (
    "https://catalog.data.gov/dataset/"
    "kodavantip-acute_hsab_aop_neurotox_science-hub-data-090319"
)
SOURCE_DOWNLOAD_URL = (
    "https://pasteur.epa.gov/uploads/10.23719/1504280/"
    "KodavantiP_Acute_HSAB_AOP_Neurotox_Science%20Hub.xlsx"
)
SOURCE_PAPER_DOI = "10.1016/j.tiv.2020.104989"
LICENSE_NOTE = (
    "U.S. EPA states its data are public domain unless otherwise specified; "
    "the source record links the applicable EPA data disclaimer."
)
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _read_grids(path: Path) -> dict[str, list[list[str]]]:
    """Read all worksheets with the standard library, preserving empty cells."""
    with ZipFile(path) as archive:
        shared_path = "xl/sharedStrings.xml"
        shared = (
            _shared_strings(ET.fromstring(archive.read(shared_path)))
            if shared_path in archive.namelist()
            else []
        )
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationships = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationship_root
        }
        grids = {}
        for sheet in workbook.findall(f".//{{{_NS}}}sheet"):
            relation_id = sheet.attrib[f"{{{_REL_NS}}}id"]
            target = relationships[relation_id].lstrip("/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            root = ET.fromstring(archive.read(sheet_path))
            rows = []
            for row in root.findall(f".//{{{_NS}}}sheetData/{{{_NS}}}row"):
                cells = {
                    _column_number(cell.attrib["r"]): _cell_value(cell, shared)
                    for cell in row.findall(f"{{{_NS}}}c")
                }
                rows.append(
                    [cells.get(index, "") for index in range(max(cells, default=-1) + 1)]
                )
            grids[sheet.attrib["name"]] = rows
    return grids


def _melt_dose_blocks(
    grid: Sequence[Sequence[str]], assay: str
) -> pd.DataFrame:
    """Tidy the source workbook's chemical × concentration matrix layout."""
    if len(grid) < 4:
        return pd.DataFrame(columns=["date", "chemical", "dose_uM", "target"])
    observations = []
    for start, chemical in enumerate(grid[1]):
        chemical = chemical.strip()
        if not chemical:
            continue
        date_column = start - 1
        for row in grid[3:]:
            experiment_date = row[date_column].strip() if 0 <= date_column < len(row) else ""
            if not experiment_date:
                continue
            for offset in range(6):
                column = start + offset
                if column >= len(grid[2]) or column >= len(row):
                    continue
                dose_text = grid[2][column].strip()
                value_text = row[column].strip()
                if not dose_text or not value_text:
                    continue
                dose = 0.0 if dose_text.lower() in {"ctrl", "control"} else float(dose_text)
                observations.append(
                    {
                        "assay": assay,
                        "date": experiment_date,
                        "chemical": chemical,
                        "dose_uM": dose,
                        "target": float(value_text),
                    }
                )
    return pd.DataFrame.from_records(observations)


def _read_row_table(grid: Sequence[Sequence[str]], assay: str) -> pd.DataFrame:
    if not grid:
        return pd.DataFrame()
    headers = [value.strip() for value in grid[0]]
    rows = []
    for source_row in grid[1:]:
        row = {
            headers[index]: source_row[index].strip()
            for index in range(min(len(headers), len(source_row)))
            if headers[index]
        }
        if not any(row.values()):
            continue
        if assay == "LDH":
            rows.append(
                {
                    "assay": assay,
                    "date": row["Date"],
                    "chemical": row["Chemcial"],
                    "dose_uM": float(row["Dose (uM)"].replace(",", "")),
                    "target": float(row["LDH"]),
                }
            )
        else:
            rows.append(
                {
                    "assay": assay,
                    "date": row["ExpDate"],
                    "chemical": row["Chem"],
                    "dose_uM": float(row["Dose (uM)"].replace(",", "")),
                    "pre_rate": float(row["meanFR_pre"]),
                    "target": float(row["pcntBL"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def load_epa_assays(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load tidy LDH, neurite, neuron-count and MEA endpoints from the EPA file."""
    grids = _read_grids(Path(path))
    required = {"LDH", "NLPN", "NPF", "MEA"}
    missing = required - set(grids)
    if missing:
        raise ValueError(f"EPA workbook is missing worksheets: {sorted(missing)}")
    return {
        "LDH": _read_row_table(grids["LDH"], "LDH"),
        "NLPN": _melt_dose_blocks(grids["NLPN"], "NLPN"),
        "NPF": _melt_dose_blocks(grids["NPF"], "NPF"),
        "MEA": _read_row_table(grids["MEA"], "MEA"),
    }


def make_group_splits(
    frame: pd.DataFrame, split_by: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create leakage-resistant batch or treatment-compound holdout folds.

    In the MEA workbook, the common vehicle-control rows use chemical ``CTR``.
    They are retained in training for each held-out treatment compound and are
    not counted as a test fold. Other assays encode vehicle controls at dose 0
    within each chemical; they are only evaluated by batch holdout.
    """
    if split_by == "date":
        groups = frame["date"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=min(10, len(np.unique(groups))))
        return list(splitter.split(frame, frame["target"], groups))
    if split_by != "chemical":
        raise ValueError("split_by must be 'date' or 'chemical'")
    if "pre_rate" not in frame:
        raise ValueError("Chemical holdout is restricted to the MEA assay")
    chemicals = frame["chemical"].astype(str).to_numpy()
    controls = np.array([value.strip().upper() == "CTR" for value in chemicals])
    folds = []
    for chemical in sorted(set(chemicals[~controls])):
        test = np.flatnonzero((chemicals == chemical) & ~controls)
        train = np.flatnonzero(chemicals != chemical)
        if len(test) and len(train):
            folds.append((train, test))
    return folds


def _pipeline(features: Sequence[str], model_kind: str):
    categorical = [name for name in ("chemical",) if name in features]
    numeric = [name for name in features if name not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(("chemical", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocessor = ColumnTransformer(transformers)
    if model_kind == "ridge":
        estimator = Ridge(alpha=10.0)
    elif model_kind == "forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            max_features=0.9,
            random_state=42,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    return make_pipeline(preprocessor, estimator)


def _evaluate(
    frame: pd.DataFrame, features: Sequence[str], split_by: str, model_kind: str
) -> dict[str, object]:
    y = frame["target"].to_numpy(float)
    predictions = np.full(len(frame), np.nan)
    folds = make_group_splits(frame, split_by)
    fold_metrics = []
    for train, test in folds:
        model = _pipeline(features, model_kind)
        model.fit(frame.iloc[train][list(features)], y[train])
        prediction = model.predict(frame.iloc[test][list(features)])
        predictions[test] = prediction
        fold_metrics.append(
            {
                "group": str(frame.iloc[test][split_by].iloc[0]),
                "n": int(len(test)),
                "mae": float(mean_absolute_error(y[test], prediction)),
                "r2": float(r2_score(y[test], prediction)) if len(test) > 1 else None,
            }
        )
    scored = np.isfinite(predictions)
    if not scored.any():
        raise ValueError(f"No held-out rows were evaluated for {split_by}")
    group_maes = np.asarray([fold["mae"] for fold in fold_metrics], dtype=float)
    return {
        "split": split_by,
        "model": model_kind,
        "features": list(features),
        "n_evaluated": int(scored.sum()),
        "n_groups": len(folds),
        "mae": float(mean_absolute_error(y[scored], predictions[scored])),
        "r2_oof": float(r2_score(y[scored], predictions[scored])),
        "group_macro_mae": float(group_maes.mean()),
        "group_macro_mae_bootstrap_95ci": _bootstrap_mean_ci(group_maes),
        "fold_metrics": fold_metrics,
    }


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 42, resamples: int = 10_000
) -> list[float]:
    """Percentile CI for the mean of independent held-out-group metrics."""
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or not len(observations):
        raise ValueError("Bootstrap CI requires a non-empty one-dimensional sample")
    if not np.isfinite(observations).all() or resamples < 1:
        raise ValueError("Bootstrap observations must be finite and resamples positive")
    rng = np.random.default_rng(seed)
    draws = rng.choice(observations, size=(resamples, len(observations)), replace=True)
    lower, upper = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return [float(lower), float(upper)]


def _paired_group_mae_difference(
    forest: dict[str, object], ridge: dict[str, object]
) -> dict[str, object]:
    """Compare models on identical held-out groups; negative favors the forest."""
    forest_folds = {fold["group"]: fold["mae"] for fold in forest["fold_metrics"]}
    ridge_folds = {fold["group"]: fold["mae"] for fold in ridge["fold_metrics"]}
    if not forest_folds or forest_folds.keys() != ridge_folds.keys():
        raise ValueError("Paired model comparison requires identical held-out groups")
    differences = np.asarray(
        [forest_folds[group] - ridge_folds[group] for group in sorted(forest_folds)],
        dtype=float,
    )
    return {
        "estimand": "equal-weight mean of per-treatment-label MAE differences",
        "direction": "negative favors Random Forest",
        "n_groups": int(len(differences)),
        "forest_minus_ridge_mae": float(differences.mean()),
        "bootstrap_95ci": _bootstrap_mean_ci(differences),
    }


def evaluate_epa_assays(path: str | Path) -> dict[str, object]:
    """Run fixed, non-tuned assay baselines under grouped cross-validation."""
    input_path = Path(path)
    tables = load_epa_assays(input_path)
    results: dict[str, object] = {}
    for assay, frame in tables.items():
        context_features = ["chemical", "dose_uM"]
        if "pre_rate" in frame:
            context_features.append("pre_rate")
        results[assay] = {
            "n_observations": int(len(frame)),
            "n_dates": int(frame["date"].nunique()),
            "n_source_chemical_labels": int(frame["chemical"].nunique()),
            "dose_levels_uM": sorted(float(value) for value in frame["dose_uM"].unique()),
            "date_holdout_dose_ridge": _evaluate(
                frame, ["dose_uM"], "date", "ridge"
            ),
            "date_holdout_context_ridge": _evaluate(
                frame, context_features, "date", "ridge"
            ),
            "date_holdout_context_forest": _evaluate(
                frame, context_features, "date", "forest"
            ),
        }
        if assay == "MEA":
            unseen_chemical_features = ["dose_uM", "pre_rate"]
            results[assay]["chemical_holdout_dose_pre_ridge"] = _evaluate(
                frame, unseen_chemical_features, "chemical", "ridge"
            )
            results[assay]["chemical_holdout_dose_pre_forest"] = _evaluate(
                frame, unseen_chemical_features, "chemical", "forest"
            )
            results[assay]["chemical_holdout_paired_mae_comparison"] = (
                _paired_group_mae_difference(
                    results[assay]["chemical_holdout_dose_pre_forest"],
                    results[assay]["chemical_holdout_dose_pre_ridge"],
                )
            )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    return {
        "schema_version": "epa_external_assay_audit_v1",
        "dataset": "U.S. EPA Acute HSAB AOP Neurotox Science Hub workbook",
        "source_catalog_url": SOURCE_CATALOG_URL,
        "source_download_url": SOURCE_DOWNLOAD_URL,
        "source_paper_doi": SOURCE_PAPER_DOI,
        "license_note": LICENSE_NOTE,
        "input_file": input_path.name,
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": digest,
        "protocol": {
            "date_holdout": "GroupKFold by experimental date; preprocessing fit within each training fold",
            "chemical_holdout": "Leave one MEA treatment label out; vehicle CTR rows remain training-only",
            "hyperparameter_tuning": "none; fixed Ridge and RandomForest configurations",
            "models": {
                "Ridge": {"alpha": 10.0},
                "RandomForestRegressor": {
                    "n_estimators": 300,
                    "min_samples_leaf": 4,
                    "max_features": 0.9,
                    "random_state": 42,
                    "n_jobs": 1,
                },
            },
            "paired_bootstrap": {
                "unit": "treatment label",
                "resamples": 10_000,
                "seed": 42,
                "interval": "percentile 95%",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "assays": results,
        "limitations": [
            "Acute embryonic rat DRG culture is not organ-on-chip data.",
            "This tabular audit does not evaluate the synthetic microscopy segmentation or temporal image model.",
            "The MEA workbook label BNB is not defined by its glossary and is preserved as an unresolved source label.",
            "The benchmark is small and heterogeneous; only the 12-label MEA paired MAE comparison has a label-bootstrap interval, and it is exploratory.",
            "No result is clinical performance or evidence of human biological transfer.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Downloaded EPA XLSX workbook")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()
    result = evaluate_epa_assays(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
