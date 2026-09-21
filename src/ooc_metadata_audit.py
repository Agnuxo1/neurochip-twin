"""Reproducible metadata audit for the public Organ-on-a-Chip dataset.

The Zenodo record publishes a small ``OOC_datasheet.xlsx`` alongside a much
larger image archive.  This module intentionally reads only the spreadsheet:
it records data coverage and missingness for domain-shift planning, but it
does not train a model or present the dataset's good/bad *sample-quality*
labels as toxicity or biological-response labels.

The XLSX reader uses only the Python standard library so the audit remains
usable in a clean Kaggle-style environment without adding an Excel engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZipFile
from xml.etree import ElementTree

SOURCE_RECORD_URL = "https://zenodo.org/records/10203721"
METADATA_URL = (
    "https://zenodo.org/records/10203721/files/"
    "OOC_datasheet.xlsx?download=1"
)
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_CELL_REF = re.compile(r"([A-Z]+)")


def _column_number(reference: str) -> int:
    """Return a zero-based column number from an A1-style cell reference."""

    match = _CELL_REF.match(reference.upper())
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference!r}")
    number = 0
    for character in match.group(1):
        number = number * 26 + ord(character) - ord("A") + 1
    return number - 1


def _shared_strings(root: ElementTree.Element) -> list[str]:
    return [
        "".join(text.text or "" for text in item.iter(f"{{{_NS}}}t"))
        for item in root.findall(f"{{{_NS}}}si")
    ]


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{{{_NS}}}t"))
    value = cell.find(f"{{{_NS}}}v")
    if value is None or value.text is None:
        return ""
    raw = value.text
    if kind == "s":
        return shared[int(raw)]
    if kind == "b":
        return "true" if raw == "1" else "false"
    return raw


def read_xlsx_table(path: str | Path) -> list[dict[str, str]]:
    """Read the first worksheet as a list of row dictionaries.

    The parser covers the shared-string, inline-string, numeric and boolean
    cell forms used by the published datasheet.  Empty cells are retained so
    missingness is not accidentally converted into a shifted column.
    """

    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = _shared_strings(
                ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            )
        sheet_name = sorted(
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )[0]
        root = ElementTree.fromstring(archive.read(sheet_name))

    rows: list[list[str]] = []
    for row in root.findall(f".//{{{_NS}}}sheetData/{{{_NS}}}row"):
        cells = {
            _column_number(cell.attrib["r"]): _cell_value(cell, shared)
            for cell in row.findall(f"{{{_NS}}}c")
        }
        width = max(cells, default=-1) + 1
        rows.append([cells.get(index, "") for index in range(width)])

    if not rows:
        return []
    headers = [value.strip() for value in rows[0]]
    if not all(headers):
        raise ValueError("The first XLSX row must contain non-empty headers")
    if len(set(headers)) != len(headers):
        raise ValueError("The XLSX header row contains duplicate column names")
    return [
        {
            header: values[index] if index < len(values) else ""
            for index, header in enumerate(headers)
        }
        for values in rows[1:]
    ]


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _number(value: Any) -> float | None:
    if _missing(value):
        return None
    cleaned = str(value).strip().replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned)
        if match is None:
            return None
        number = float(match.group(0))
    return number if math.isfinite(number) else None


def _display_number(number: float) -> int | float:
    return int(number) if number.is_integer() else number


def _numeric_stats(rows: list[Mapping[str, Any]], column: str) -> dict[str, int | float]:
    values = [number for row in rows if (number := _number(row.get(column))) is not None]
    missing = sum(_missing(row.get(column)) for row in rows)
    if not values:
        return {
            "present": 0,
            "missing": missing,
            "missing_fraction": missing / len(rows) if rows else 0.0,
        }
    return {
        "present": len(values),
        "missing": missing,
        "missing_fraction": missing / len(rows) if rows else 0.0,
        "unique": len(set(values)),
        "min": _display_number(min(values)),
        "max": _display_number(max(values)),
    }


def _quality_label(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "1.0", "good"}:
        return "good"
    if raw in {"2", "2.0", "bad"}:
        return "bad"
    return raw or "missing"


def summarize_ooc_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize coverage and missingness without fitting a predictive model."""

    materialized = [dict(row) for row in rows]
    columns = sorted({key for row in materialized for key in row})
    nonempty_rows = sum(any(not _missing(value) for value in row.values()) for row in materialized)
    missingness = {
        column: {
            "missing": sum(_missing(row.get(column)) for row in materialized),
            "missing_fraction": (
                sum(_missing(row.get(column)) for row in materialized)
                / len(materialized)
                if materialized
                else 0.0
            ),
        }
        for column in columns
    }
    image_ids = [
        str(row["imageID"]).strip()
        for row in materialized
        if not _missing(row.get("imageID"))
    ]
    quality_raw = Counter(
        str(row.get("Decision 1/2 (good/bad)", "")).strip()
        or "missing"
        for row in materialized
    )
    quality_mapped = Counter(
        _quality_label(row.get("Decision 1/2 (good/bad)")) for row in materialized
    )
    cell_types = sorted(
        {
            str(row["cell type"]).strip()
            for row in materialized
            if not _missing(row.get("cell type"))
        }
    )
    return {
        "schema_version": "ooc_metadata_audit_v1",
        "dataset": "Organ-on-a-Chip (OOC) Image Dataset",
        "source_record_url": SOURCE_RECORD_URL,
        "metadata_url": METADATA_URL,
        "rows": len(materialized),
        "nonempty_rows": nonempty_rows,
        "fully_blank_rows": len(materialized) - nonempty_rows,
        "columns": columns,
        "unique_nonempty_image_ids": len(set(image_ids)),
        "duplicate_nonempty_image_ids": len(image_ids) - len(set(image_ids)),
        "missingness": missingness,
        "cell_types": cell_types,
        "cell_type_counts": dict(
            sorted(
                Counter(
                    str(row["cell type"]).strip()
                    for row in materialized
                    if not _missing(row.get("cell type"))
                ).items()
            )
        ),
        "quality_counts_raw": dict(sorted(quality_raw.items())),
        "quality_counts_mapped": dict(sorted(quality_mapped.items())),
        "quality_mapping_note": (
            "Convenience mapping from the sheet header: 1=good and 2=bad; "
            "these are expert-assessed sample-quality labels, not toxicity "
            "or treatment-response labels."
        ),
        "numeric_fields": {
            "seeding_density_cells_per_ml": _numeric_stats(
                materialized, "seeding density, cells/ml"
            ),
            "time_after_seeding_hours": _numeric_stats(
                materialized, "time after seeding, h"
            ),
            "day": _numeric_stats(materialized, "day"),
            "flow_rate": _numeric_stats(materialized, "flow rate"),
        },
        "split_column_present": any(
            str(column).strip().lower() in {"split", "train/val/test", "set"}
            for column in columns
        ),
        "interpretation": (
            "Metadata/domain audit only. It supports planning a real OOC "
            "domain-shift evaluation; it is not biological validation, "
            "toxicity validation, or clinical performance evidence."
        ),
    }


def audit_ooc_metadata(path: str | Path) -> dict[str, Any]:
    """Read and summarize a downloaded OOC datasheet with provenance."""

    source = Path(path)
    summary = summarize_ooc_rows(read_xlsx_table(source))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    summary["input_file"] = source.name
    summary["input_size_bytes"] = source.stat().st_size
    summary["input_sha256"] = digest
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_ooc_metadata(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
