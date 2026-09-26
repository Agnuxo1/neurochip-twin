"""Extract the 17 named MEA analog features from Brewer subregion spike lists."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))

from neurotwin.features.mea import FEATURES, extract_features  # noqa: E402


def run(wells_path: Path, output_path: Path) -> dict:
    wells = pq.read_table(wells_path).to_pylist()
    groups = defaultdict(dict)
    for row in wells:
        key = (row["condition"], int(row["fid"]), row["subregion"])
        if row["electrode"] in groups[key]:
            raise ValueError(f"Duplicate electrode in group {key}")
        groups[key][row["electrode"]] = row["spike_times_s"]
    rows = []
    for (condition, fid, region), trains in sorted(groups.items()):
        features = extract_features(trains, duration_s=300.0)
        rows.append({"condition": condition, "fid": fid, "subregion": region,
                     "duration_s": 300.0, "n_electrodes": len(trains),
                     "n_spikes": sum(len(t) for t in trains.values()), **features})
    if len(rows) != 84 or len(wells) != 1596 or any(r["n_electrodes"] != 19 for r in rows):
        raise ValueError("Unexpected Brewer recording/subregion geometry")
    result = {"source": "data/processed/brewer/wells.parquet",
              "status": "cross-platform feature analogs; EPA fidelity not established",
              "feature_order": FEATURES, "n_recordings": 21, "n_groups": len(rows),
              "n_electrode_trains": len(wells), "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"n_groups": len(rows), "n_electrode_trains": len(wells),
            "n_features": len(FEATURES), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wells", type=Path, default=ROOT / "data/processed/brewer/wells.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "repo/results/brewer_features.json")
    args = parser.parse_args()
    print(json.dumps(run(args.wells, args.output), indent=2))


if __name__ == "__main__":
    main()
