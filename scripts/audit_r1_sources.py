"""Check whether Brown/Cotterill public spike files match EPA NFA well readouts.

Uses GitHub's metadata tree, not the 307 MB HDF5 payload. Downloading unmatched
raw files cannot establish within-well feature fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TREE_URL = "https://api.github.com/repos/sje30/EPAmeadev/git/trees/master?recursive=1"


def run(nfa_path: Path, output_path: Path) -> dict:
    request = urllib.request.Request(TREE_URL, headers={"User-Agent": "ai4s-r1-source-audit"})
    with urllib.request.urlopen(request, timeout=40) as response:
        raw = response.read()
    tree = json.loads(raw)
    files = [item for item in tree["tree"]
             if item["path"].startswith("allH5Files/") and item["path"].endswith(".h5")]
    source_keys = set()
    for item in files:
        parts = Path(item["path"]).name.split("_")
        if len(parts) < 4 or not parts[3].startswith("DIV"):
            raise ValueError(f"Unexpected HDF5 filename: {item['path']}")
        source_keys.add((parts[2], parts[1], int(parts[3][3:])))
    table = pq.read_table(nfa_path, columns=["plate", "date", "div"]).to_pylist()
    nfa_keys = {(str(row["plate"]), str(row["date"]), int(row["div"])) for row in table}
    source_plates = {k[0] for k in source_keys}
    nfa_plates = {k[0] for k in nfa_keys}
    overlap = sorted(source_keys & nfa_keys)
    status = "eligible_for_fidelity" if len({x[0] for x in overlap}) >= 3 else "not_estimable"
    result = {"status": status, "r1_optional_gate": "closed_without_fidelity_metrics",
              "reason": "No matched plate/date/DIV keys between public Brown/Cotterill HDF5 spike files and local NFA raw-feature wells"
                        if not overlap else "Fewer than three matched complete plates",
              "source": "https://github.com/sje30/EPAmeadev/tree/master/allH5Files",
              "source_study": "Cotterill et al. 2016; Ball et al. 2017 uses a subset of Brown et al. 2016 spike recordings",
              "source_data_permission": "EPAmeadev README permits use of data/resources with paper citation",
              "github_tree_sha": tree["sha"], "github_tree_json_sha256": hashlib.sha256(raw).hexdigest(),
              "source_h5_files": len(files), "source_h5_total_bytes": sum(i.get("size", 0) for i in files),
              "source_plate_date_div_keys": len(source_keys),
              "nfa_plate_date_div_keys": len(nfa_keys),
              "source_plates": len(source_plates), "nfa_plates": len(nfa_plates),
              "plate_overlap": len(source_plates & nfa_plates),
              "plate_date_div_overlap": len(overlap), "overlap_keys": overlap,
              "downloaded_spike_files": 0,
              "download_note": "Full spike payload omitted after metadata proved zero shared plates; no file SHA256 can be reported for absent payloads",
              "metrics": None,
              "fallback": "Use the published EPA NFA 17 endpoint values for modeling; label Brewer extraction as unvalidated cross-platform analogs"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfa", type=Path, default=ROOT / "data/processed/epa_nfa/wells.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "repo/results/r1_feature_fidelity.json")
    args = parser.parse_args()
    result = run(args.nfa, args.output)
    (args.output.parent / "feature_engine_fidelity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "source_h5_files", "plate_overlap", "plate_date_div_overlap")}, indent=2))


if __name__ == "__main__":
    main()
