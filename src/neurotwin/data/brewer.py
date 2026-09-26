"""Read Brewer hippocampal MEA MATLAB v7.3 tables without MATLAB.

The MATLAB tables use MCOS objects. Their variable arrays and categorical labels
are accessible in ``#subsystem#/MCOS``; the spike trains are HDF5 references in
``#refs#``. We validate this particular layout before decoding it. FIDs are
condition-local table ordinals, because the six HFS cells do not identify which
of the nine physical FIDs they represent.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

SAMPLE_RATE_HZ = 25_000
N_SAMPLES = 7_500_000
DURATION_S = N_SAMPLES / SAMPLE_RATE_HZ
SUBREGIONS = {"EC", "DG", "CA3", "CA1"}
CONDITIONS = ("NoStim", "HFS5", "HFS40")
TUNNELS = {
    *(f"{letter}6-{letter}7" for letter in "ABCDE"),
    *(f"F{i}-G{i}" for i in range(1, 6)),
    *(f"{letter}6-{letter}7" for letter in "HJKLM"),
    *(f"F{i}-G{i}" for i in range(8, 13)),
}


def _matlab_class(obj: h5py.Dataset) -> bytes:
    return obj.attrs.get("MATLAB_class", b"")


def _categorical_strings(obj: h5py.Dataset, expected_rows: int) -> list[str]:
    """Decode MCOS categorical payload: header, UTF-16 lengths, packed text."""
    raw = np.asarray(obj[()]).ravel()
    if obj.dtype != np.dtype("uint64") or len(raw) < 4:
        raise ValueError(f"Unexpected categorical payload: {obj.name}")
    n = int(raw[2])
    if n != expected_rows or int(raw[3]) != 1:
        raise ValueError(f"Categorical row count differs at {obj.name}")
    lengths = raw[4 : 4 + n].astype(int)
    if len(lengths) != n or np.any(lengths <= 0):
        raise ValueError(f"Invalid categorical lengths at {obj.name}")
    text = raw[4 + n :].astype("<u8", copy=False).tobytes().decode("utf-16le")
    if sum(lengths) > len(text):
        raise ValueError(f"Truncated categorical text at {obj.name}")
    starts = np.cumsum(np.r_[0, lengths])
    return [text[int(a) : int(b)] for a, b in zip(starts[:-1], starts[1:])]


def _matlab_chars(obj: h5py.Dataset) -> str:
    if _matlab_class(obj) != b"char":
        raise ValueError(f"Expected MATLAB char: {obj.name}")
    return "".join(chr(int(x)) for x in obj[()].flat)


def _tables(path: Path, expected_columns: tuple[str, ...]):
    """Yield each condition-local table with its categorical labels and columns."""
    with h5py.File(path) as file:
        roots = [value for key, value in file.items() if not key.startswith("#")]
        if len(roots) != 1 or roots[0].ndim != 2 or roots[0].shape[1] != 1:
            raise ValueError(f"Unexpected MATLAB table root: {path}")
        root = roots[0]
        mcos = file["#subsystem#"]["MCOS"]
        expected_fids = 9 if path.name.startswith("NoStim") else 6
        if root.shape[0] != expected_fids or mcos.shape[1] != 9 * expected_fids + 4:
            raise ValueError(f"Unexpected FID/MCOS count: {path}")
        for ordinal, ref in enumerate(root[:, 0], start=1):
            table = file[ref]
            code = table[()].ravel().tolist()
            if _matlab_class(table) != b"table" or len(code) != 6 or code[4] != 1 + 3 * (ordinal - 1):
                raise ValueError(f"Unexpected MCOS table order: {table.name}")
            base = 9 * (ordinal - 1)
            labels_a = file[mcos[0, 2 + base]]
            labels_b = file[mcos[0, 3 + base]]
            payload = file[mcos[0, 4 + base]]
            names_obj = file[mcos[0, 9 + base]]
            names = tuple(_matlab_chars(file[r]) for r in names_obj[:, 0])
            if names != expected_columns or payload.shape != (len(names), 1):
                raise ValueError(f"Unexpected columns in {path}: {names}")
            cols = {name: file[payload[i, 0]] for i, name in enumerate(names)}
            rows = cols[names[2]].shape[-1]
            label_a = _categorical_strings(labels_a, rows)
            label_b = _categorical_strings(labels_b, rows)
            for name in names[3:]:
                if cols[name].shape != (1, rows):
                    raise ValueError(f"Column length differs: {cols[name].name}")
            yield file, ordinal, label_a, label_b, cols


def _spike_times(obj: h5py.Dataset) -> list[float]:
    if _matlab_class(obj) != b"logical" or obj.shape != (N_SAMPLES, 1):
        raise ValueError(f"Expected {N_SAMPLES}-sample logical train at {obj.name}")
    return (np.flatnonzero(obj[:, 0]) / SAMPLE_RATE_HZ).tolist()


def _cell_items(file: h5py.File, column: h5py.Dataset, row: int) -> list[h5py.Dataset]:
    cell = file[column[0, row]]
    if _matlab_class(cell) == b"canonical empty":
        return []
    if _matlab_class(cell) != b"cell":
        raise ValueError(f"Unexpected nested cell: {cell.name}")
    return [file[ref] for ref in cell[()].flat]


def _orientation(regions: list[str], electrodes: list[str]) -> str:
    # EC is in the upper left in both layouts; DG is upper right (clockwise)
    # or lower left (counterclockwise). The first DG well disambiguates them.
    dg = [electrode for region, electrode in zip(regions, electrodes) if region == "DG"]
    if not dg:
        raise ValueError("No DG wells to infer array orientation")
    if all(e[0] in "ABCDE" and int(e[1:]) >= 8 for e in dg):
        return "clockwise"
    if all(e[0] in "HJKLM" and int(e[1:]) <= 5 for e in dg):
        return "counterclockwise"
    raise ValueError(f"DG positions do not identify one orientation: {dg}")


def _tunnel(pair: str) -> str:
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid electrode pair: {pair}")
    tunnel = "-".join(parts)
    if tunnel not in TUNNELS:
        tunnel = "-".join(reversed(parts))
    if tunnel not in TUNNELS:
        raise ValueError(f"Unknown tunnel: {pair}")
    return tunnel


def read_wells(path: Path, condition: str) -> tuple[list[dict], dict[int, str]]:
    rows = []
    orientations = {}
    for file, fid, regions, electrodes, cols in _tables(
        path, ("Subregion", "Electrode", "regi", "SpikeTrain")
    ):
        if len(regions) != 76 or set(regions) != SUBREGIONS or len(set(electrodes)) != 76:
            raise ValueError(f"Invalid well labels in {path}, FID {fid}")
        regi = np.asarray(cols["regi"][()]).ravel()
        if len(regi) != 76 or set(regi) != {1, 2, 3, 4}:
            raise ValueError(f"Invalid well region IDs in {path}, FID {fid}")
        orientations[fid] = _orientation(regions, electrodes)
        for i, (region, electrode) in enumerate(zip(regions, electrodes)):
            if int(regi[i]) != ("EC", "DG", "CA3", "CA1").index(region) + 1:
                raise ValueError(f"Region ID mismatch in {path}, FID {fid}")
            times = _spike_times(file[cols["SpikeTrain"][0, i]])
            rows.append(dict(condition=condition, fid=fid, subregion=region,
                             electrode=electrode, n_spikes=len(times), spike_times_s=times))
    return rows, orientations


def read_axons(path: Path, condition: str) -> list[dict]:
    rows = []
    names = ("Subregion", "ElectrodePairs", "regi", "chani", "up_ff",
             "down_ff", "up_fb", "down_fb", "ff_cdt", "fb_cdt")
    for file, fid, regions, pairs, cols in _tables(path, names):
        if len(pairs) != 20 or len({_tunnel(pair) for pair in pairs}) != 20:
            raise ValueError(f"Expected 20 distinct tunnels in {path}, FID {fid}")
        if set(regions) != {"EC-DG", "DG-CA3", "CA3-CA1", "CA1-EC"}:
            raise ValueError(f"Unexpected tunnel subregions in {path}, FID {fid}")
        for i, (region, pair) in enumerate(zip(regions, pairs)):
            for direction, upstream, downstream, conduction in (
                ("ff", "up_ff", "down_ff", "ff_cdt"),
                ("fb", "up_fb", "down_fb", "fb_cdt"),
            ):
                up = _cell_items(file, cols[upstream], i)
                down = _cell_items(file, cols[downstream], i)
                delays = _cell_items(file, cols[conduction], i)
                if not (len(up) == len(down) == len(delays)):
                    raise ValueError(f"Axon count mismatch in {path}, FID {fid}, {pair}")
                for axon_idx, (u, d, delay) in enumerate(zip(up, down, delays), start=1):
                    if _matlab_class(delay) != b"double" or delay.size != 1:
                        raise ValueError(f"Invalid conduction delay at {delay.name}")
                    conduction_ms = float(delay[()].flat[0])
                    if not np.isfinite(conduction_ms) or conduction_ms <= 0:
                        raise ValueError(f"Invalid conduction delay at {delay.name}")
                    up_times, down_times = _spike_times(u), _spike_times(d)
                    rows.append(dict(condition=condition, fid=fid, tunnel=_tunnel(pair),
                                     electrode_pair_upstream_first=pair,
                                     subregion_pair=region, axon_idx=axon_idx,
                                     direction=direction, conduction_ms=conduction_ms,
                                     upstream_spike_times_s=up_times,
                                     downstream_spike_times_s=down_times))
    return rows


def _write_table(rows: list[dict], path: Path) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # JSON strings in npz avoid pickle and preserve nested lists.
        np.savez_compressed(path.with_suffix(".npz"),
                            rows=np.asarray([json.dumps(row) for row in rows]))
        return str(path.with_suffix(".npz"))
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return str(path)


def process(raw_dir: Path, out_dir: Path, audit_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    wells, axons, orientations = [], [], {}
    for condition in CONDITIONS:
        wr, ori = read_wells(raw_dir / f"{condition}WellSpikes.mat", condition)
        ar = read_axons(raw_dir / f"{condition}SortedAxons.mat", condition)
        wells.extend(wr)
        axons.extend(ar)
        orientations[condition] = ori
    well_counts = defaultdict(lambda: Counter())
    axon_counts = defaultdict(lambda: Counter())
    conduction = defaultdict(list)
    for row in wells:
        well_counts[(row["condition"], row["fid"])][row["subregion"]] += row["n_spikes"]
    for row in axons:
        key = (row["condition"], row["fid"])
        axon_counts[key][row["direction"]] += 1
        conduction[key].append(row["conduction_ms"])
    recordings = []
    for condition in CONDITIONS:
        n = 9 if condition == "NoStim" else 6
        for fid in range(1, n + 1):
            key = (condition, fid)
            counts = well_counts[key]
            recordings.append(dict(condition=condition, fid=fid,
                                   orientation=orientations[condition][fid],
                                   **{f"n_spikes_{r}": counts[r] for r in ("EC", "DG", "CA3", "CA1")}))
    paths = dict(wells=_write_table(wells, out_dir / "wells.parquet"),
                 axons=_write_table(axons, out_dir / "axons.parquet"))
    with (out_dir / "recordings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(recordings[0]))
        writer.writeheader()
        writer.writerows(recordings)
    paths["recordings"] = str(out_dir / "recordings.csv")
    details = []
    for record in recordings:
        key = (record["condition"], record["fid"])
        details.append({**record, "n_axons_ff": axon_counts[key]["ff"],
                        "n_axons_fb": axon_counts[key]["fb"],
                        "median_conduction_ms": float(np.median(conduction[key])) if conduction[key] else None})
    report = dict(source="https://zenodo.org/records/10257483", license="CC0-1.0",
                  sample_rate_hz=SAMPLE_RATE_HZ, n_samples=N_SAMPLES,
                  duration_s=DURATION_S, tunnels_per_recording=20,
                  fid_semantics="1-based ordinal within each condition; physical HFS FID mapping is unverified",
                  n_recordings=len(recordings), n_well_rows=len(wells), n_axons=len(axons),
                  n_axons_ff=sum(r["direction"] == "ff" for r in axons),
                  n_axons_fb=sum(r["direction"] == "fb" for r in axons),
                  median_conduction_ms=float(np.median([r["conduction_ms"] for r in axons])),
                  outputs=paths, recordings=details)
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/brewer_hfs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/brewer"))
    parser.add_argument("--audit", type=Path, default=Path("repo/results/data_audit_brewer.json"))
    args = parser.parse_args()
    report = process(args.raw_dir, args.out_dir, args.audit)
    print(json.dumps({key: report[key] for key in (
        "n_recordings", "n_well_rows", "n_axons", "n_axons_ff", "n_axons_fb", "median_conduction_ms"
    )}, indent=2))


if __name__ == "__main__":
    main()
