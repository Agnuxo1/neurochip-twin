# Brewer 4-compartment MEA ingestion

This module converts the [Brewer hippocampal MEA dataset](https://zenodo.org/records/10257483)
(Dryad DOI [10.5061/dryad.7h44j1013](https://doi.org/10.5061/dryad.7h44j1013), CC0)
to tidy tables. It uses 9 unstimulated and 6 recordings for each HFS condition.
Each logical spike train has 7,500,000 samples at 25 kHz (300 seconds).

## Reproduce

From the workspace root, with `numpy`, `h5py`, `pytest`, and optionally
`pyarrow` installed:

```powershell
python repo/scripts/download_brewer.py --verify-only
$env:PYTHONPATH = "repo/src"
python -m neurotwin.data.brewer
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest repo/tests/test_brewer.py -q
```

The downloader uses Zenodo's record API and verifies every MAT file against the
bundled `repo/scripts/brewer_sha256sums.txt` manifest. Omit `--verify-only` to fetch
missing files. The converter writes `data/processed/brewer/wells.parquet`,
`axons.parquet`, `recordings.csv`, and `repo/results/data_audit_brewer.json`.
If `pyarrow` is unavailable, it writes `.npz` tables with JSON strings under
`rows`; load these with `numpy.load(..., allow_pickle=False)`.

## Reader and schema

The v7.3 MAT files store MATLAB table objects using MCOS. The reader validates
the 9-object-per-table layout, decodes UTF-16 categorical labels from
`#subsystem#/MCOS`, and follows HDF5 references in `#refs#` for logical spike
trains. It does not need MATLAB or the much larger Wave_Clus ZIP archives.
Malformed layouts, labels, row counts, and spike-train shapes raise errors.

`wells` has one row per well electrode and recording, including `n_spikes` and
the complete `spike_times_s` list. `axons` has one row per sorted axon and
direction, including its 20-tunnel identity, conduction time in milliseconds,
and upstream/downstream spike times in seconds. The `electrode_pair_upstream_first`
column preserves the source electrode order; `tunnel` uses the canonical pair
from the dataset README. `recordings.csv` includes inferred orientation and
spike counts by subregion. The JSON audit includes axon direction counts and
median conduction times for each recording.

**FID caveat:** `fid` is a 1-based table position *within each condition*.
The HFS MAT files hold six cells, while the associated Wave_Clus ZIP names
refer to a subset of physical FIDs. No source field in these MAT tables maps
those six cells to the physical IDs. Do not join conditions on `fid` until that
mapping is independently established. Orientation is inferred from the DG
electrodes' position, using the layout stated in the dataset README.
