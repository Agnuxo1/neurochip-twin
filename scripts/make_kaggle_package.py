"""Assemble the CPU-only Kaggle packages for NeuroChip Twin v2. Builds files only: it never uploads.

    python scripts/make_kaggle_package.py              # -> ../_kaggle_pkg/dataset and ../_kaggle_pkg/kernel
    python scripts/make_kaggle_package.py --execute    # also run the notebook locally on CPU against the snapshot
    python scripts/make_kaggle_package.py --exclude site   # leave extra top-level folders out of the snapshot
    python scripts/make_kaggle_package.py --out D:/elsewhere --dataset-id owner/slug --kernel-id owner/slug-demo

Layout produced (default --out is the folder that contains this repository, plus /_kaggle_pkg):
    dataset/                    repository snapshot without .git, outputs, caches or hidden files
    dataset/dataset-metadata.json
    dataset/kaggle_package_manifest.json   SHA-256 and size of every snapshot file
    kernel/neurochip_twin_v2_kaggle.ipynb  copy of notebooks/neurochip_twin_v2_kaggle.ipynb
    kernel/kernel-metadata.json            CPU only: enable_gpu false, enable_internet false
    local_run/                             (--execute) executed notebook, figures and run_report.json

The ids are placeholders. Uploading is a manual decision of the team lead, after review:
    kaggle datasets create -p <out>/dataset --dir-mode zip      # zip mode keeps the sub-folders
    kaggle kernels push -p <out>/kernel
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "neurochip_twin_v2_kaggle.ipynb"
DATASET_TITLE = "NeuroChip Twin v2 code and bundle"
DATASET_ID = "franciscoangulo/neurochip-twin-v2"
KERNEL_ID = "franciscoangulo/neurochip-twin-v2-demo"
KERNEL_TITLE = "NeuroChip Twin v2 demo"
COMPETITION = "ai-4-s-open-innovation-artificial-intelligence-for-life-scien"
LICENCE = "MIT"
MARKER = "kaggle_package_manifest.json"          # written into every dataset/ this script builds

EXCLUDE_NAMES = {".git", "outputs", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache",
                 ".ruff_cache", ".venv", "venv", "neurochip_outputs", "local_run", "dataset-metadata.json", MARKER}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log~"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def excluded(rel: Path, extra: frozenset = frozenset()) -> bool:
    if rel.parts and rel.parts[0] in extra:
        return True
    return any(p in EXCLUDE_NAMES or p.startswith(".") for p in rel.parts) or rel.suffix in EXCLUDE_SUFFIXES


def snapshot_files(repo: Path, extra: frozenset = frozenset()) -> list[Path]:
    out = []
    for root, dirs, files in os.walk(repo):
        rel_root = Path(root).relative_to(repo)
        dirs[:] = sorted(d for d in dirs if not excluded(rel_root / d, extra))
        out += [rel_root / f for f in sorted(files) if not excluded(rel_root / f, extra)]
    return out


def git_commit(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if not ref.startswith("ref:"):
        return ref
    target = repo / ".git" / ref.split(" ", 1)[1]
    if target.is_file():
        return target.read_text(encoding="utf-8").strip()
    packed = repo / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(ref.split(" ", 1)[1]):
                return line.split()[0]
    return None


def fresh_dir(path: Path, marker: str | None) -> None:
    """Recreate an output folder, refusing to wipe one this script did not build."""
    if path.exists():
        ours = marker is None or (path / marker).is_file() or not any(path.iterdir())
        if not ours:
            raise SystemExit(f"refusing to overwrite {path}: it was not built by this script (no {marker})")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build_dataset(out: Path, dataset_id: str, extra: frozenset = frozenset()) -> dict:
    ds = out / "dataset"
    fresh_dir(ds, MARKER)
    files, total = {}, 0
    for rel in snapshot_files(REPO, extra):
        dst = ds / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
        n = dst.stat().st_size
        total += n
        files[rel.as_posix()] = {"sha256": sha256(dst), "bytes": n}
    bundle = json.loads((REPO / "data_bundle" / "manifest.json").read_text(encoding="utf-8"))
    licence_file = next((f for f in ("LICENSE", "LICENSE.md", "LICENSE.txt") if f in files), None)
    subtitle = "CPU demo, trained models and results for few-shot MEA neurotoxicity forecasts"
    description = "\n".join([
        f"# {DATASET_TITLE}",
        "",
        "Snapshot of the NeuroChip Twin v2 repository for the Kaggle AI4S Open Innovation hackathon "
        "(AI + neural organ-on-a-chip). It contains the code (`src/neurotwin`, `demo.py`, `scripts/`, "
        "`tests/`), the compact data bundle (`data_bundle/`: transformed EPA NFA trajectories for "
        f"{bundle['n_chemicals']} chemicals / {bundle['n_wells']} wells and the cross-validation model "
        "weights) and the result JSONs (`results/`).",
        "",
        "Run on CPU without internet: attach this dataset to the notebook "
        f"`{KERNEL_ID}` or run `python demo.py --list`.",
        "",
        "Evidence boundaries: rat cortical cultures on 48-well MEA plates (EPA NFA), not a validated twin of "
        "any commercial chip; compartment scenarios are simulations; not for clinical use.",
        "",
        "Code: MIT. Per-file sources and licences of the data files: `data_bundle/manifest.json`.",
    ])
    meta = {"title": DATASET_TITLE, "id": dataset_id, "subtitle": subtitle, "description": description,
            "licenses": [{"name": LICENCE}]}
    (ds / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest = {
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_repo": str(REPO), "git_commit": git_commit(REPO),
        "excluded": sorted(EXCLUDE_NAMES) + ["hidden files and folders (.*)"] + sorted(EXCLUDE_SUFFIXES)
        + [f"top-level {x}/ (--exclude)" for x in sorted(extra)],
        "licence_file_present": licence_file, "n_files": len(files), "total_bytes": total, "files": files}
    (ds / MARKER).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def build_kernel(out: Path, kernel_id: str, dataset_id: str) -> Path:
    if not NOTEBOOK.is_file():
        raise SystemExit(f"missing {NOTEBOOK}")
    kd = out / "kernel"
    fresh_dir(kd, "kernel-metadata.json")
    shutil.copy2(NOTEBOOK, kd / NOTEBOOK.name)
    meta = {"id": kernel_id, "title": KERNEL_TITLE, "code_file": NOTEBOOK.name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [dataset_id], "competition_sources": [COMPETITION],
            "kernel_sources": [], "model_sources": []}
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return kd / NOTEBOOK.name


def execute(out: Path, nb_path: Path, timeout: int) -> dict:
    """Run the kernel copy of the notebook on CPU against the dataset snapshot (not the live repo)."""
    import nbformat
    from nbclient import NotebookClient
    if sys.platform.startswith("win"):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run = out / "local_run"
    fresh_dir(run, "run_report.json")
    os.environ.update({"NEUROCHIP_REPO": str(out / "dataset"), "NEUROCHIP_OUT": str(run / "outputs"),
                       "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    nb = nbformat.read(nb_path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(run)}}).execute()
    wall = round(time.time() - t0, 1)
    executed = run / nb_path.name.replace(".ipynb", ".executed.ipynb")
    nbformat.write(nb, executed)
    errors = [o.get("ename") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    report = {"executed_notebook": str(executed), "wall_s": wall, "errors": errors,
              "n_code_cells": sum(c.cell_type == "code" for c in nb.cells), "python": sys.version.split()[0]}
    (run / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO.parent / "_kaggle_pkg"))
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--kernel-id", default=KERNEL_ID)
    ap.add_argument("--exclude", nargs="*", default=[], metavar="TOP_LEVEL_NAME",
                    help="extra top-level repo entries to leave out of the snapshot, e.g. --exclude site")
    ap.add_argument("--execute", action="store_true", help="run the notebook locally on CPU after building")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds for --execute")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit("--out must be outside the repository (the snapshot would copy itself)")
    man = build_dataset(out, a.dataset_id, frozenset(a.exclude))
    nb = build_kernel(out, a.kernel_id, a.dataset_id)
    print(f"dataset: {out / 'dataset'}  files={man['n_files']}  MB={man['total_bytes'] / 1e6:.1f}  "
          f"commit={man['git_commit']}")
    if not man["licence_file_present"]:
        print("warning: the repository has no LICENSE file; add one before publishing (metadata says MIT)")
    print(f"kernel:  {nb.parent}  (enable_gpu=false, enable_internet=false)")
    if a.execute:
        rep = execute(out, nb, a.timeout)
        print(json.dumps(rep, indent=2))
        if rep["errors"]:
            raise SystemExit(1)
    print("nothing was uploaded; see the module docstring for the manual kaggle CLI commands")


if __name__ == "__main__":
    main()
