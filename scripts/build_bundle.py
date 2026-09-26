"""Build data_bundle/ from the processed data (run after the ingestion modules).

    python scripts/build_bundle.py [--models-from ../data/processed/models]

Writes nfa_tasks.npz (transformed trajectories), transform.json, the per-fold model weights used
by demo.py, and manifest.json with SHA-256, sources and licences for every file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO.parent
sys.path.insert(0, str(REPO / "src"))
from neurotwin.bundle import BUNDLE, save_tasks  # noqa: E402
from neurotwin.models.trajectory import load_tasks  # noqa: E402

SOURCES = {
    "nfa_tasks.npz": {
        "source": "US EPA Network Formation Assay (NFA), All_DIV_Data.Rdata via USEPA/CompTox-DNT-NFA-Refinement "
                  "(pinned commit 01adf3e); rat primary cortical neurons on 48-well MEA, DIV 5/7/9/12",
        "licence": "US Government work (public domain, 17 U.S.C. 105); data only, EPA R code not redistributed",
        "citation": "Shafer TJ et al.; Carstens KE et al. (EPA CCTE) - see docs/DATA.md"},
    "transform.json": {"source": "derived (vehicle wells only)", "licence": "MIT (this repository)"},
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-from", default=str(WS / "data/processed/models"))
    a = ap.parse_args()
    BUNDLE.mkdir(exist_ok=True)
    cache = WS / "data/processed/epa_nfa/tasks_cache.pkl"
    tasks, tr = pickle.load(open(cache, "rb")) if cache.exists() else load_tasks(WS)
    save_tasks(tasks, BUNDLE / "nfa_tasks.npz")
    (BUNDLE / "transform.json").write_text(json.dumps(tr.to_json(), indent=1), encoding="utf-8")
    mdir = BUNDLE / "models"
    mdir.mkdir(exist_ok=True)
    for p in sorted(Path(a.models_from).glob("cnp_fold*_seed*.pt")) + sorted(Path(a.models_from).glob("cnp_full_seed*.pt")):
        shutil.copy2(p, mdir / p.name)
    files = {}
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = p.relative_to(BUNDLE).as_posix()
            meta = SOURCES.get(p.name, {"source": "trained by scripts/run_trajectory_cv.py or scripts/train_full.py",
                                        "licence": "MIT (this repository)"})
            files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size, **meta}
    man = {"n_chemicals": len(tasks), "n_wells": int(sum(len(t.logc) for t in tasks)), "files": files}
    (BUNDLE / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"bundle: {len(files)} files, {sum(f['bytes'] for f in files.values())/1e6:.1f} MB")


if __name__ == "__main__":
    main()
