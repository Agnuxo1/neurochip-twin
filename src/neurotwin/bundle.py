"""Compact, versioned data bundle shipped with the repository (runs without downloads).

`data_bundle/nfa_tasks.npz` holds the transformed EPA NFA trajectories (one task per chemical)
and `data_bundle/manifest.json` records provenance, licences and SHA-256 of every file.
Rebuild with `python scripts/build_bundle.py` after running the full data pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models.trajectory import ChemTask

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "data_bundle"


def save_tasks(tasks: list[ChemTask], path: Path) -> None:
    off = np.cumsum([0] + [len(t.logc) for t in tasks])
    np.savez_compressed(
        path,
        chem=np.array([t.chem for t in tasks]), fold=np.array([t.fold for t in tasks], np.int16),
        label=np.array([t.label for t in tasks]), offsets=off.astype(np.int64),
        logc=np.concatenate([t.logc for t in tasks]).astype(np.float32),
        y=np.concatenate([t.y for t in tasks]).astype(np.float16),
        m=np.concatenate([t.m for t in tasks]),
        plate=np.concatenate([t.plate for t in tasks]))


def load_tasks(path: Path | None = None) -> list[ChemTask]:
    z = np.load(path or BUNDLE / "nfa_tasks.npz", allow_pickle=False)
    off = z["offsets"]
    tasks = []
    for i, c in enumerate(z["chem"]):
        a, b = off[i], off[i + 1]
        tasks.append(ChemTask(chem=str(c), fold=int(z["fold"][i]), logc=z["logc"][a:b],
                              y=z["y"][a:b].astype(np.float32), m=z["m"][a:b],
                              plate=z["plate"][a:b], label=str(z["label"][i])))
    return tasks


def manifest() -> dict:
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
