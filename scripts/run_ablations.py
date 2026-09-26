"""R9: ablations of NeuroTrajectory on the outer folds (same designs and metric as R2).

For every outer fold, starting from the configuration chosen by nested selection for that fold
(results/<summary>.json), one component is removed and the model is retrained with seed 0 on the same
outer training set:
  no_interp_decoder  : decoder without the interpolation-informed residual path (plain CNP)
  no_dose_attention  : no dose-local attention path (global chemical summary only)
Reference = the seed-0 model of the full configuration (already trained, same seed and data), so each
comparison changes exactly one factor. The value of the 3-seed ensemble is reported by comparing the
ensemble with its seed-0 member. Metric: curve MAE at k = 1, 2, 3; paired bootstrap over chemicals of
(ablated - full), positive = the component helps.

Usage: python repo/scripts/run_ablations.py --summary trajectory_cv_v2.json --models-dir data/processed/models_v2 [--folds 0 1]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo/src"))
sys.path.insert(0, str(ROOT / "repo/scripts"))
from neurotwin.models.cnp import TrainConfig, load_model, predict, train_cnp  # noqa: E402
from neurotwin.models.trajectory import level_means, split_context  # noqa: E402
from run_trajectory_cv import designs, paired_boot  # noqa: E402

KS = [1, 2, 3]
ABL = {"no_interp_decoder": {"use_interp": False}, "no_dose_attention": {"use_attention": False}}


def curve_mae(models, t, ic, it, dev):
    lv = np.unique(t.logc[it])
    mu = np.mean([predict(m, t, ic, lv, device=dev)[0] for m in models], 0)
    _, obs, ok = level_means(t, it)
    return float(np.abs(mu - obs)[ok].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="trajectory_cv_v2.json")
    ap.add_argument("--models-dir", default=str(ROOT / "data/processed/models_v2"))
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--out", default=str(ROOT / "repo/results"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    summ = json.loads((Path(a.out) / a.summary).read_text(encoding="utf-8"))
    folds = a.folds if a.folds else sorted(int(f) for f in summ["nested_selection"])
    parts = Path(a.out) / "ablation_parts"
    parts.mkdir(exist_ok=True)
    mdir = Path(a.models_dir)
    for f in folds:
        cfg = summ["nested_selection"][str(f)]["chosen"]
        train = [t for t in tasks if t.fold != f]
        test = [t for t in tasks if t.fold == f]
        full0, _ = load_model(mdir / f"cnp_fold{f}_seed0.pt", dev)
        ens = [load_model(p, dev)[0] for p in sorted(mdir.glob(f"cnp_fold{f}_seed*.pt"))]
        abl_models = {}
        for name, change in ABL.items():
            c = {**cfg, **change}
            abl_models[name] = train_cnp(train, TrainConfig(steps=c["steps"], likelihood=c["likelihood"],
                                                            use_interp=c.get("use_interp", True), n_freq=c.get("n_freq", 8),
                                                            use_attention=c.get("use_attention", True), seed=0), device=dev)
        rows = []
        for t in test:
            for k in KS:
                if k >= len(t.levels):
                    continue
                acc = {"full_seed0": [], "ensemble": [], **{n: [] for n in ABL}}
                for ctx in designs(t, k):
                    ic, it = split_context(t, ctx)
                    acc["full_seed0"].append(curve_mae([full0], t, ic, it, dev))
                    acc["ensemble"].append(curve_mae(ens, t, ic, it, dev))
                    for n, m in abl_models.items():
                        acc[n].append(curve_mae([m], t, ic, it, dev))
                rows.append({"chemical": t.chem, "fold": f, "k": k, **{n: float(np.mean(v)) for n, v in acc.items()}})
        (parts / f"rows_f{f}.json").write_text(json.dumps({"cfg": cfg, "rows": rows}), encoding="utf-8")
        print(f"fold {f} done", flush=True)
    files = sorted(parts.glob("rows_f*.json"))
    import pandas as pd
    df = pd.DataFrame([r for fp in files for r in json.loads(fp.read_text(encoding="utf-8"))["rows"]])
    out = {"description": __doc__.strip().splitlines()[0], "protocol": __doc__.strip(), "folds": sorted(df.fold.unique().tolist()),
           "n_chemicals": int(df.chemical.nunique()), "by_k": {}}
    for k in KS:
        d = df[df.k == k]
        out["by_k"][str(k)] = {
            "mean_curve_mae": {c: round(float(d[c].mean()), 4) for c in ["full_seed0", "ensemble", *ABL]},
            **{f"{n}_minus_full": paired_boot(d[n], d["full_seed0"]) for n in ABL},
            "single_seed_minus_ensemble": paired_boot(d["full_seed0"], d["ensemble"])}
        print(k, out["by_k"][str(k)]["mean_curve_mae"], flush=True)
    (Path(a.out) / "r9_ablations.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote r9_ablations.json")


if __name__ == "__main__":
    main()
