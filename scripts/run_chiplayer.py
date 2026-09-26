"""Reproduce Brewer ChipLayer R8 and simulator parameters on CPU.

Run from the workspace root: python repo/scripts/run_chiplayer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repo" / "src"))

from neurotwin.chip.propagation import write_outputs  # noqa: E402


def figures(result_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((result_dir / "r8_chiplayer.json").read_text(encoding="utf-8"))
    figdir = result_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    graphs = [g for g in result["graphs"] if g["condition"] == "NoStim"]
    pairs = ["EC-DG", "DG-CA3", "CA3-CA1", "CA1-EC"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, pair in enumerate(pairs):
        values = [next(e["directionality_d"] for e in g["edges"] if e["pair"] == pair)
                  for g in graphs]
        ax.scatter([i] * len(values), values, alpha=.65, s=28)
        ax.plot([i-.2, i+.2], [sum(values)/len(values)]*2, color="black", lw=2)
    ax.set_xticks(range(4), pairs)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", lw=.7)
    ax.set_ylabel("Observed directionality d = (FF − FB) / (FF + FB)")
    ax.set_title("Brewer NoStim: recording-level directed axon balance")
    fig.tight_layout()
    fig.savefig(figdir / "chip_direction.png", dpi=170)
    plt.close(fig)

    folds = result["cv"]["folds"]
    labels = [f"FID {f['fid']}" for f in folds]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(axes, ("ff_fraction_mae", "conduction_mae_ms"),
                                 ("FF fraction MAE", "Conduction MAE (ms)")):
        x = list(range(len(folds)))
        model = [f["scores"]["model"][metric] for f in folds]
        base = [f["scores"]["tunnel_mean"][metric] for f in folds]
        ax.scatter(base, x, marker="o", label="Tunnel mean")
        ax.scatter(model, x, marker="x", label="Pair/tunnel shrinkage")
        for y, a, b in zip(x, base, model):
            ax.plot([a, b], [y, y], color="gray", lw=.5)
        ax.set_xlabel(title)
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Leave-one-recording-out, NoStim only")
    fig.tight_layout()
    fig.savefig(figdir / "chip_cv.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "brewer")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "repo" / "results")
    args = parser.parse_args()
    compact = write_outputs(args.data_dir, args.result_dir)
    figures(args.result_dir)
    print(json.dumps({"n_recordings": compact["n_recordings"], "n_axons": compact["n_axons"],
                      "summary": compact["cv"]["summary"],
                      "comparisons": compact["cv"]["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
