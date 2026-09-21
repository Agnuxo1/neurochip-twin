"""Reproducible multi-seed validation for NeuroChip Twin.

This module keeps the benchmark honest: a single held-out split is useful for
smoke testing, but it is not enough evidence for a scientific claim.  The
validation runner executes the existing end-to-end pipeline for explicit
seeds, preserves each run's artifacts, and writes a compact per-seed and
aggregate report for the technical writeup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .neurochip_twin import run


CLASSIFICATION_MODELS = (
    "baseline",
    "physics_only",
    "temporal_reservoir",
    "multimodal_no_interactions",
    "multimodal_physics",
)
CLASSIFICATION_METRICS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "brier_score",
    "expected_calibration_error",
)
REGRESSION_HEADS = ("multimodal_viability", "multimodal_ic50")
REGRESSION_METRICS = ("rmse", "r2")


def _flatten_result(result: dict[str, object], seed: int) -> dict[str, float | int]:
    """Convert one pipeline result into a stable, tabular record."""
    row: dict[str, float | int] = {"seed": seed}
    for model in CLASSIFICATION_MODELS:
        metrics = result[model]
        assert isinstance(metrics, dict)
        for metric in CLASSIFICATION_METRICS:
            row[f"{model}_{metric}"] = float(metrics[metric])
    for head in REGRESSION_HEADS:
        metrics = result[head]
        assert isinstance(metrics, dict)
        for metric in REGRESSION_METRICS:
            row[f"{head}_{metric}"] = float(metrics[metric])
    row["multimodal_vs_temporal_delta_roc_auc"] = float(
        result["multimodal_vs_temporal_delta_roc_auc"]
    )
    return row


def _aggregate(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return mean, standard deviation, min and max for every metric."""
    aggregate: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column == "seed":
            continue
        values = frame[column].astype(float)
        aggregate[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def run_validation(
    out_dir: Path,
    seeds: Iterable[int] = range(10),
    n_samples: int = 180,
    split_mode: str = "stratified",
    scenario: str = "compound_specific",
) -> dict[str, object]:
    """Run the benchmark for explicit seeds and persist an audit report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("At least one validation seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Validation seeds must be unique")
    if n_samples < 32:
        raise ValueError("n_samples must be at least 32 for a holdout")

    rows: list[dict[str, float | int]] = []
    for seed in seed_values:
        result = run(
            out_dir / f"seed_{seed}",
            seed=seed,
            n_samples=n_samples,
            split_mode=split_mode,
            scenario=scenario,
        )
        rows.append(_flatten_result(result, seed))

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    frame.to_csv(out_dir / "validation_per_seed.csv", index=False)
    report = {
        "data_kind": "synthetic_organ_on_chip_proxy",
        "seeds": seed_values,
        "n_samples_per_seed": n_samples,
        "split_mode": split_mode,
        "scenario": scenario,
        "per_seed": frame.to_dict(orient="records"),
        "aggregate": _aggregate(frame),
        "interpretation": (
            "Synthetic regression-test evidence only. Replace or augment this "
            "suite with experiment/chip-level grouped splits before making a "
            "biological generalization claim."
        ),
    }
    (out_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed NeuroChip Twin validation")
    parser.add_argument("--out", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--split-mode", choices=["stratified", "grouped"], default="stratified")
    parser.add_argument("--scenario", choices=["exposure_only", "compound_specific"], default="compound_specific")
    args = parser.parse_args()
    report = run_validation(
        args.out,
        seeds=args.seeds,
        n_samples=args.samples,
        split_mode=args.split_mode,
        scenario=args.scenario,
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
