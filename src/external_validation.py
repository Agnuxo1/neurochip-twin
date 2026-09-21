"""External microscopy portability check on BBBC038.

This is deliberately a front-end audit, not a claim of organ-on-chip
generalization.  BBBC038 provides real microscopy images and CC0 masks; the
check reuses NeuroChip Twin's unmodified ``segment`` function and reports
pixel-level overlap plus object-count error on a deterministic sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .neurochip_twin import segment


SOURCE_URL = "https://bbbc.broadinstitute.org/BBBC038"
DOWNLOAD_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"


def _normalise(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    lo, hi = np.quantile(values, [0.01, 0.998])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def _pixel_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(union, 1),
        "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "precision": precision,
        "recall": recall,
    }


def _evaluate_case(
    image_path: Path,
    threshold_scale: float,
    min_area: int,
    opening_size: int,
    closing_size: int,
) -> dict[str, float | int | str]:
    case_dir = image_path.parent.parent
    mask_paths = sorted((case_dir / "masks").glob("*.png"))
    with Image.open(image_path) as handle:
        raw = np.asarray(handle)
    truth = np.zeros(raw.shape[:2], dtype=bool)
    for mask_path in mask_paths:
        truth |= _read_mask(mask_path)
    with Image.open(image_path) as handle:
        image = _normalise(np.asarray(handle.convert("L")))
    labels, objects = segment(
        image,
        threshold_scale=threshold_scale,
        min_area=min_area,
        opening_size=opening_size,
        closing_size=closing_size,
    )
    metrics = _pixel_metrics(labels > 0, truth)
    return {
        "case_id": case_dir.name,
        "image": image_path.name,
        "ground_truth_objects": len(mask_paths),
        "predicted_objects": len(objects),
        "count_abs_error": abs(len(objects) - len(mask_paths)),
        **metrics,
    }


def run_external_validation(
    root: Path,
    out_dir: Path,
    samples: int = 24,
    seed: int = 42,
    threshold_scale: float = 0.35,
    min_area: int = 8,
    calibrate: bool = False,
    opening_size: int = 2,
    closing_size: int = 3,
) -> dict[str, object]:
    cases = sorted(root.glob("*/images/*.png"))
    if not cases:
        raise FileNotFoundError(f"No BBBC038 cases found below {root}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(cases, size=min(samples, len(cases)), replace=False).tolist(), key=str)
    if calibrate and len(chosen) >= 12:
        calibration_count = max(8, len(chosen) // 3)
        calibration_cases, evaluation_cases = chosen[:calibration_count], chosen[calibration_count:]
        candidates = [
            (scale, area, opening, closing)
            for scale in [0.12, 0.18, 0.24, 0.30, 0.35]
            for area in [2, 4, 8, 16]
            for opening in [0, 2]
            for closing in [0, 3]
        ]
        scores = []
        for scale, area, opening, closing in candidates:
            candidate_rows = [_evaluate_case(path, scale, area, opening, closing) for path in calibration_cases]
            scores.append((float(np.mean([row["dice"] for row in candidate_rows])), scale, area, opening, closing))
        _, threshold_scale, min_area, opening_size, closing_size = max(scores, key=lambda item: item[0])
    else:
        calibration_cases, evaluation_cases = [], chosen
        calibration_count = 0
    rows = [_evaluate_case(path, threshold_scale, min_area, opening_size, closing_size) for path in evaluation_cases]
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "bbbc038_per_image.csv", index=False)
    numeric = frame.select_dtypes(include=["number"])
    summary = {
        "dataset": "BBBC038v1",
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "license": "CC0 / public domain according to the Broad BBBC038 page",
        "task": "external segmentation portability audit only",
        "sampled_cases": int(len(chosen)),
        "calibration_samples": calibration_count,
        "evaluation_samples": int(len(frame)),
        "calibrated": calibrate and bool(calibration_cases),
        "seed": seed,
        "threshold_scale": threshold_scale,
        "min_area": min_area,
        "opening_size": opening_size,
        "closing_size": closing_size,
        "metrics": {column: {"mean": float(numeric[column].mean()), "std": float(numeric[column].std(ddof=1)) if len(frame) > 1 else 0.0} for column in ["iou", "dice", "precision", "recall", "count_abs_error"]},
        "interpretation": "Real microscopy front-end evidence; not organ-on-chip validation, response prediction, or clinical performance.",
    }
    (out_dir / "bbbc038_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BBBC038 external segmentation audit")
    parser.add_argument("--root", type=Path, required=True, help="Extracted stage1_train directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/external_validation"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-scale", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--opening-size", type=int, default=2)
    parser.add_argument("--closing-size", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_external_validation(args.root, args.out, args.samples, args.seed, args.threshold_scale, args.min_area, args.calibrate, args.opening_size, args.closing_size), indent=2))


if __name__ == "__main__":
    main()
