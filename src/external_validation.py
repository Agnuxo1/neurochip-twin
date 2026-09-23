"""External microscopy portability check on BBBC038.

This is deliberately a front-end audit, not a claim of organ-on-chip
generalization. BBBC038 provides real microscopy images and CC0 masks; the
check reuses NeuroChip Twin's ``segment`` function and reports both pixel-level
overlap and one-to-one instance scores on a deterministic sample.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .neurochip_twin import segment


SOURCE_URL = "https://bbbc.broadinstitute.org/BBBC038"
DOWNLOAD_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
INSTANCE_IOU_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 0.951, 0.05))


def _io_path(path: Path) -> Path:
    """Return a filesystem-safe path for Windows long-path extraction trees."""
    if os.name != "nt":
        return path
    absolute = path if path.is_absolute() else path.resolve()
    text = str(absolute)
    if len(text) >= 240 and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return absolute


def _normalise(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    lo, hi = np.quantile(values, [0.01, 0.998])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(_io_path(path)) as handle:
        return np.asarray(handle) > 0


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


def _instance_metrics(pred_labels: np.ndarray, truth_labels: np.ndarray) -> dict[str, float]:
    """Compute one-to-one instance F1 over DSB-style IoU thresholds.

    Pixel Dice alone can reward merging adjacent nuclei.  The Data Science
    Bowl 2018 evaluation instead matches predicted and reference objects at
    multiple IoU thresholds; this reports that score alongside pixel metrics.
    """
    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids != 0]
    truth_ids = np.unique(truth_labels)
    truth_ids = truth_ids[truth_ids != 0]
    if pred_labels.shape != truth_labels.shape:
        raise ValueError("Predicted and reference label images must have the same shape")

    iou = np.zeros((len(pred_ids), len(truth_ids)), dtype=float)
    if iou.size:
        pred_counts = np.bincount(pred_labels.ravel().astype(np.int64))
        truth_counts = np.bincount(truth_labels.ravel().astype(np.int64))
        pred_areas = pred_counts[pred_ids.astype(np.int64)]
        truth_areas = truth_counts[truth_ids.astype(np.int64)]
        both_foreground = (pred_labels > 0) & (truth_labels > 0)
        pred_index = np.searchsorted(pred_ids, pred_labels[both_foreground])
        truth_index = np.searchsorted(truth_ids, truth_labels[both_foreground])
        intersections = np.zeros_like(iou, dtype=np.int64)
        np.add.at(intersections, (pred_index, truth_index), 1)
        unions = pred_areas[:, None] + truth_areas[None, :] - intersections
        iou = intersections / np.maximum(unions, 1)
        matched_ious = iou[linear_sum_assignment(-iou)]
    else:
        matched_ious = np.empty(0, dtype=float)

    def at_threshold(threshold: float) -> tuple[float, float, float]:
        matches = int(np.count_nonzero(matched_ious >= threshold))
        precision = matches / max(len(pred_ids), 1)
        recall = matches / max(len(truth_ids), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return precision, recall, f1

    values = [at_threshold(threshold) for threshold in INSTANCE_IOU_THRESHOLDS]
    precision50, recall50, f150 = values[0]
    return {
        "instance_precision_iou50": precision50,
        "instance_recall_iou50": recall50,
        "instance_f1_iou50": f150,
        "instance_f1_mean_iou50_95": float(np.mean([value[2] for value in values])),
    }


def _evaluate_case(
    image_path: Path,
    threshold_scale: float,
    min_area: int,
    opening_size: int,
    closing_size: int,
    method: str = "threshold_cc",
    min_distance: int = 5,
    min_peak_height: float = 2.0,
) -> dict[str, float | int | str]:
    case_dir = image_path.parent.parent
    mask_paths = sorted((case_dir / "masks").glob("*.png"))
    with Image.open(_io_path(image_path)) as handle:
        raw = np.asarray(handle)
    truth_labels = np.zeros(raw.shape[:2], dtype=np.int32)
    for object_id, mask_path in enumerate(mask_paths, start=1):
        mask = _read_mask(mask_path)
        truth_labels[mask & (truth_labels == 0)] = object_id
    truth = truth_labels > 0
    with Image.open(_io_path(image_path)) as handle:
        image = _normalise(np.asarray(handle.convert("L")))
    if method == "threshold_cc":
        labels, objects = segment(
            image,
            threshold_scale=threshold_scale,
            min_area=min_area,
            opening_size=opening_size,
            closing_size=closing_size,
        )
    elif method == "distance_watershed":
        labels, objects = segment(
            image,
            threshold_scale=threshold_scale,
            min_area=min_area,
            opening_size=opening_size,
            closing_size=closing_size,
            method="distance_watershed",
            min_distance=min_distance,
            min_peak_height=min_peak_height,
        )
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
    metrics = _pixel_metrics(labels > 0, truth)
    instance_metrics = _instance_metrics(labels, truth_labels)
    return {
        "case_id": case_dir.name,
        "image": image_path.name,
        "ground_truth_objects": len(mask_paths),
        "predicted_objects": len(objects),
        "count_abs_error": abs(len(objects) - len(mask_paths)),
        **metrics,
        **instance_metrics,
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
    exclude_case_ids: set[str] | None = None,
    method: str = "threshold_cc",
    min_distance: int = 5,
    min_peak_height: float = 2.0,
    case_split: dict[str, object] | None = None,
) -> dict[str, object]:
    excluded = {str(case_id) for case_id in (exclude_case_ids or set())}
    all_cases = sorted(root.glob("*/images/*.png"))
    cases_by_id = {path.parent.parent.name: path for path in all_cases}
    cases = [path for path in all_cases if path.parent.parent.name not in excluded]
    if not cases:
        raise FileNotFoundError(f"No BBBC038 cases found below {root}")

    if case_split is not None:
        calibration_ids = [str(value) for value in case_split.get("calibration_case_ids", [])]
        evaluation_ids = [str(value) for value in case_split.get("evaluation_case_ids", [])]
        requested_ids = calibration_ids + evaluation_ids
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("Explicit split case IDs must be unique and disjoint")
        if not calibration_ids or not evaluation_ids:
            raise ValueError("Explicit split must include calibration and evaluation cases")
        missing_ids = sorted(set(requested_ids) - set(cases_by_id))
        if missing_ids:
            raise ValueError(f"Explicit split contains unknown case IDs: {missing_ids[:5]}")
        excluded_split_ids = sorted(set(requested_ids).intersection(excluded))
        if excluded_split_ids:
            raise ValueError(f"Explicit split includes excluded case IDs: {excluded_split_ids[:5]}")
        calibration_cases = [cases_by_id[case_id] for case_id in calibration_ids]
        evaluation_cases = [cases_by_id[case_id] for case_id in evaluation_ids]
        chosen = calibration_cases + evaluation_cases
        calibration_count = len(calibration_cases)
        split_source = "explicit_case_ids"
    else:
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(cases, size=min(samples, len(cases)), replace=False).tolist(), key=str)
        if calibrate and len(chosen) >= 12:
            calibration_count = max(8, len(chosen) // 3)
            calibration_cases, evaluation_cases = chosen[:calibration_count], chosen[calibration_count:]
        else:
            calibration_cases, evaluation_cases = [], chosen
            calibration_count = 0
        split_source = "seeded_random_sample"

    if calibrate and calibration_cases:
        if len(calibration_cases) < 8 or not evaluation_cases:
            raise ValueError("Calibration needs at least 8 cases and a non-empty evaluation split")
        if method == "threshold_cc":
            # Freeze morphology and search a compact local grid over threshold
            # and size, avoiding an unnecessarily broad repeated image sweep.
            candidates = [
                (scale, area, min_distance, min_peak_height)
                for scale in [0.18, 0.24, 0.30, 0.35, 0.40]
                for area in [2, 4, 8]
            ]
        elif method == "distance_watershed":
            # Hold foreground parameters fixed and calibrate only the watershed
            # seed spacing/height on the calibration images.
            candidates = [
                (threshold_scale, min_area, distance, height)
                for distance in [3, 5, 7, 9]
                for height in [1.5, 2.0, 2.5]
            ]
        else:
            raise ValueError(f"Unsupported segmentation method: {method}")
        scores = []
        for scale, area, distance, height in candidates:
            candidate_rows = [
                _evaluate_case(
                    path,
                    scale,
                    area,
                    opening_size,
                    closing_size,
                    method,
                    distance,
                    height,
                )
                for path in calibration_cases
            ]
            # Select against the instance-level objective, not pixel Dice,
            # which can prefer merging adjacent objects.
            scores.append((float(np.mean([row["instance_f1_mean_iou50_95"] for row in candidate_rows])), scale, area, distance, height))
        _, threshold_scale, min_area, tuned_distance, tuned_height = max(scores, key=lambda item: item[0])
        if method == "distance_watershed":
            min_distance, min_peak_height = int(tuned_distance), float(tuned_height)
    rows = [
        _evaluate_case(
            path,
            threshold_scale,
            min_area,
            opening_size,
            closing_size,
            method,
            min_distance,
            min_peak_height,
        )
        for path in evaluation_cases
    ]
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
        "calibration_case_ids": [path.parent.parent.name for path in calibration_cases],
        "evaluation_case_ids": [path.parent.parent.name for path in evaluation_cases],
        "dataset_case_count": int(len(all_cases)),
        "excluded_case_count": int(len(excluded.intersection({path.parent.parent.name for path in all_cases}))),
        "excluded_case_ids": sorted(excluded.intersection({path.parent.parent.name for path in all_cases})),
        "calibration_samples": calibration_count,
        "evaluation_samples": int(len(frame)),
        "calibrated": calibrate and bool(calibration_cases),
        "split_source": split_source,
        "segmentation_method": method,
        "seed": seed,
        "threshold_scale": threshold_scale,
        "min_area": min_area,
        "opening_size": opening_size,
        "closing_size": closing_size,
        "min_distance": min_distance,
        "min_peak_height": min_peak_height,
        "selection_metric": "mean per-image instance F1 averaged over IoU thresholds 0.50 to 0.95 in 0.05 increments",
        "metrics": {column: {"mean": float(numeric[column].mean()), "std": float(numeric[column].std(ddof=1)) if len(frame) > 1 else 0.0} for column in ["iou", "dice", "precision", "recall", "count_abs_error", "instance_precision_iou50", "instance_recall_iou50", "instance_f1_iou50", "instance_f1_mean_iou50_95"]},
        "interpretation": "Real microscopy front-end evidence; not organ-on-chip validation, response prediction, or clinical performance.",
    }
    (out_dir / "bbbc038_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BBBC038 external segmentation audit")
    parser.add_argument("--root", type=Path, required=True, help="Extracted stage1_train directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/external_validation"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold-scale", type=float, default=0.35)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--opening-size", type=int, default=2)
    parser.add_argument("--closing-size", type=int, default=3)
    parser.add_argument("--exclude-cases", type=Path, help="CSV with a case_id column or newline-separated case IDs to keep out of calibration/evaluation")
    parser.add_argument("--split-file", type=Path, help="JSON with explicit calibration_case_ids and evaluation_case_ids")
    parser.add_argument("--method", choices=["threshold_cc", "distance_watershed"], default="threshold_cc")
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--min-peak-height", type=float, default=2.0)
    args = parser.parse_args()
    excluded: set[str] = set()
    if args.exclude_cases:
        if args.exclude_cases.suffix.lower() == ".csv":
            excluded = set(pd.read_csv(args.exclude_cases, usecols=["case_id"])["case_id"].astype(str))
        else:
            excluded = {line.strip() for line in args.exclude_cases.read_text(encoding="utf-8").splitlines() if line.strip()}
    case_split = json.loads(args.split_file.read_text(encoding="utf-8")) if args.split_file else None
    seed = args.seed if args.seed is not None else int((case_split or {}).get("split_seed", 42))
    print(json.dumps(run_external_validation(
        args.root,
        args.out,
        samples=args.samples,
        seed=seed,
        threshold_scale=args.threshold_scale,
        min_area=args.min_area,
        calibrate=args.calibrate,
        opening_size=args.opening_size,
        closing_size=args.closing_size,
        exclude_case_ids=excluded,
        method=args.method,
        min_distance=args.min_distance,
        min_peak_height=args.min_peak_height,
        case_split=case_split,
    ), indent=2))


if __name__ == "__main__":
    main()
